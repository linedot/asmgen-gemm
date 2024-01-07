#include "performance_counters.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <unordered_map>
#include <utility> // std::pair
#include <string_view>

#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>

#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

template <typename key_type, typename value_type, std::size_t size>
struct const_map 
{
    std::array<std::pair<key_type, value_type>, size> data;

    constexpr const_map(const std::array<std::pair<key_type, value_type>, size> data)
        : data(data)
    {
    }

    [[nodiscard]] constexpr value_type at(const key_type &key) const 
    {
        const auto itr =
            std::find_if(begin(data), end(data),
                    [&key](const auto &v) { return v.first == key; });
        if (itr != end(data)) 
        {
            return itr->second;
        } 
        else 
        {
            throw std::range_error("Not Found");
        }
    }

};


static constexpr perf_event_attr get_cache_attr(std::uint64_t cache_config)
{
    return perf_event_attr
    {
        .type           = PERF_TYPE_HW_CACHE,
        .size           = sizeof(perf_event_attr),
        .config         = cache_config,
        .read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_ID,
        .disabled       = 1,
        .exclude_kernel = 1,
        .exclude_hv     = 1,
    };
}

constexpr auto make_cache_config(std::uint32_t cache, std::uint32_t op, std::uint32_t result)
    -> std::int32_t
{
    //NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers,hicpp-signed-bitwise)
    return (cache | (op << 8) | (result <<16));
};

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CACHE_CONFIG(CACHE,OP,RESULT)\
    make_cache_config(PERF_COUNT_HW_CACHE_ ##CACHE,\
            PERF_COUNT_HW_CACHE_OP_ ##OP,\
            PERF_COUNT_HW_CACHE_RESULT_ ##RESULT)

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CCONF_PAIR(CACHE,OP,RESULT)\
        {#CACHE "_" #OP "_" #RESULT ## sv, get_cache_attr(CACHE_CONFIG(CACHE,OP,RESULT))}

using namespace std::literals::string_view_literals;

typedef std::pair<std::string_view, perf_event_attr> sva;
static constexpr std::array supported_events =
{
    sva{"CYCLES"sv,
     {
         .type           = PERF_TYPE_HARDWARE,
         .size           = sizeof(perf_event_attr),
         .config         = PERF_COUNT_HW_CPU_CYCLES,
         .read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_ID,
         .disabled       = 1,
         .exclude_kernel = 1,
         .exclude_hv     = 1,
     }},
    sva{"INSTRUCTIONS"sv,
     {
         .type           = PERF_TYPE_HARDWARE,
         .size           = sizeof(perf_event_attr),
         .config         = PERF_COUNT_HW_INSTRUCTIONS,
         .read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_ID,
         .disabled       = 1,
         .exclude_kernel = 1,
         .exclude_hv     = 1,
     }},
    sva CCONF_PAIR(L1D,READ,ACCESS),
    sva CCONF_PAIR(L1D,READ,MISS),
    sva CCONF_PAIR(L1D,WRITE,ACCESS),
    sva CCONF_PAIR(L1D,WRITE,MISS),
    sva CCONF_PAIR(LL,READ,ACCESS),
    sva CCONF_PAIR(LL,READ,MISS),
    sva CCONF_PAIR(LL,WRITE,ACCESS),
    sva CCONF_PAIR(LL,WRITE,MISS)
};

#undef CCONF_PAIR
#undef CACHE_CONFIG

perf_event_attr lookup_event(const std::string_view sv)
{
    static constexpr auto map =
        const_map<std::string_view, perf_event_attr, supported_events.size()>{supported_events};

    return map.at(sv);
}

struct perf_specific_data
{
    std::size_t read_format_size;
    std::size_t provided_counter_count;
    std::size_t current_offset;

    std::vector<perf_event_attr> event_attrs;
    std::vector<int> event_fds;
    std::vector<int> event_ids;
};


performance_counters::performance_counters(const std::string event_filename, bool discard_first)
{
    /***************************************
     * Read events from configuration file *
     ***************************************/

    std::vector<std::string> event_names{};

    std::ifstream event_file;
    event_file.open(event_filename);
    if(!event_file.is_open())
    {
        std::cout << "Could not open event file. please create a text file called \"events\"\n"
                     " and add the events you wish to measure to the file. Write one event per line.\n"
                     " You can list available events with 'perf list'.\n";
        throw std::runtime_error("Could not open event file");
    }

    std::string event;
    while (event_file >> event)
    {
        event_names.push_back(event);
    }

    event_file.close();

    performance_counters(event_names, discard_first);
}

performance_counters::performance_counters(const std::vector<std::string>& event_names, bool discard_first)
    : event_set(0),
      named_events(event_names),
      discard_first(discard_first),
      impl_dependent_data(nullptr)
{
    if(event_names.empty())
    {
        return;
    }

    auto event_count = event_names.size();

    auto* perf_data = new perf_specific_data{};

    impl_dependent_data = reinterpret_cast<void*>(perf_data);
    perf_data->event_attrs.resize(event_count);
    perf_data->event_ids.resize(event_count);
    perf_data->event_fds.resize(event_count);
    // 2x64 bit for each event + 1 for um... I forgot
    perf_data->read_format_size = (2*event_count+1)*sizeof(std::uint64_t);
    perf_data->provided_counter_count = event_count;

    auto& fds = perf_data->event_fds;
    auto& ids = perf_data->event_ids;


    // Since this qualifies as "stupidly clever" (or dangerous), some explanation:
    // The first event is the "leader" and when creating the fd for it,
    // the leader fd is set to -1. subsequently using the first fd when
    // creating the rest of the events specifies the correct leader fd
    auto& leader_fd = fds.front();
    leader_fd = -1;

    for(std::size_t i = 0; i < event_count; i++)
    {
        auto attr = lookup_event(event_names[i]);
        fds.at(i) = syscall(__NR_perf_event_open,
                &attr, 0, -1, leader_fd, 0);

        if(-1 == fds.at(i))
        {
            std::stringstream errmsg;
            errmsg << "Failed to add perf_event " << ((0==i) ? "leader":"counter")
                   << " \"" << event_names[i] << "\"";

            delete perf_data;
            throw std::runtime_error(errmsg.str());

        }
        auto ret = ioctl(fds.at(i), PERF_EVENT_IOC_ID, &ids[i]);
        if(-1 == ret)
        {
            std::stringstream errmsg;
            errmsg << "Failed to get ID of perf_event " << ((0==i) ? "leader":"counter")
                   << " \"" << event_names[i] << "\": " << strerror(errno);

            delete perf_data;
            throw std::runtime_error(errmsg.str());
        }
    }

    auto ret = ioctl(leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    if(-1 == ret)
    {
        std::stringstream errmsg;
        errmsg << "PERF_EVENT_IOC_RESET ioctl failed: " << strerror(errno);
        delete perf_data;
        throw std::runtime_error(errmsg.str());
    }
    ret = ioctl(leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    if(-1 == ret)
    {
        std::stringstream errmsg;
        errmsg << "PERF_EVENT_IOC_ENABLE ioctl failed: " << strerror(errno);
        delete perf_data;
        throw std::runtime_error(errmsg.str());
    }
}

performance_counters::~performance_counters()
{
    if(nullptr == impl_dependent_data)
        return;
    auto* perf_data = reinterpret_cast<perf_specific_data*>(impl_dependent_data);

    auto& leader_fd = perf_data->event_fds.front();

    auto ret = ioctl(leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    if(-1 == ret)
    {
        std::stringstream errmsg;
        errmsg << "PERF_EVENT_IOC_DISABLE ioctl failed: " << strerror(errno);
        delete perf_data;
    }

    delete perf_data;
    impl_dependent_data = nullptr;
}

//TODO: dedupe code

void performance_counters::tic()
{
    auto* perf_data = reinterpret_cast<perf_specific_data*>(impl_dependent_data);
    if(nullptr == perf_data)
        return;

    auto& leader_fd = perf_data->event_fds.front();

    std::vector<std::uint64_t> counter_data(perf_data->read_format_size);
    
    if(perf_data->read_format_size != 
            read(leader_fd, counter_data.data(), perf_data->read_format_size))
    {
        throw std::runtime_error("Failed to read perf_event counter data");
    }

    std::vector<std::uint64_t> counters(perf_data->provided_counter_count);

    for(std::size_t i = 0; i < perf_data->provided_counter_count; i++)
    {
        // skip 1 element in the beginning of the data, then
        // 0th element: counter value
        // 1st element: counter id
        constexpr size_t val_off = 1;
        //constexpr size_t id_off = 2;
        counters[i] = counter_data[i*2+val_off];
    }

    counter_storage.insert(std::end(counter_storage),
            std::begin(counters), std::end(counters));
} 

std::vector<std::uint64_t> performance_counters::toc()
{
    auto* perf_data = reinterpret_cast<perf_specific_data*>(impl_dependent_data);
    if(nullptr == perf_data)
        return {};

    auto& leader_fd = perf_data->event_fds.front();

    auto event_count = perf_data->provided_counter_count;

    std::vector<std::uint64_t> counter_data(perf_data->read_format_size);
    
    if(perf_data->read_format_size != 
            read(leader_fd, counter_data.data(), perf_data->read_format_size))
    {
        throw std::runtime_error("Failed to read perf_event counter data");
    }

    std::vector<std::uint64_t> counters(event_count);

    for(std::size_t i = 0; i < event_count; i++)
    {
        // skip 1 element in the beginning of the data, then
        // 0th element: counter value
        // 1st element: counter id
        constexpr size_t val_off = 1;
        //constexpr size_t id_off = 2;
        counters[i] = counter_data[i*2+val_off];
    }

    for(std::size_t i = 0; i < perf_data->provided_counter_count; i++)
    {
        std::transform(std::begin(counters),std::end(counters),
                       std::end(counter_storage) - event_count,
                       std::begin(counters),
                       std::minus<std::uint64_t>{});
    }
    counter_storage.clear();

    return counters;
}

void performance_counters::toc_stat()
{
    auto* perf_data = reinterpret_cast<perf_specific_data*>(impl_dependent_data);
    if(nullptr == perf_data)
        return;

    auto& leader_fd = perf_data->event_fds.front();

    auto event_count = perf_data->provided_counter_count;

    std::vector<std::uint64_t> counter_data(perf_data->read_format_size);
    
    if(perf_data->read_format_size != 
            read(leader_fd, counter_data.data(), perf_data->read_format_size))
    {
        throw std::runtime_error("Failed to read perf_event counter data");
    }

    std::vector<std::uint64_t> counters(event_count);

    for(std::size_t i = 0; i < event_count; i++)
    {
        // skip 1 element in the beginning of the data, then
        // 0th element: counter value
        // 1st element: counter id
        constexpr size_t val_off = 1;
        //constexpr size_t id_off = 2;
        counters[i] = counter_data[i*2+val_off];
    }

    for(std::size_t i = 0; i < event_count; i++)
    {
        std::transform(std::begin(counters),std::end(counters),
                       std::end(counter_storage) - event_count,
                       std::end(counter_storage) - event_count,
                       std::minus<std::uint64_t>{});
    }
}

const std::vector<std::string> performance_counters::get_names()
{
    return named_events;
}

std::vector<std::tuple<std::string,std::uint64_t,std::uint64_t,std::uint64_t>>
    performance_counters::get_counter_statistics()
{
    std::vector<std::tuple<std::string,std::uint64_t,std::uint64_t,std::uint64_t>> results;
    std::size_t event_count = named_events.size();
    std::size_t measurement_count = counter_storage.size() / event_count;

    std::size_t start_with = 1;
    if(!discard_first)
    {
        start_with = 0;
    }

    std::uint64_t offset = 0;
    for (const auto& event : named_events)
    {
        std::uint64_t min = std::numeric_limits<std::uint64_t>::max(),
                      max = 0,
                      avg = 0;


        for(std::size_t i = start_with; i < measurement_count; i++)
        {
            std::uint64_t counter = counter_storage[event_count*i+offset];

            avg += counter;
            min = std::min(min,counter);
            max = std::max(max,counter);
        }
        avg /= measurement_count-start_with;

        offset++;

        results.push_back({event,min,avg,max});
    }
    return results;
}

void performance_counters::reset_counter_storage()
{
    counter_storage.clear();
}


std::uint64_t performance_counters::get_iterations_for_efficiency(
        double eff,
        std::uint64_t expected_benchmark_cycles,
        std::uint64_t overhead_measurements,
        const std::string cycles_counter)
{
    auto cycles_iter = std::find(named_events.begin(), named_events.end(), cycles_counter);
    if (named_events.end() == cycles_iter)
    {
        std::stringstream msg;
        msg << "Tried using \"" << cycles_counter << "\" as perf_event event to measure overhead, but this event is not in the performance counter list!";
        throw std::runtime_error(msg.str());
    }
    std::size_t cycles_offset = std::distance(named_events.begin(),cycles_iter);

    std::size_t start_with = 1;
    if(!discard_first)
    {
        start_with = 0;
    }
    for( std::size_t i = start_with; i < overhead_measurements; i++)
    {
        tic();
        toc_stat();
    }
    auto [name,min,avg,max] = get_counter_statistics()[cycles_offset];
    reset_counter_storage();

    std::uint64_t iters = avg/((1.0-eff)*expected_benchmark_cycles);
    /*
    if (iters < 1000)
        iters = 1000;
    else if(iters > 50000)
        iters = 50000;*/

    return iters;
}
