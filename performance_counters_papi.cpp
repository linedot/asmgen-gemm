#include "performance_counters.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include <papi.h>

performance_counters::performance_counters(const std::string papi_event_filename, bool discard_first)
{
    /***************************************
     * Read events from configuration file *
     ***************************************/

    std::vector<std::string> papi_event_names{};

    std::ifstream event_file;
    event_file.open(papi_event_filename);
    if(!event_file.is_open())
    {
        std::cout << "Could not open event file. please create a text file called \"events\"\n"
                     " and add PAPI events you wish to measure to the file. Write one event per line.\n"
                     " You can list available events with papi_avail or papi_native_avail commands.\n";
        throw std::runtime_error("Could not open event file");
    }

    std::string event;
    while (event_file >> event)
    {
        papi_event_names.push_back(event);
    }

    event_file.close();

    performance_counters(papi_event_names, discard_first);
}

performance_counters::performance_counters(const std::vector<std::string>& papi_event_names, bool discard_first)
    : event_set(PAPI_NULL),
      named_events(papi_event_names),
      discard_first(discard_first)
{
    int papi_ret = PAPI_library_init(PAPI_VER_CURRENT);
    if(0 < papi_ret && papi_ret != PAPI_VER_CURRENT)
    {
        throw std::runtime_error("PAPI library mismatch");
    }
    if(0 > papi_ret)
    {
        throw std::runtime_error("Error initializing PAPI");
    }
    if(PAPI_LOW_LEVEL_INITED != PAPI_is_initialized())
    {
        throw std::runtime_error("PAPI not initialized");
    }

    if (PAPI_OK != PAPI_create_eventset(&event_set))
    {
        throw std::runtime_error("Can't create PAPI event set");
    }
    for(const auto& event : named_events)
    {
        if(PAPI_OK != PAPI_add_named_event(event_set, event.c_str()))
        {
            // TODO: Handle missing papi events, either deleting from named_events or more complex handling of read counter data later on
            std::cout << "Warning: Failed adding named PAPI event - results will be incorrect \"" << event << "\"\n";
        }
    }
}

performance_counters::~performance_counters()
{
    int papi_ret = PAPI_cleanup_eventset(event_set);
    if(PAPI_OK != papi_ret)
    {
        std::cout << "Failed to clean up PAPI event set: ";

        switch(papi_ret)
        {
            case PAPI_EINVAL:
                std::cout << "PAPI_EINVAL: Invalid argument.";
                break;
            case PAPI_ENOEVST:
                std::cout << "PAPI_ENOEVST: Event set doesn't exist.";
                break;
            case PAPI_EISRUN:
                std::cout << "PAPI_EISRUN: Event set is counting events.";
                break;
            case PAPI_EBUG:
                std::cout << "PAPI_EBUG: Internal error, send mail to ptools-perfapi@icl.utk.edu and complain.";
                break;
        }
        std::cout << "\n";
    }

    if(PAPI_OK != PAPI_destroy_eventset(&event_set))
    {
        std::cout << "Failed to destroy PAPI event set.\n";
    }

    PAPI_shutdown();
}


void performance_counters::tic()
{
    PAPI_reset(event_set);
    PAPI_start(event_set);
}

std::vector<std::uint64_t> performance_counters::toc()
{
    std::vector<std::uint64_t> counters(named_events.size());
    PAPI_stop(event_set, reinterpret_cast<long long int*>(counters.data()));
    return counters;
}

void performance_counters::toc_stat()
{
    std::vector<std::uint64_t> counters(named_events.size());
    PAPI_stop(event_set, reinterpret_cast<long long int*>(counters.data()));
    counter_storage.insert(counter_storage.end(), counters.begin(), counters.end());
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
        msg << "Tried using \"" << cycles_counter << "\" as PAPI event to measure overhead, but this event is not in the performance counter list!";
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

    return avg/((1.0-eff)*expected_benchmark_cycles);
}
