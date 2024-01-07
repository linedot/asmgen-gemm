#include "performance_counters.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

std::uint64_t read_cycles()
{
    std::uint64_t cycles = 0;
#if defined(__x86_64__)
    std::uint32_t hi=0,lo=0;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    cycles = (static_cast<std::uint64_t>(lo))|(static_cast<std::uint64_t>(hi)<<32);
#elif defined(__aarch64__)
    __asm__ volatile("mrs %[cycles], cntvct_el0" : [cycles] "=r" (cycles));    
#elif defined(__riscv)
    __asm__ volatile("rdcycle %[cycles]" : [cycles] "=r" (cycles));
#else
#error simple performance counters not supported on this architecture
#endif
    return cycles;
}


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
                     " and add events you wish to measure in the file. This implementation only\n"
                     " supports \"CYCLES\"\n";
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
      discard_first(discard_first)
{
    for(const auto& event : named_events)
    {
        if("CYCLES" != event)
        {
            std::cout << "Warning: event " << event << " not supported with simple performance_counters implementation\n";
        }
    }
}

performance_counters::~performance_counters()
{
}


void performance_counters::tic()
{
    auto size = named_events.size();
    std::vector<std::uint64_t> counters(size);
    for(std::uint64_t i = 0; i < size; i++)
    {
        counters[i] = read_cycles();
    }
    counter_storage.insert(counter_storage.end(), counters.begin(), counters.end());
}

std::vector<std::uint64_t> performance_counters::toc()
{
    auto size = named_events.size();
    std::vector<std::uint64_t> counters(size);
    for(std::uint64_t i = 0; i < size; i++)
    {
        counters[i] = read_cycles() - counter_storage[i];
    }
    counter_storage.clear();
    return counters;
}

void performance_counters::toc_stat()
{
    auto size = named_events.size();
    auto stored_size = counter_storage.size();
    for(std::uint64_t i = 0; i < size; i++)
    {
        counter_storage[stored_size-size+i] = read_cycles() - counter_storage[stored_size-size+i];
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

    if(0 == avg)
    {
        avg = 1;
    }

    return avg/((1.0-eff)*expected_benchmark_cycles);
}
