#include <vector>
#include <cstdint>
#include <string>
#include <tuple>

class performance_counters
{
public:
    performance_counters(const std::vector<std::string>& papi_event_names, bool discard_first = true);
    performance_counters(const std::string papi_event_filename, bool discard_first = true);

    performance_counters() = delete;
    performance_counters(const performance_counters&) = delete;
    ~performance_counters();

    void tic();
    std::vector<std::uint64_t> toc();
    void toc_stat();

    const std::vector<std::string> get_names();

    std::vector<std::tuple<std::string,std::uint64_t,std::uint64_t,std::uint64_t>>
        get_counter_statistics();

    void reset_counter_storage();

    std::uint64_t get_iterations_for_efficiency(
            double eff = 0.99,
            std::uint64_t expected_benchmark_cycles = 1,
            std::uint64_t overhead_measurements = 1000,
            const std::string cycles_counter = "CPU_CYCLES");
private:
    int event_set;
    std::vector<std::string> named_events;
    std::vector<std::uint64_t> counter_storage;
    bool discard_first;

    std::uint64_t overhead_cycles;

    void* impl_dependent_data;
};
