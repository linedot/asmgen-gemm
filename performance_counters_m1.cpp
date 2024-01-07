#include "performance_counters.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <map>

#include <dlfcn.h>
#include <pthread.h>

#define CFGWORD_EL0A32EN_MASK (0x10000)
#define CFGWORD_EL0A64EN_MASK (0x20000)
#define CFGWORD_EL1EN_MASK (0x40000)
#define CFGWORD_EL3EN_MASK (0x80000)
#define CFGWORD_ALLMODES_MASK (0xf0000)

#define CPMU_NONE 0
#define CPMU_CORE_CYCLE 0x02
#define CPMU_INST_A64 0x8c
#define CPMU_INST_BRANCH 0x8d
#define CPMU_SYNC_DC_LOAD_MISS 0xbf
#define CPMU_SYNC_DC_STORE_MISS 0xc0
#define CPMU_SYNC_DTLB_MISS 0xc1
#define CPMU_SYNC_ST_HIT_YNGR_LD 0xc4
#define CPMU_SYNC_BR_ANY_MISP 0xcb
#define CPMU_FED_IC_MISS_DEM 0xd3
#define CPMU_FED_ITLB_MISS 0xd4

#define KPC_CLASS_FIXED (0)
#define KPC_CLASS_CONFIGURABLE (1)
#define KPC_CLASS_POWER (2)
#define KPC_CLASS_RAWPMU (3)
#define KPC_CLASS_FIXED_MASK (1u << KPC_CLASS_FIXED)
#define KPC_CLASS_CONFIGURABLE_MASK (1u << KPC_CLASS_CONFIGURABLE)
#define KPC_CLASS_POWER_MASK (1u << KPC_CLASS_POWER)
#define KPC_CLASS_RAWPMU_MASK (1u << KPC_CLASS_RAWPMU)

#define COUNTERS_COUNT 10
#define CONFIG_COUNT 8
#define KPC_MASK (KPC_CLASS_CONFIGURABLE_MASK | KPC_CLASS_FIXED_MASK)

constexpr bool high_performance_cores = true;

#define KPERF_LIST                                                             \
  /*  ret, name, params */                                                     \
  F(int, kpc_get_counting, void)                                               \
  F(int, kpc_force_all_ctrs_set, int)                                          \
  F(int, kpc_set_counting, uint32_t)                                           \
  F(int, kpc_set_thread_counting, uint32_t)                                    \
  F(int, kpc_set_config, uint32_t, void *)                                     \
  F(int, kpc_get_config, uint32_t, void *)                                     \
  F(int, kpc_set_period, uint32_t, void *)                                     \
  F(int, kpc_get_period, uint32_t, void *)                                     \
  F(uint32_t, kpc_get_counter_count, uint32_t)                                 \
  F(uint32_t, kpc_get_config_count, uint32_t)                                  \
  F(int, kperf_sample_get, int *)                                              \
  F(int, kpc_get_thread_counters, int, unsigned int, void *)

#define F(ret, name, ...)                                                      \
  typedef ret name##proc(__VA_ARGS__);                                         \
  static name##proc *name;
KPERF_LIST
#undef F

uint64_t m1_counters[COUNTERS_COUNT];
uint64_t m1_config[COUNTERS_COUNT];

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
                     " and add m1 events you wish to measure to the file. Write one event per line.\n"
                     " Available events are: \n"
                     "  - CPMU_CORE_CYCLES\n"
                     "  - CPMU_CORE_CYCLE\n"
                     "  - CPMU_INST_A64\n"
                     "  - CPMU_INST_BRANCH\n"
                     "  - CPMU_SYNC_DC_LOAD_MISS\n"
                     "  - CPMU_SYNC_DC_STORE_MISS\n"
                     "  - CPMU_SYNC_DTLB_MISS\n"
                     "  - CPMU_SYNC_ST_HIT_YNGR_LD\n"
                     "  - CPMU_SYNC_BR_ANY_MISP\n"
                     "  - CPMU_FED_IC_MISS_DEM\n"
                     "  - CPMU_FED_ITLB_MISS\n";
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
    : event_set(0),
      named_events(papi_event_names),
      discard_first(discard_first)
{
    if(high_performance_cores)
    {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    }
    else
    {
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND,0);
    }

    // Open kperf
    constexpr char kperf_filename[] =
        "/System/Library/PrivateFrameworks/kperf.framework/Versions/A/kperf";
    void* kperf = dlopen(kperf_filename, RTLD_LAZY);
    if(nullptr == kperf)
    {
        throw std::runtime_error("Error opening kperf framework. Root rights?");
    }

    // Get functions
 
    #define F(ret, name, ...)                                                      \
      name = (name##proc *)(dlsym(kperf, #name));                                  \
      if (!name) {                                                                 \
        printf("%s = %p\n", #name, (void *)name);                                  \
        return;                                                                    \
      }
      KPERF_LIST
    #undef F

    // Init kpc stuff
    if(COUNTERS_COUNT != kpc_get_counter_count(KPC_MASK))
    {
        std::string msg = "Fixed counter count is not ";
        msg.append(std::to_string(COUNTERS_COUNT));
        throw std::runtime_error(msg);
    }
    if(CONFIG_COUNT != kpc_get_config_count(KPC_MASK))
    {
        std::string msg = "Fixed config count is not ";
        msg.append(std::to_string(CONFIG_COUNT));
        throw std::runtime_error(msg);
    }

    // Choose counters
    if(CONFIG_COUNT < named_events.size())
    {
        throw std::runtime_error("Too many performance counters at the same time");
    }

    // Remove duplicates
    std::sort( named_events.begin(), named_events.end() );
    named_events.erase( std::unique( named_events.begin(), named_events.end() ), named_events.end() );

    // cycles,icm,itlbm
    std::vector<std::string> group1_events = {"CPMU_CORE_CYCLE","CPMU_FED_IC_MISS_DEM","CPMU_FED_ITLB_MISS"};
    std::size_t min_group1 = 0;
    std::size_t max_group1 = 7;
    std::size_t pos_group1 = 0;

    std::vector<std::size_t> group1_exclude_positions = {};

    // br_inst,br_misp,dc_ldm,dc_stm,hit_ld,dtlb
    std::vector<std::string> group2_events = {"CPMU_INST_BRANCH","CPMU_SYNC_BR_ANY_MISP","CPMU_SYNC_DC_LOAD_MISS","CPMU_SYNC_DC_STORE_MISS","CPMU_SYNC_ST_HIT_YNGR_LD","CPMU_SYNC_DTLB_MISS"};
    std::size_t min_group2 = 3;
    std::size_t max_group2 = 5;
    std::size_t pos_group2 = 3;

    // inst
    std::vector<std::string> group3_events = {"CPMU_INST_A64"};
    std::size_t min_group3 = 5;
    std::size_t max_group3 = 5;
    std::size_t pos_group3 = 5;

    // First fill group 3 (inst)
    // Then fill group 2, adding to group 1 exclude list,
    // Then fill group 1
    //
    #define NFIND(x)\
      named_events.end() != std::find(named_events.begin(), named_events.end(), x)

    std::map<std::string,int> event_kpc_map = 
    {
        {"CPMU_CORE_CYCLE",          CPMU_CORE_CYCLE},
        {"CPMU_INST_A64",            CPMU_INST_A64},
        {"CPMU_INST_BRANCH",         CPMU_INST_BRANCH},
        {"CPMU_SYNC_DC_LOAD_MISS",   CPMU_SYNC_DC_LOAD_MISS},
        {"CPMU_SYNC_DC_STORE_MISS",  CPMU_SYNC_DC_STORE_MISS},
        {"CPMU_SYNC_DTLB_MISS",      CPMU_SYNC_DTLB_MISS},
        {"CPMU_SYNC_ST_HIT_YNGR_LD", CPMU_SYNC_ST_HIT_YNGR_LD},
        {"CPMU_SYNC_BR_ANY_MISP",    CPMU_SYNC_BR_ANY_MISP},
        {"CPMU_FED_IC_MISS_DEM",     CPMU_FED_IC_MISS_DEM},
        {"CPMU_FED_ITLB_MISS",       CPMU_FED_ITLB_MISS}
    };
    std::vector<std::string> new_named_events(COUNTERS_COUNT);
    std::fill(new_named_events.begin(), new_named_events.end(), "unused");

    if (NFIND("CPMU_INST_A64"))
    {
        m1_config[5] = event_kpc_map["CPMU_INST_A64"] | CFGWORD_EL0A64EN_MASK;
        group1_exclude_positions.push_back(5);
        max_group2 = 4;

        new_named_events[5] = "CPMU_INST_A64";
    }
    for ( auto& g2ev : group2_events)
    {
        if(NFIND(g2ev))
        {
            if(pos_group2 > max_group2)
            {
                throw std::runtime_error("Too many group2 counters (max 3 if not counting \"CPMU_INST_A64\", otherwise 2)");
            }
            m1_config[pos_group2] = event_kpc_map[g2ev];
            group1_exclude_positions.push_back(pos_group2);
            new_named_events[pos_group2] = g2ev;

            pos_group2++;
        }
    }
    for ( auto& g1ev : group1_events)
    {
        if(NFIND(g1ev))
        {
            for(;pos_group1 <= max_group1; pos_group1++)
            {
                auto& g1ex = group1_exclude_positions;
                if(g1ex.end() == std::find(g1ex.begin(), g1ex.end(), pos_group1))
                {
                    m1_config[pos_group1] = event_kpc_map[g1ev];
                    new_named_events[pos_group1] = g1ev;

                    // Don't forget - on break it's not incremented!
                    pos_group1++;
                    break;
                }
            }
            if(pos_group1 > max_group1)
            {
                throw std::runtime_error("Too many group1 counters (max 8 minus used group2+group3 counters)");
            }
        }
    }
    #undef NFIND

    named_events = new_named_events;


    // Finalize counters
    if(0 != kpc_set_config(KPC_MASK,m1_config))
    {
        throw std::runtime_error("kpc_set_config failed");
    }
    if(0 != kpc_force_all_ctrs_set(1))
    {
        throw std::runtime_error("kpc_force_all_ctrs_set failed");
    }
    if(0 != kpc_set_counting(KPC_MASK))
    {
        throw std::runtime_error("kpc_set_counting failed");
    }
    if(0 != kpc_set_thread_counting(KPC_MASK))
    {
        throw std::runtime_error("kpc_set_thread_counting failed");
    }
}

performance_counters::~performance_counters()
{
}


void performance_counters::tic()
{
    if(0 != kpc_get_thread_counters(0, COUNTERS_COUNT, m1_counters))
    {
        throw std::runtime_error("Can't kpc_get_thread_counters, are you root?");
    }
    counter_storage.insert(counter_storage.end(), m1_counters, m1_counters + COUNTERS_COUNT);    
}

std::vector<std::uint64_t> performance_counters::toc()
{
    std::vector<std::uint64_t> counters(COUNTERS_COUNT);

    if(0 != kpc_get_thread_counters(0, COUNTERS_COUNT, m1_counters))
    {
        throw std::runtime_error("Can't kpc_get_thread_counters, are you root?");
    }

    std::size_t last_counters_offset = counter_storage.size() - COUNTERS_COUNT;

    if(counter_storage.size() < last_counters_offset)
    {
        throw std::runtime_error("toc used before tic!");
    }


    for(std::size_t i = 0; i < COUNTERS_COUNT; i++)
    {
        counters[i] = m1_counters[i] - counter_storage[last_counters_offset+i];
    }
    reset_counter_storage();
    return counters;
}

void performance_counters::toc_stat()
{
    std::vector<std::uint64_t> counters(named_events.size());

    if(0 != kpc_get_thread_counters(0, COUNTERS_COUNT, m1_counters))
    {
        throw std::runtime_error("Can't kpc_get_thread_counters, are you root?");
    }

    std::size_t last_counters_offset = counter_storage.size() - COUNTERS_COUNT;

    if(counter_storage.size() < last_counters_offset)
    {
        throw std::runtime_error("toc used before tic!");
    }


    for(std::size_t i = 0; i < COUNTERS_COUNT; i++)
    {
        counter_storage[last_counters_offset+i] = m1_counters[i] - counter_storage[last_counters_offset+i];
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
    std::size_t event_count = COUNTERS_COUNT;
    std::size_t measurement_count = counter_storage.size() / event_count;

    std::size_t start_with = 1;
    if(!discard_first)
    {
        start_with = 0;
    }

    for (std::size_t i = 0; i < COUNTERS_COUNT-2; i++)
    {
        auto& event = named_events[i];
        if("unused" == event) continue;
        std::uint64_t min = std::numeric_limits<std::uint64_t>::max(),
                      max = 0,
                      avg = 0;


        for(std::size_t j = start_with; j < measurement_count; j++)
        {

            std::uint64_t counter = counter_storage[event_count*j+i+2];

            avg += counter;
            min = std::min(min,counter);
            max = std::max(max,counter);
        }
        avg /= measurement_count-start_with;

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
