#!/bin/bash

usage() { echo "Usage: $0 -T <avx_fma_128|avx_fma_256|avx512|neon|sve|rvv|rvv071> [-m <mr>] [-m <nr>] [-U <unroll>] [-V <max_vregs>] [-M <samedata|l1|contiguous>] [-t <double|single>] [-B <auto|dist1_boff|dist1_inc|fmaidx|fmavf|noload>] [-A <auto|preload|postload>] [-X]" 1>&2; exit 1; }

mr=16
nr=30
max_vregs=16
mem_use="samedata"
data_type="double"
bvec_strat="auto"
avec_strat="auto"
unroll=0
noexec=0
nocompile=0
while getopts "T:m:n:V:U:M:t:B:A:XC" o; do
    case "${o}" in
        T)
            T=${OPTARG}
            ((T == "avx_fma_128" || T == "avx_fma_256" || T == "avx512" || T == "neon" || T == "sve" || T == "rvv" || T == "rvv071")) || usage
            ;;
        m)
            mr=${OPTARG}
            ;;
        n)
            nr=${OPTARG}
            ;;
        V)
            max_vregs=${OPTARG}
            ;;
        U)
            unroll=${OPTARG}
            ;;
        M)
            mem_use=${OPTARG}
            ;;
        t)
            data_type=${OPTARG}
            ;;
        B)
            bvec_strat=${OPTARG}
            ;;
        A)
            avec_strat=${OPTARG}
            ;;
        X)
            noexec=1
            ;;
        C)
            nocompile=1
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${T}" ] ; then
    echo T is unset
    usage
fi

CXX=${CXX:-"g++"}
CFLAGS=""

case "${T}" in
    avx_fma_128)
        echo "Using AVX FMA (128 bit):"
        CFLAGS="-mavx -mfma"
        ;;
    avx_fma_256)
        echo "Using AVX2 FMA (256 bit):"
        CFLAGS="-mavx -mfma"
        ;;
    avx512)
        echo "Using AVX512 FMA (512 bit):"
        CFLAGS="-mavx512f -mfma"
        ;;
    neon)
        echo "Using NEON FMA (128 bit):"
        CFLAGS="-march=armv8-a+simd"
        ;;
    sve)
        echo "Using SVE FMA (<platform-defined> bit):"
        CFLAGS="-march=armv8-a+sve"
        ;;
    rvv)
        echo "Using RVV 1.0 FMA (<platform-defined> bit):"
        CFLAGS="-march=rv64gcv1p0"
        ;;
    rvv071)
        echo "Using RVV 0.7.1 FMA (<platform-defined> bit):"
        #CFLAGS="-march=rv64gcv0p7 -menable-experimental-extensions"
        CFLAGS="-mepi"
        ;;
esac

if [[ "${T}" != "avx512"* ]] && [[ "${T}" != "avx_fma_256"* ]] && [[ "${T}" != "avx_fma_128"* ]] && [[ "${T}" != "sve"* ]] && [[ "${T}" != "neon"* ]] && [[ "${T}" != "rvv"* ]] && [[ "${T}" != "rvv071"* ]]; then
    echo "-T ${T} unsupported"
    exit -1
fi

PCTR_FILES=""

if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$M1PERF_I_KNOW_WHAT_IM_DOING" == "1" ]]; then
        PCTR_FILES="performance_counters_m1.cpp"
    else
        echo "Running on MacOS requires sudo rights and uses kperf.framework."
        echo "This is unsafe and also only works on M1 Macs."
        echo "Please set M1PERF_I_KNOW_WHAT_IM_DOING=1 to use this benchmark"
        exit -1
    fi
else
    if [[ "$UARCH_BENCH_USE_PERF" == "1" ]]; then
        PCTR_FILES="performance_counters_perf.cpp"
    elif [[ "$UARCH_BENCH_USE_SIMPLE" == "1" ]]; then
        echo "Using simple performance_counters implementation"
        echo "On x86_64 this uses rdtsc, which does not actually"
        echo "measure CPU clock cycles, the result will be inaccurate"
        PCTR_FILES="performance_counters_simple.cpp"
    else
        PCTR_FILES="performance_counters_papi.cpp -lpapi"
    fi
fi


bench_name="gemmbench_${mr}_${nr}_avec${avec_strat}_bvec${bvec_strat}"
python3 gemmerator.py -T $T --nr $nr --mr $mr -U $unroll -V $max_vregs -M $mem_use -t $data_type --bvec-strat $bvec_strat --avec-strat $avec_strat gemmbench.cpp.in > $bench_name.cpp
compile_command="$CXX  -g $CFLAGS -std=c++17 -o $bench_name $bench_name.cpp $PCTR_FILES"
echo Compilation command: $compile_command
if [[ "$nocompile" == "0" ]]; then
    $compile_command
else
    echo nocompile flag set. skipping compilation
fi
if [[ "$noexec" == "0" ]]; then
echo "Kernel ${mr}x${nr} ($data_type) avs=$avec_strat; bvs=$bvec_strat; mem=$mem_use; max_vregs=$max_vregs; unroll=$unroll; ISA=$T:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        ./$bench_name
    else
        numactl --physcpubind=0 ./$bench_name
    fi
else
    echo noexec flag set. skipping execution
fi
