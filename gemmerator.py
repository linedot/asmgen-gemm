#!/usr/bin/env python3
from nanogemm import nanogemm,gemm_params,prefetch_options,mem_use_type,kernel_layout,bvec_strategy_type,avec_strategy_type
from asmgen.asmblocks.noarch import asm_data_type
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071

from asmgen.compilation.compiler_presets import cross_archs


import sys
import argparse
import shutil
from string import Template

from subprocess import Popen, PIPE

#from math import lcm
import math

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

mem_use_map = {
        "samedata" : mem_use_type.SAMEDATA,
        "l1" : mem_use_type.L1,
        "contiguous" : mem_use_type.CONTIGUOUS,
}

bvec_strat_map = {
        "dist1_boff" : bvec_strategy_type.DIST1_BOFF,
        "dist1_inc" : bvec_strategy_type.DIST1_INC,
        "fmaidx" : bvec_strategy_type.FMAIDX,
        "fmavf" : bvec_strategy_type.FMAVF,
        "noload" : bvec_strategy_type.NOLOAD,
}

avec_strat_map = {
        "postload" : avec_strategy_type.POSTLOAD,
        "preload"  : avec_strategy_type.PRELOAD,
}

genmap = {
        'sve' : sve,
        'neon' : neon,
        'fma128' : fma128,
        'fma256' : fma256,
        'avx512' : avx512,
        'rvv' : rvv,
        'rvv071' : rvv071,
}

dt_ctype_map = {
        "double" : "double",
        "single" : "float",
}

def valid_bvec_strats_for(simd_type):
    if "neon" == simd_type:
        return ["dist1_inc","fmaidx","noload"]
    elif "sve" == simd_type:
        return ["dist1_boff","noload"]
    elif "fma128" == simd_type:
        return ["dist1_boff","noload"]
    elif "fma256" == simd_type:
        return ["dist1_boff","noload"]
    elif "avx512" == simd_type:
        return ["dist1_boff","noload"]
    elif "rvv" == simd_type:
        return ["dist1_boff","fmavf","noload"]
    elif "rvv071" == simd_type:
        return ["dist1_boff","fmavf","noload"]
    else:
        raise RuntimeError("Invalid bvec strat")

def main():

    valid_mem_use_str = ",".join(mem_use_map.keys())
    valid_bvec_strat_str = ",".join(bvec_strat_map.keys())
    valid_simd_types = ",".join(genmap.keys())

    parser = argparse.ArgumentParser(description="GEMM kernel assembly generator")
    parser.add_argument("--nr", metavar="int", type=int,
            required=True, help="# scalar elements in second microkernel dimension")
    parser.add_argument("--mr", metavar="int", type=int,
            required=True, help="(VLA: # number of simd registers in -; FIXED: # number of scalar elements in-) first microkernel dimension")
    parser.add_argument("-U","--unroll", type=int, default=argparse.SUPPRESS, help="unroll this many k iterations (optional, will use free_bregs*nr if unspecified)")
    parser.add_argument("-V","--max_vregs", metavar="int", type=int,
            required=True, help="Limit vector register usage to this many")
    parser.add_argument("-M", "--mem", metavar="usage", type=str,
            default="l1", help=f"memory usage (valid values: {{{valid_mem_use_str}}})")
    parser.add_argument("-t", "--datatype", metavar="type", type=str,
            default="double", help="datatype (valid values: {double,single})")
    parser.add_argument("-T", "--simd-type", metavar="simd", type=str,
            required=True,
            choices=genmap.keys(),
            help=f"Type of instructions to generate. Valid options: {{{valid_simd_types}}}")
    parser.add_argument("--bvec-strat", metavar="strat", type=str,
            default="auto", help=f"Strategy dealing with values of B. valid values: {{{valid_bvec_strat_str}}} or \"auto\"")
    parser.add_argument("--avec-strat", metavar="strat", type=str,
            default="auto", help=f"Strategy dealing with values of A. valid values: {{postload,preload}} or \"auto\"")
    parser.add_argument("--output-filename", metavar="output_filename", type=str,
                        required=True, help=f"Output the generated code into this file")
    parser.add_argument("--compile-benchmark", action="store_true",
                        help=f"Instead of outputting the generated code, compile a benchmark executable")
    parser.add_argument("filename", type=str, help="Name of the input file where $GEMM will be replaced with the actual GEMM assembly")

    args = parser.parse_args()

    if not args.mem in mem_use_map.keys():
        sys.stderr.write(f"Invalid mem usage: {args.mem}\n")
        exit(-1)

    if not args.datatype in ["double","single"]:
        sys.stderr.write(f"Invalid data type: {args.datatype}\n")
        exit(-1)

    if not args.simd_type in genmap.keys():
        sys.stderr.write(f"Invalid simd type: {args.simd_type}\n")
        exit(-1)

    gen = genmap[args.simd_type]()
    layout = kernel_layout()



    if "auto" == args.bvec_strat:
        if "neon" == args.simd_type:
            layout.bvec_strat = bvec_strategy_type.FMAIDX
    else:
        if not args.bvec_strat in bvec_strat_map.keys():
            sys.stderr.write(f"Invalid strategy for B values: {args.bvec_strat}\n")
            exit(-1)
        valid_strats = valid_bvec_strats_for(args.simd_type)
        if args.bvec_strat in valid_strats:
            layout.bvec_strat = bvec_strat_map[args.bvec_strat]
        else:
            valid_strats_str = ','.join(valid_strats)
            sys.stderr.write(f"Invalid bvec_strat for {args.simd_type}. Valid values: {{{valid_strats_str}}}")

    if "auto" != args.avec_strat:
        if not args.avec_strat in avec_strat_map.keys():
            sys.stderr.write(f"Invalid strategy for A values: {args.avec_strat}\n")
            exit(-1)
        layout.avec_strat = avec_strat_map[args.avec_strat]


    mr = args.mr
    nr = args.nr
    max_vregs = args.max_vregs

    
    mem_use = mem_use_type.L1
    if "samedata" == args.mem:
        mem_use = mem_use_type.SAMEDATA
    elif "l1" == args.mem:
        mem_use = mem_use_type.L1
    elif "contiguous" == args.mem:
        mem_use = mem_use_type.CONTIGUOUS


    dt = asm_data_type.DOUBLE
    if "double" == args.datatype:
        dt = asm_data_type.DOUBLE
    elif "single" == args.datatype:
        dt = asm_data_type.SINGLE

    vectors_in_mr = 0
    if gen.is_vla:
        vectors_in_mr = mr
        elements_in_vector = "unknown"
    else:
        vectors_in_mr = mr//(gen.simd_size//dt.value)
        elements_in_vector = gen.simd_size//dt.value


    avec_count = vectors_in_mr
    if avec_strategy_type.POSTLOAD == layout.avec_strat:
        avec_count = vectors_in_mr
    elif avec_strategy_type.PRELOAD == layout.avec_strat:
        avec_count = 2*vectors_in_mr

    if max_vregs < vectors_in_mr*nr+2:
        print(f"specified {max_vregs} maximum vregs to use, but at least {vectors_in_mr*nr+2} architectural vector registers required to express a {mr}x{nr} kernel")
        exit(-2)

    if max_vregs > gen.max_vregs:
        print(f"specified {max_vregs} maximum vregs to use, but at most {gen.max_vregs} are available")
        exit(-2)

    if not "unroll" in args or 0 == args.unroll:
        if bvec_strategy_type.FMAVF == layout.bvec_strat:
            # use 2xnr fregs to avoid huge code
            #b_regs = gen.max_fregs()
            b_regs = 2*nr if 2*nr < gen.max_fregs else nr
        else:
            b_regs = (max_vregs-vectors_in_mr*nr-avec_count)
        smallest_unroll = lcm(b_regs,nr)//nr
        if(bvec_strategy_type.FMAIDX == layout.bvec_strat):
            smallest_unroll = lcm(b_regs*elements_in_vector,nr)//nr
        unroll_factor = smallest_unroll
        if 3 > unroll_factor:
            unroll_factor = 4
        if 4 == unroll_factor:
            unroll_factor = 8
        if 6 == unroll_factor:
            unroll_factor = 12
    else:
        unroll_factor = args.unroll

    sys.stderr.write(f"Using unroll factor of {unroll_factor}\n")

    pf = prefetch_options()
    pf.a_init_count=vectors_in_mr*4
    pf.b_init_count=nr//2
    pf.c_init_count=2
    # remove prefetching for now
    #pf.a_init_count=0
    #pf.b_init_count=0
    #pf.c_init_count=0

    free_vreg_count = max_vregs - vectors_in_mr*nr

    vecs_required = avec_count
    if bvec_strategy_type.FMAVF != layout.bvec_strat:
        vecs_required = avec_count+1

    if free_vreg_count < vecs_required:
        sys.stderr.write(f"Can't use {layout.avec_strat.name} layout for A: only {free_vreg_count} vector registers available, but {vecs_required} are required")
        exit(-3)

    gemm_asm = nanogemm(asm=gen, pf=pf, layout=layout,
                        vectors_in_mr=vectors_in_mr, nr=nr,
                        unroll_factor=unroll_factor,
                        max_vregs=max_vregs,
                        mem_use=mem_use, datatype=dt, params=gemm_params())

    substitutions = {
            "MEM": mem_use.name,
            "MR" : mr,
            "DT" : dt_ctype_map[args.datatype],
            "VECINMR" : vectors_in_mr,
            "NR" : nr,
            "UNROLL" : unroll_factor,
            "GEMM" : gemm_asm,
            "GETSIMDSIZE" : gen.c_simd_size_function
            }

    source_code = ""
    with open(args.filename, 'r') as f:
        src = Template(f.read())
        source_code = src.substitute(substitutions)

    clang_format_exe = shutil.which("clang-format")
    if not None == clang_format_exe:
        print(f"Found clang-format: {clang_format_exe}, will pretty-print the source")
        cmd = f"{clang_format_exe}"
        p = Popen([cmd,"--style=llvm"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        pretty_source_code = p.communicate(input=source_code.encode())[0].decode()
        if 0 == p.returncode:
            source_code = pretty_source_code

    if not args.compile_benchmark:
        print(f"writing generated code to {args.output_filename}")
        with open(args.output_filename, 'w') as f:
            f.write(source_code)
    else:
        print(f"Compiling benchmark executable {args.output_filename}")
        from asmgen.compilation.tools import compiler
        cxx = compiler('clang++',args.simd_type)
        cross_compile='native'
        if not gen.supported_on_host():
            cross_compile=cross_archs[args.simd_type]
        result = cxx.compile_exe(
                source=source_code,
                output_filename=args.output_filename,
                libs=["performance_counters_perf.cpp"],
                cross_compile=cross_compile)
        if not result:
            print("compilation failed")


if __name__ == "__main__":
    main()
