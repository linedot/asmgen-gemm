# asmgen-gemm: Generator for GEMM microkernels using asmgen

Code previously part of the uarch\_bench project

## Requirements:
- [asmgen](https://github.com/linedot/asmgen)
- c++ compiler
- BLAS library (to test against)
- fmtlib (output in tests)


## Run performance test  (example on AVX2-capable machine)
```
$ cd asmgen-gemm
$ python3 ./gemmerator.py --mr 8 --nr 6 -V 16 -M l1 -t double -T fma256 --bvec-strat dist1_boff --avec-strat postload --output-filename gemmbench_8_6.cpp gemmbench.cpp.in
$ clang++ -std=c++20 -Ofast -g -fopenmp -march=native -mtune=native -I performance_counters/ -o gemmbench_8_6 gemmbench_8_6.cpp performance_counters/performance_counters_perf.cpp
$ ./gemmbench_8_6
Will perform 5979 iterations, reducing measurement overhead to 0.1%
Event: CYCLES - min: 286640; avg: 287538; max: 289944
Avg. 15.9697 FLOPS/cycle
```
With a sufficiently large microkernel you should see a number very close to 100% of the theoretical peak performance of your CPU

## Run correctness test (example on AVX2-capable machine, using BLIS as BLAS library)

```bash
$ cd asmgen-gemm/tests
$ python3 ../gemmerator.py --mr 8 --nr 6 -V 16 -U2 -M contiguous -t double -T fma256 --bvec-strat dist1_boff --avec-strat postload --output-filename naive_asmgen_gemm.cpp naive_asmgen_gemm.cpp.in
$ g++ -std=c++20 -Ofast -g -fopenmp -march=native -mtune=native -I/usr/include/blis/ naivegemm.cpp naive_asmgen_gemm.cpp -o naivegemm -lfmt -lblis-mt
$ ./naivegemm 
BLAS library finished in 95818us. (721.383 GFLOP/s)
Naive implementation finished in 1312114us. (52.679 GFLOP/s)
Naive asmgen implementation finished in 311049us. (222.220 GFLOP/s)
Testing correctness with epsilon=2.220446049250313e-13
Biggest difference: 4.88426e-15
OK:   Difference between BLAS library and naive-implementation within tolerance.
Biggest difference: 1.547e-14
OK:   Difference between BLAS library and naive-asmgen implementation within tolerance.
Biggest difference: 1.63961e-14
OK:   Difference between naive and naive-asmgen implementation within tolerance.
```

Output of correctness test is not suitable for performance-testing as the test fixture for asmgen ukernels is not optimized

## Using QEMU to correctness-test other ISAs

You will need [fmt](https://github.com/fmtlib/fmt) and a BLAS library for the target architecture to run the test. Here are some instructions on how to cross compile and install fmt and BLIS into `$HOME/Software/{aarch64,riscv64}/`.

Assuming an aarch64 GCC toolchain in `$AARCH64_TOOLCHAIN` and a riscv64 one in `$RISCV64_TOOLCHAIN`

### Cross-compile and install blis as blas library

```bash
$ git clone git clone https://github.com/flame/blis.git
$ cd blis
$ PATH=$RISCV64_TOOLCHAIN/bin:$PATH CC=riscv64-linux-gnu-gcc CXX=riscv64-linux-gnu-g++ AS=riscv64-linux-gnu-as ./configure --enable-static --enable-shared -t openmp --enable-cblas --prefix=$HOME/Software/riscv64/blis rv64iv
$ make -j $(nproc)
$ make install
$ PATH=$AARCH64_TOOLCHAIN/bin:$PATH CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AS=aarch64-linux-gnu-as ./configure --enable-static --enable-shared -t openmp --enable-cblas --prefix=$HOME/Software/aarch64/blis armsve
$ make -j $(nproc)
$ make install

```

### Cross-compile and install fmt

CMake toolchain file aarch64.cmake:

```
# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# where is the target environment located
set(CMAKE_FIND_ROOT_PATH
    $ENV{AARCH64_TOOLCHAIN}
    $ENV{HOME}/Software/aarch64/)

# adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

CMake toolchain file riscv64.cmake

```
# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# where is the target environment located
set(CMAKE_FIND_ROOT_PATH
    $ENV{RISCV64_TOOLCHAIN}
    $ENV{HOME}/Software/riscv64/)

# adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

Build and install fmt

```bash
$ git clone git clone https://github.com/fmtlib/fmt
$ mkdir fmt-aarch64-build
$ cmake -DCMAKE_INSTALL_PREFIX=$HOME/Software/aarch64/fmt -DCMAKE_TOOLCHAIN_FILE=aarch64.cmake -DFMT_TEST=OFF ../fmt/
$ make -j $(nproc)
$ make install
$ cd ../
$ mkdir fmt-riscv64-build
$ cmake -DCMAKE_INSTALL_PREFIX=$HOME/Software/riscv64/fmt -DCMAKE_TOOLCHAIN_FILE=riscv64.cmake -DFMT_TEST=OFF ../fmt/
$ make -j $(nproc)
$ make install
```


### Generate and cross-compile test for other ISAS

RVV:
```bash
$ cd asmgen-gemm/tests
$ python3 ../gemmerator.py --mr 2 --nr 6 -V 32 -M contiguous -t double -T rvv --bvec-strat fmavf --avec-strat preload --output-filename naive_asmgen_gemm_rvv.cpp naive_asmgen_gemm.cpp.in
$ $RISCV64_TOOLCHAIN/bin/riscv64-linux-gnu-g++ -march=rv64imafdcv_zicbop -g -Ofast -fopenmp -I $HOME/Software/riscv64/blis/include/blis -I $HOME/Software/riscv64/fmt/include naivegemm.cpp naive_asmgen_gemm_rvv.cpp -o naivegemm_rvv -L $HOME/Software/riscv64/blis/lib -L $HOME/Software/riscv64/fmt/lib -lblis -lfmt
```

SVE:
```bash
$ cd asmgen-gemm/tests
$ python3 ../gemmerator.py --mr 2 --nr 6 -V 32 -M contiguous -t double -T sve --bvec-strat dist1_boff --avec-strat preload --output-filename naive_asmgen_gemm_sve.cpp naive_asmgen_gemm.cpp.in
$ $AARCH64_TOOLCHAIN/bin/aarch64-linux-gnu-g++ -march=armv8-a+sve -g -Ofast -fopenmp -I $HOME/Software/aarch64/blis/include/blis -I $HOME/Software/aarch64/fmt/include naivegemm.cpp naive_asmgen_gemm_sve.cpp -o naivegemm_sve -L $HOME/Software/aarch64/blis/lib -L $HOME/Software/aarch64/fmt/lib -lblis -lfmt
```


### Run the tests with QEMU

RVV:
```bash
$ LD_LIBRARY_PATH=$HOME/Software/riscv64/blis/lib:$RISCV64_TOOLCHAIN/riscv64-linux-gnu/lib/ qemu-riscv64-static -L RISCV64_TOOLCHAIN/sysroot ./naivegemm_rvv
BLAS library finished in 14619us. (0.085 GFLOP/s)
Naive implementation finished in 17006us. (0.073 GFLOP/s)
Naive asmgen implementation finished in 9829us. (0.126 GFLOP/s)
Testing correctness with epsilon=2.220446049250313e-13
Biggest difference: 8.09829e-16
OK:   Difference between BLAS library and naive-implementation within tolerance.
Biggest difference: 7.70228e-16
OK:   Difference between BLAS library and naive-asmgen implementation within tolerance.
Biggest difference: 8.74075e-16
OK:   Difference between naive and naive-asmgen implementation within tolerance.
```

SVE:
```bash
$ LD_LIBRARY_PATH=$HOME/Software/aarch64/blis/lib:$AARCH64_TOOLCHAIN/aarch64-linux-gnu/lib/ qemu-aarch64-static -L AARCH64_TOOLCHAIN/sysroot ./naivegemm_sve
BLAS library finished in 40488us. (0.031 GFLOP/s)
Naive implementation finished in 35492us. (0.035 GFLOP/s)
Naive asmgen implementation finished in 14751us. (0.084 GFLOP/s)
Testing correctness with epsilon=2.220446049250313e-13
Biggest difference: 7.2256e-16
OK:   Difference between BLAS library and naive-implementation within tolerance.
Biggest difference: 4.98801e-16
OK:   Difference between BLAS library and naive-asmgen implementation within tolerance.
Biggest difference: 7.2256e-16
OK:   Difference between naive and naive-asmgen implementation within tolerance.
```
