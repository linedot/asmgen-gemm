#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cblas.h>
#include <fmt/format.h>


typedef double data_type;
void naive_asmgen_gemm(char transa, char transb, const std::size_t M,
                       const std::size_t N, const std::size_t K,
                       const data_type alpha, data_type *A, std::size_t lda,
                       data_type *B, std::size_t ldb, const data_type beta,
                       data_type *C, std::size_t ldc);


template<typename data_type>
void naive_gemm(
        char transa, char transb,
        const std::size_t M, const std::size_t N, const std::size_t K,
        const data_type alpha,
        data_type* A, std::size_t lda,
        data_type* B, std::size_t ldb,
        const data_type beta,
        data_type* C, std::size_t ldc)
{
    // Not quite "naive", but come on...
    constexpr std::size_t chunk_size = 16;
    std::size_t rest = N%chunk_size;
    std::vector<data_type> Bcache(K);
    #pragma omp parallel for
    for(std::size_t m = 0; m < M; m++)
    {
        std::size_t n = 0;
        for(n = 0; n < N-rest; n+=chunk_size)
        {
            double Caccum1 = 0.0;
            double Caccum2 = 0.0;
            double Caccum3 = 0.0;
            double Caccum4 = 0.0;
            double Caccum5 = 0.0;
            double Caccum6 = 0.0;
            double Caccum7 = 0.0;
            double Caccum8 = 0.0;
            double Caccum9 = 0.0;
            double Caccum10 = 0.0;
            double Caccum11 = 0.0;
            double Caccum12 = 0.0;
            double Caccum13 = 0.0;
            double Caccum14 = 0.0;
            double Caccum15 = 0.0;
            double Caccum16 = 0.0;
            #pragma omp simd reduction(+:Caccum1,Caccum2,Caccum3,Caccum4,Caccum5,Caccum6,Caccum7,Caccum8,Caccum9,Caccum10,Caccum11,Caccum12,Caccum13,Caccum14,Caccum15,Caccum16)
            for(std::size_t k = 0; k < K; k++)
            {
                Caccum1 += alpha*B[k*ldb+n+0]*A[k+m*lda];
                Caccum2 += alpha*B[k*ldb+n+1]*A[k+m*lda];
                Caccum3 += alpha*B[k*ldb+n+2]*A[k+m*lda];
                Caccum4 += alpha*B[k*ldb+n+3]*A[k+m*lda];
                Caccum5 += alpha*B[k*ldb+n+4]*A[k+m*lda];
                Caccum6 += alpha*B[k*ldb+n+5]*A[k+m*lda];
                Caccum7 += alpha*B[k*ldb+n+6]*A[k+m*lda];
                Caccum8 += alpha*B[k*ldb+n+7]*A[k+m*lda];
                Caccum9 += alpha*B[k*ldb+n+8]*A[k+m*lda];
                Caccum10 += alpha*B[k*ldb+n+9]*A[k+m*lda];
                Caccum11 += alpha*B[k*ldb+n+10]*A[k+m*lda];
                Caccum12 += alpha*B[k*ldb+n+11]*A[k+m*lda];
                Caccum13 += alpha*B[k*ldb+n+12]*A[k+m*lda];
                Caccum14 += alpha*B[k*ldb+n+13]*A[k+m*lda];
                Caccum15 += alpha*B[k*ldb+n+14]*A[k+m*lda];
                Caccum16 += alpha*B[k*ldb+n+15]*A[k+m*lda];
            }
            C[ldc*m+n+0] = beta*C[ldc*m+n+0] + Caccum1;
            C[ldc*m+n+1] = beta*C[ldc*m+n+1] + Caccum2;
            C[ldc*m+n+2] = beta*C[ldc*m+n+2] + Caccum3;
            C[ldc*m+n+3] = beta*C[ldc*m+n+3] + Caccum4;
            C[ldc*m+n+4] = beta*C[ldc*m+n+4] + Caccum5;
            C[ldc*m+n+5] = beta*C[ldc*m+n+5] + Caccum6;
            C[ldc*m+n+6] = beta*C[ldc*m+n+6] + Caccum7;
            C[ldc*m+n+7] = beta*C[ldc*m+n+7] + Caccum8;
            C[ldc*m+n+8] = beta*C[ldc*m+n+8] + Caccum9;
            C[ldc*m+n+9] = beta*C[ldc*m+n+9] + Caccum10;
            C[ldc*m+n+10] = beta*C[ldc*m+n+10] + Caccum11;
            C[ldc*m+n+11] = beta*C[ldc*m+n+11] + Caccum12;
            C[ldc*m+n+12] = beta*C[ldc*m+n+12] + Caccum13;
            C[ldc*m+n+13] = beta*C[ldc*m+n+13] + Caccum14;
            C[ldc*m+n+14] = beta*C[ldc*m+n+14] + Caccum15;
            C[ldc*m+n+15] = beta*C[ldc*m+n+15] + Caccum16;
        }
        for(; n < N; n++)
        {
            double Caccum = 0.0;
            #pragma omp simd reduction(+:Caccum)
            for(std::size_t k = 0; k < K; k++)
            {
                Caccum += alpha*B[k*ldb+n]*A[k+m*lda];
            }
            C[ldc*m+n] = beta*C[ldc*m+n] + Caccum;
        }
    }
}

void print_matrix(const std::size_t M, const size_t N, const std::vector<double>& matrixdata)
{
    for(std::size_t m = 0; m < M; m++)
    {
        for(std::size_t n = 0; n < N; n++)
        {
            fmt::print("{:<8.1f}", matrixdata[m*N+n]);
        }
        fmt::println("");
    }
}

int main()
{
    using hrc = std::chrono::high_resolution_clock;

    constexpr std::size_t M = 1200;
    constexpr std::size_t N = 1200;
    constexpr std::size_t K = 16000;
    std::vector<double> C1(M*N);
    std::vector<double> C2(M*N);
    std::vector<double> C3(M*N);
    std::vector<double> B(N*K);
    std::vector<double> A(M*K);


    std::random_device rdev{};
    std::vector<std::uint64_t> rdata(sizeof(std::mt19937_64)/sizeof(std::uint64_t));
#if 0 
    std::generate(rdata.begin(), rdata.end(), [&]{return rdev();});
#else
    std::iota(rdata.begin(), rdata.end(), 0);
#endif
    std::seed_seq sseq(rdata.begin(), rdata.end());
    std::mt19937_64 rng(sseq);
    std::uniform_real_distribution<double> dist(0.0,1.0);

    std::generate(C1.begin(), C1.end(), [&]{return dist(rng);} );
    std::copy(C1.begin(),C1.end(), C2.begin());
    std::copy(C1.begin(),C1.end(), C3.begin());
    std::generate(B.begin(), B.end(), [&]{return dist(rng);} );
    std::generate(A.begin(), A.end(), [&]{return dist(rng);} );

    double alpha = dist(rng);
    double beta = dist(rng);

#if 0 
    //std::cout << "BLAS library:\n";
    //print_matrix(M, N, C1);
    //std::cout << "naive:\n";
    //print_matrix(M, N, C2);
    std::cout << "naive asmgen:\n";
    print_matrix(M, N, C3);
#endif

    auto start = hrc::now();

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            alpha, 
            A.data(), K,
            B.data(), N,
            beta,
            C1.data(), N);

    auto end = hrc::now();
    auto us =  std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    auto gflops = static_cast<double>(M*N + M*N*K*3)/us/1e3;
    fmt::print("BLAS library finished in {}us. ({:.3f} GFLOP/s)\n",us,gflops);

    start = hrc::now();
    naive_gemm('n','n',
            M, N, K,
            alpha, 
            A.data(), K,
            B.data(), N,
            beta,
            C2.data(), N);

    end = hrc::now();
    us =  std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    gflops = static_cast<double>(M*N + M*N*K*3)/us/1e3;
    fmt::print("Naive implementation finished in {}us. ({:.3f} GFLOP/s)\n",us,gflops);

    start = hrc::now();
    naive_asmgen_gemm('n','n',
            M, N, K,
            alpha, 
            A.data(), K,
            B.data(), N,
            beta,
            C3.data(), N);

    end = hrc::now();
    us =  std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    gflops = static_cast<double>(M*N + M*N*K*3)/us/1e3;
    fmt::print("Naive asmgen implementation finished in {}us. ({:.3f} GFLOP/s)\n",us,gflops);

    // TODO: find some reliable info on acceptable tolerance.
    //       epsilon is obviously too small, but this seems excessive
    constexpr double epsilon = std::numeric_limits<double>::epsilon()*1000.0;

    fmt::print("Testing correctness with epsilon={}\n",epsilon);

    auto minabsdiff = [epsilon](auto a, auto b)
        -> bool
    {
        return std::fabs(b-a) < epsilon;
    };

    auto findmax = [](auto first, auto second)
    {
        return std::max(first, second);
    };
    auto reldiff = [](auto first, auto second)
    {
        return std::fabs(second-first)/first;
    };

#if 0 
    std::cout << "BLAS library:\n";
    print_matrix(M, N, C1);
    //std::cout << "naive:\n";
    //print_matrix(M, N, C2);
    std::cout << "naive asmgen:\n";
    print_matrix(M, N, C3);
#endif

    auto biggest_difference =
        std::transform_reduce(
                C1.begin(), C1.end(),
                C2.begin(),
                0.0,
                findmax,
                reldiff);
    std::cout << "Biggest difference: " << biggest_difference << "\n";
    if(biggest_difference < epsilon)
    {
        std::cout << "OK:   Difference between BLAS library and naive-implementation within tolerance.\n";
    }
    else
    {
        std::cout << "FAIL: Difference between BLAS library and naive-implementation outside tolerance.\n";
    }

    biggest_difference =
        std::transform_reduce(
                C1.begin(), C1.end(),
                C3.begin(),
                0.0,
                findmax,
                reldiff);
    std::cout << "Biggest difference: " << biggest_difference << "\n";
    if(biggest_difference < epsilon)
    {
        std::cout << "OK:   Difference between BLAS library and naive-asmgen implementation within tolerance.\n";
    }
    else
    {
        std::cout << "FAIL: Difference between BLAS library and naive-asmgen implementation outside tolerance.\n";
    }


    biggest_difference =
        std::transform_reduce(
                C2.begin(), C2.end(),
                C3.begin(),
                0.0,
                findmax,
                reldiff);
    std::cout << "Biggest difference: " << biggest_difference << "\n";
    if(biggest_difference < epsilon)
    {
        std::cout << "OK:   Difference between naive and naive-asmgen implementation within tolerance.\n";
    }
    else
    {
        std::cout << "FAIL: Difference between naive and naive-asmgen implementation outside tolerance.\n";
    }


    return 0;
}
