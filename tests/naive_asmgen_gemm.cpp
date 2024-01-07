#include <cstdint>
#include <vector>

using std::size_t;

typedef double data_type;

size_t get_simd_size() { return 32; }

void naive_asmgen_gemm(char transa, char transb, const std::size_t M,
                       const std::size_t N, const std::size_t K,
                       const data_type alpha_v, data_type *A, std::size_t lda,
                       data_type *B, std::size_t ldb, const data_type beta_v,
                       data_type *C, std::size_t ldc) {
  std::size_t nrest = N % 6;
  std::size_t mrest = M % 8;
  auto iterations = K / 2;
  auto kleft = (K % 2);
#pragma omp parallel for
  for (std::size_t m = 0; m < M - mrest; m += 8) {
    std::vector<data_type> packed_b(6 * K);
    std::vector<data_type> packed_a(8 * K);
    std::vector<data_type> c_tile(8 * 6);
    for (std::size_t k = 0; k < K; k++) {
#pragma omp unroll(8)
      for (std::size_t mi = 0; mi < 8; mi++) {
        packed_a[k * 8 + mi] = A[(m + mi) * lda + k];
      }
    }
    for (std::size_t n = 0; n < N - nrest; n += 6) {
      for (std::size_t k = 0; k < K; k++) {
#pragma omp unroll(6)
        for (std::size_t ni = 0; ni < 6; ni++) {
          packed_b[k * 6 + ni] = B[k * ldb + n + ni];
        }
      }
#pragma omp unroll(8)
      for (std::size_t mr = 0; mr < 8; mr++) {
#pragma omp unroll(6)
        for (std::size_t nr = 0; nr < 6; nr++) {
          c_tile[nr * 8 + mr] = C[(m + mr) * ldc + n + nr];
        }
      }
      auto *a = packed_a.data();
      auto *b = packed_b.data();
      auto *c = c_tile.data();
      const auto *alpha = &alpha_v;
      const auto *beta = &beta_v;
#if 1
      asm("vpxor %%ymm4,%%ymm4,%%ymm4\n\t"
          "vpxor %%ymm5,%%ymm5,%%ymm5\n\t"
          "vpxor %%ymm6,%%ymm6,%%ymm6\n\t"
          "vpxor %%ymm7,%%ymm7,%%ymm7\n\t"
          "vpxor %%ymm8,%%ymm8,%%ymm8\n\t"
          "vpxor %%ymm9,%%ymm9,%%ymm9\n\t"
          "vpxor %%ymm10,%%ymm10,%%ymm10\n\t"
          "vpxor %%ymm11,%%ymm11,%%ymm11\n\t"
          "vpxor %%ymm12,%%ymm12,%%ymm12\n\t"
          "vpxor %%ymm13,%%ymm13,%%ymm13\n\t"
          "vpxor %%ymm14,%%ymm14,%%ymm14\n\t"
          "vpxor %%ymm15,%%ymm15,%%ymm15\n\t"
          "mov %[a],%%r8\n\t"
          "mov %[b],%%r9\n\t"
          "mov %[c],%%r10\n\t"
          "prefetcht0 0(%%r8)\n\t"
          "prefetcht0 64(%%r8)\n\t"
          "prefetcht0 128(%%r8)\n\t"
          "prefetcht0 192(%%r8)\n\t"
          "prefetcht0 256(%%r8)\n\t"
          "prefetcht0 320(%%r8)\n\t"
          "prefetcht0 384(%%r8)\n\t"
          "prefetcht0 448(%%r8)\n\t"
          "prefetcht0 0(%%r9)\n\t"
          "prefetcht0 64(%%r9)\n\t"
          "prefetcht0 128(%%r9)\n\t"
          "prefetcht0 0(%%r10)\n\t"
          "prefetcht0 64(%%r10)\n\t"
          "vmovupd (%%r8),%%ymm0\n\t"
          "vmovupd 32(%%r8),%%ymm1\n\t"
          "addq $64,%%r8\n\t"
          "vbroadcastsd 0(%%r9),%%ymm2\n\t"
          "vbroadcastsd 8(%%r9),%%ymm3\n\t"
          "addq $16,%%r9\n\t"
          ".load_b_end:\n\t"
          "movq %[iterations],%%r11\n\t"
          "test %%r11,%%r11\n\t"
          "jz .kloopend\n\t"
          "addq $-1,%%r11\n\t"
          "test %%r11,%%r11\n\t"
          "jz .klast\n\t"
          ".kloop:\n\t"
          "sub $1, %%r11\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm4\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm5\n\t"
          "vbroadcastsd 0(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm6\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm7\n\t"
          "vbroadcastsd 8(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm8\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm9\n\t"
          "vbroadcastsd 16(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm10\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm11\n\t"
          "vbroadcastsd 24(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm12\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm13\n\t"
          "vbroadcastsd 32(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm14\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm15\n\t"
          "vbroadcastsd 40(%%r9),%%ymm3\n\t"
          "vmovupd (%%r8),%%ymm0\n\t"
          "vmovupd 32(%%r8),%%ymm1\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm4\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm5\n\t"
          "vbroadcastsd 48(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm6\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm7\n\t"
          "vbroadcastsd 56(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm8\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm9\n\t"
          "vbroadcastsd 64(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm10\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm11\n\t"
          "vbroadcastsd 72(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm12\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm13\n\t"
          "vbroadcastsd 80(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm14\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm15\n\t"
          "vbroadcastsd 88(%%r9),%%ymm3\n\t"
          "vmovupd 64(%%r8),%%ymm0\n\t"
          "vmovupd 96(%%r8),%%ymm1\n\t"
          "addq $128,%%r8\n\t"
          "addq $96,%%r9\n\t"
          "cmp $0x0,%%r11\n\t"
          "jne .kloop\n\t"
          ".klast:\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm4\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm5\n\t"
          "vbroadcastsd 0(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm6\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm7\n\t"
          "vbroadcastsd 8(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm8\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm9\n\t"
          "vbroadcastsd 16(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm10\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm11\n\t"
          "vbroadcastsd 24(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm12\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm13\n\t"
          "vbroadcastsd 32(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm14\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm15\n\t"
          "vbroadcastsd 40(%%r9),%%ymm3\n\t"
          "vmovupd (%%r8),%%ymm0\n\t"
          "vmovupd 32(%%r8),%%ymm1\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm4\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm5\n\t"
          "vbroadcastsd 48(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm6\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm7\n\t"
          "vbroadcastsd 56(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm8\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm9\n\t"
          "vbroadcastsd 64(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm10\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm11\n\t"
          "vbroadcastsd 72(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm12\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm13\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm14\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm15\n\t"
          "addq $64,%%r8\n\t"
          "addq $80,%%r9\n\t"
          ".kloopend:\n\t"
          "movq %[kleft],%%r11\n\t"
          "test %%r11,%%r11\n\t"
          "jz .k1loopend\n\t"
          "vmovupd (%%r8),%%ymm0\n\t"
          "vmovupd 32(%%r8),%%ymm1\n\t"
          "addq $64,%%r8\n\t"
          "vbroadcastsd 0(%%r9),%%ymm2\n\t"
          "vbroadcastsd 8(%%r9),%%ymm3\n\t"
          "vbroadcastsd 16(%%r9),%%ymm2\n\t"
          "vbroadcastsd 24(%%r9),%%ymm3\n\t"
          "vbroadcastsd 32(%%r9),%%ymm2\n\t"
          "vbroadcastsd 40(%%r9),%%ymm3\n\t"
          "addq $48,%%r9\n\t"
          ".k1novecload:\n\t"
          "addq $-1,%%r11\n\t"
          "test %%r11,%%r11\n\t"
          "jz .k1last\n\t"
          ".k1loop:\n\t"
          "sub $1, %%r11\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm4\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm5\n\t"
          "vbroadcastsd 0(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm6\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm7\n\t"
          "vbroadcastsd 8(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm8\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm9\n\t"
          "vbroadcastsd 16(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm10\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm11\n\t"
          "vbroadcastsd 24(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm12\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm13\n\t"
          "vbroadcastsd 32(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm14\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm15\n\t"
          "vbroadcastsd 40(%%r9),%%ymm3\n\t"
          "vmovupd (%%r8),%%ymm0\n\t"
          "vmovupd 32(%%r8),%%ymm1\n\t"
          "addq $64,%%r8\n\t"
          "addq $48,%%r9\n\t"
          "cmp $0x0,%%r11\n\t"
          "jne .k1loop\n\t"
          ".k1last:\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm4\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm5\n\t"
          "vbroadcastsd 0(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm6\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm7\n\t"
          "vbroadcastsd 8(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm8\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm9\n\t"
          "vbroadcastsd 16(%%r9),%%ymm2\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm10\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm11\n\t"
          "vbroadcastsd 24(%%r9),%%ymm3\n\t"
          "vfmadd231pd %%ymm0,%%ymm2,%%ymm12\n\t"
          "vfmadd231pd %%ymm1,%%ymm2,%%ymm13\n\t"
          "vfmadd231pd %%ymm0,%%ymm3,%%ymm14\n\t"
          "vfmadd231pd %%ymm1,%%ymm3,%%ymm15\n\t"
          ".k1loopend:\n\t"
          "mov %[alpha],%%r8\n\t"
          "mov %[beta],%%r9\n\t"
          "vmovsd (%%r8),%%xmm0\n\t"
          "vmovsd (%%r9),%%xmm1\n\t"
          "vpxor %%xmm2,%%xmm2,%%xmm2\n\t"
          "ucomisd %%xmm2,%%xmm1\n\t"
          "je .beta0\n\t"
          "vbroadcastsd (%%r8),%%ymm1\n\t"
          "vbroadcastsd (%%r9),%%ymm0\n\t"
          "movq %%r10, %%r13\n\t"
          "vmovupd (%%r10),%%ymm2\n\t"
          "vmulpd %%ymm2,%%ymm0,%%ymm2\n\t"
          "vfmadd231pd %%ymm1,%%ymm4,%%ymm2\n\t"
          "vmovupd 32(%%r10),%%ymm3\n\t"
          "vmulpd %%ymm3,%%ymm0,%%ymm3\n\t"
          "vfmadd231pd %%ymm1,%%ymm5,%%ymm3\n\t"
          "vmovupd 64(%%r10),%%ymm4\n\t"
          "vmulpd %%ymm4,%%ymm0,%%ymm4\n\t"
          "vfmadd231pd %%ymm1,%%ymm6,%%ymm4\n\t"
          "vmovupd 96(%%r10),%%ymm5\n\t"
          "vmulpd %%ymm5,%%ymm0,%%ymm5\n\t"
          "vfmadd231pd %%ymm1,%%ymm7,%%ymm5\n\t"
          "vmovupd 128(%%r10),%%ymm6\n\t"
          "vmulpd %%ymm6,%%ymm0,%%ymm6\n\t"
          "vfmadd231pd %%ymm1,%%ymm8,%%ymm6\n\t"
          "vmovupd 160(%%r10),%%ymm7\n\t"
          "vmulpd %%ymm7,%%ymm0,%%ymm7\n\t"
          "vfmadd231pd %%ymm1,%%ymm9,%%ymm7\n\t"
          "vmovupd 192(%%r10),%%ymm8\n\t"
          "vmulpd %%ymm8,%%ymm0,%%ymm8\n\t"
          "vfmadd231pd %%ymm1,%%ymm10,%%ymm8\n\t"
          "vmovupd %%ymm2,(%%r13)\n\t"
          "vmovupd 224(%%r10),%%ymm9\n\t"
          "vmulpd %%ymm9,%%ymm0,%%ymm9\n\t"
          "vfmadd231pd %%ymm1,%%ymm11,%%ymm9\n\t"
          "vmovupd %%ymm3,32(%%r13)\n\t"
          "vmovupd 256(%%r10),%%ymm10\n\t"
          "vmulpd %%ymm10,%%ymm0,%%ymm10\n\t"
          "vfmadd231pd %%ymm1,%%ymm12,%%ymm10\n\t"
          "vmovupd %%ymm4,64(%%r13)\n\t"
          "vmovupd 288(%%r10),%%ymm2\n\t"
          "vmulpd %%ymm2,%%ymm0,%%ymm2\n\t"
          "vfmadd231pd %%ymm1,%%ymm13,%%ymm2\n\t"
          "vmovupd %%ymm5,96(%%r13)\n\t"
          "vmovupd 320(%%r10),%%ymm11\n\t"
          "vmulpd %%ymm11,%%ymm0,%%ymm11\n\t"
          "vfmadd231pd %%ymm1,%%ymm14,%%ymm11\n\t"
          "vmovupd %%ymm6,128(%%r13)\n\t"
          "vmovupd 352(%%r10),%%ymm3\n\t"
          "vmulpd %%ymm3,%%ymm0,%%ymm3\n\t"
          "vfmadd231pd %%ymm1,%%ymm15,%%ymm3\n\t"
          "vmovupd %%ymm7,160(%%r13)\n\t"
          "vmovupd %%ymm8,192(%%r13)\n\t"
          "vmovupd %%ymm9,224(%%r13)\n\t"
          "vmovupd %%ymm10,256(%%r13)\n\t"
          "vmovupd %%ymm2,288(%%r13)\n\t"
          "vmovupd %%ymm11,320(%%r13)\n\t"
          "vmovupd %%ymm3,352(%%r13)\n\t"
          "jmp .beta0end\n\t"
          ".beta0:\n\t"
          "vbroadcastsd (%%r8),%%ymm1\n\t"
          "movq %%r10, %%r13\n\t"
          "vmulpd %%ymm4,%%ymm1,%%ymm4\n\t"
          "vmulpd %%ymm5,%%ymm1,%%ymm5\n\t"
          "vmulpd %%ymm6,%%ymm1,%%ymm6\n\t"
          "vmulpd %%ymm7,%%ymm1,%%ymm7\n\t"
          "vmulpd %%ymm8,%%ymm1,%%ymm8\n\t"
          "vmulpd %%ymm9,%%ymm1,%%ymm9\n\t"
          "vmulpd %%ymm10,%%ymm1,%%ymm10\n\t"
          "vmovupd %%ymm4,(%%r13)\n\t"
          "vmulpd %%ymm11,%%ymm1,%%ymm11\n\t"
          "vmovupd %%ymm5,32(%%r13)\n\t"
          "vmulpd %%ymm12,%%ymm1,%%ymm12\n\t"
          "vmovupd %%ymm6,64(%%r13)\n\t"
          "vmulpd %%ymm13,%%ymm1,%%ymm13\n\t"
          "vmovupd %%ymm7,96(%%r13)\n\t"
          "vmulpd %%ymm14,%%ymm1,%%ymm14\n\t"
          "vmovupd %%ymm8,128(%%r13)\n\t"
          "vmulpd %%ymm15,%%ymm1,%%ymm15\n\t"
          "vmovupd %%ymm9,160(%%r13)\n\t"
          "vmovupd %%ymm10,192(%%r13)\n\t"
          "vmovupd %%ymm11,224(%%r13)\n\t"
          "vmovupd %%ymm12,256(%%r13)\n\t"
          "vmovupd %%ymm13,288(%%r13)\n\t"
          "vmovupd %%ymm14,320(%%r13)\n\t"
          "vmovupd %%ymm15,352(%%r13)\n\t"
          ".beta0end:\n\t"
          : [dummy_c] "+m"(*(double(*)[])c)
          : [iterations] "m"((iterations)), [kleft] "m"((kleft)), [a] "m"((a)),
            [b] "m"((b)), [c] "m"((c)), [alpha] "m"((alpha)), [beta] "m"((beta))
          : "r8", "r9", "r10", "r11", "r12", "r13", "ymm0", "ymm1", "ymm2",
            "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
            "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
#else
      std::vector<data_type> accum_tile(8 * 6, 0.0);
      for (std::size_t k = 0; k < K; k++) {
#pragma omp unroll(8)
        for (std::size_t mr = 0; mr < 8; mr++) {
#pragma omp unroll(6)
          for (std::size_t nr = 0; nr < 6; nr++) {
            accum_tile[nr * 8 + mr] +=
                packed_a[k * 8 + mr] * packed_b[k * 6 + nr];
          }
        }
      }
#pragma omp unroll(8)
      for (std::size_t mr = 0; mr < 8; mr++) {
#pragma omp unroll(6)
        for (std::size_t nr = 0; nr < 6; nr++) {
          c_tile[nr * 8 + mr] =
              beta_v * c_tile[nr * 8 + mr] + alpha_v * accum_tile[nr * 8 + mr];
        }
      }
#endif
#pragma omp unroll(8)
      for (std::size_t mr = 0; mr < 8; mr++) {
#pragma omp unroll(6)
        for (std::size_t nr = 0; nr < 6; nr++) {
          C[(m + mr) * ldc + n + nr] = c_tile[nr * 8 + mr];
        }
      }
    }
    for (std::size_t n = N - nrest; n < N; n++) {
#pragma omp unroll(8)
      for (std::size_t mr = 0; mr < 8; mr++) {
        double Caccum = 0.0;
#pragma omp simd reduction(+ : Caccum)
        for (std::size_t k = 0; k < K; k++) {
          Caccum += alpha_v * B[k * ldb + n] * A[k + (m + mr) * lda];
        }
        C[ldc * (m + mr) + n] = beta_v * C[ldc * (m + mr) + n] + Caccum;
      }
    }
  }
  for (std::size_t m = M - mrest; m < M; m++) {
    for (std::size_t n = 0; n < N; n++) {
      double Caccum = 0.0;
#pragma omp simd reduction(+ : Caccum)
      for (std::size_t k = 0; k < K; k++) {
        Caccum += alpha_v * B[k * ldb + n] * A[k + m * lda];
      }
      C[ldc * m + n] = beta_v * C[ldc * m + n] + Caccum;
    }
  }
}
