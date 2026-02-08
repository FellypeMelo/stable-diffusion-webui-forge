#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <chrono>

/**
 * @file attention_kernel.cpp
 * @brief Phase 5 "Atomic XMX" (Xe2) - Standardized
 */

namespace esimd = sycl::ext::intel::esimd;
namespace esimd_exp = sycl::ext::intel::experimental::esimd;
using namespace sycl;

struct TypeEngine {
    using bf16_t = sycl::ext::oneapi::bfloat16;
    static ESIMD_INLINE float safe_exp(float x) { return std::exp(x); }
};

extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void attention_kernel_xe2(
    TypeEngine::bf16_t* q_global, TypeEngine::bf16_t* k_global, TypeEngine::bf16_t* v_global,
    float* out_global, const int seq_len, const int num_heads,
    sycl::nd_item<1> item
) {
    const int group_id = item.get_group(0);
    const int num_row_chunks = (seq_len + 7) / 8;
    const int wgs_per_batch = num_row_chunks * num_heads;
    
    const int b_id = group_id / wgs_per_batch;
    const int rem_group = group_id % wgs_per_batch;
    
    const int h_id = rem_group % num_heads;
    const int row_chunk = rem_group / num_heads;
    const int m_start = row_chunk * 8;

    if (m_start >= seq_len) return;

    // Correct offset for Batch + Head
    const int global_head_idx = b_id * num_heads + h_id;
    const int bh_offset = global_head_idx * seq_len * 64;
    
    TypeEngine::bf16_t* q_ptr = q_global + bh_offset + m_start * 64;
    TypeEngine::bf16_t* k_ptr = k_global + bh_offset;
    TypeEngine::bf16_t* v_ptr = v_global + bh_offset;
    float* o_ptr = out_global + bh_offset + m_start * 64;

    esimd::simd<float, 8> m_curr = -1e38f;
    esimd::simd<float, 8> l_curr = 0.0f;
    esimd::simd<float, 512> acc_o = 0.0f; 
    
    esimd::simd<TypeEngine::bf16_t, 512> q_reg = 0;
    // Pitch 63 (64 elements) for D=64
    q_reg.template select<256, 1>(0) = esimd_exp::lsc_load_2d<TypeEngine::bf16_t, 32, 8, 1, false, false>(q_ptr, 63, 65535, 63, 0, 0).read();
    q_reg.template select<256, 1>(256) = esimd_exp::lsc_load_2d<TypeEngine::bf16_t, 32, 8, 1, false, false>(q_ptr, 63, 65535, 63, 32, 0).read();

    for (int t = 0; t < (seq_len + 63) / 64; ++t) {
        esimd::simd<float, 512> scores = 0.0f; 
        
        #pragma unroll
        for (int k_idx = 0; k_idx < 4; ++k_idx) { 
            auto k_vnni = esimd_exp::lsc_load_2d<TypeEngine::bf16_t, 16, 16, 1, false, true>(k_ptr, 63, 65535, 63, k_idx * 16, t * 64).read();
            auto q_sub = q_reg.template select<128, 1>(k_idx * 128).read(); 
            
            #pragma unroll
            for (int c_idx = 0; c_idx < 4; ++c_idx) { 
                auto acc_a = scores.template select<8, 1>(c_idx * 128 + 0).read();
                scores.template select<8, 1>(c_idx * 128 + 0) = esimd::xmx::dpas<8, 8, float, float, TypeEngine::bf16_t, TypeEngine::bf16_t>(
                    acc_a, k_vnni.template select<128, 1>(0).read(), q_sub);
                
                auto acc_b = scores.template select<8, 1>(c_idx * 128 + 64).read();
                scores.template select<8, 1>(c_idx * 128 + 64) = esimd::xmx::dpas<8, 8, float, float, TypeEngine::bf16_t, TypeEngine::bf16_t>(
                    acc_b, k_vnni.template select<128, 1>(128).read(), q_sub);
            }
        }

        #pragma unroll
        for(int r=0; r<8; ++r) {
            esimd::simd<float, 64> row_s = scores.template select<64, 1>(r*64).read();
            float m_t = esimd::hmax<float>(row_s);
            float m_p = m_curr[r];
            float m_new = (m_p > m_t) ? m_p : m_t;
            float alpha = TypeEngine::safe_exp(m_p - m_new);
            float beta = TypeEngine::safe_exp(m_t - m_new);
            
            esimd::simd<float, 64> p_v = esimd::exp(row_s - m_t);
            l_curr[r] = l_curr[r] * alpha + beta * esimd::reduce<float>(p_v, std::plus<>());
            
            esimd::simd<float, 64> row_o = acc_o.template select<64, 1>(r*64).read();
            acc_o.template select<64, 1>(r*64) = row_o * alpha;
            scores.template select<64, 1>(r*64) = p_v * beta;
            m_curr[r] = m_new;
        }

        esimd::simd<TypeEngine::bf16_t, 512> p_reg = esimd::convert<TypeEngine::bf16_t>(scores);
        #pragma unroll
        for (int v_idx = 0; v_idx < 4; ++v_idx) {
            auto v_vnni = esimd_exp::lsc_load_2d<TypeEngine::bf16_t, 16, 16, 1, false, true>(v_ptr, 63, 65535, 63, v_idx * 16, t * 64).read();
            auto p_sub = p_reg.template select<128, 1>(v_idx * 128).read();
            
            #pragma unroll
            for (int c_idx = 0; c_idx < 4; ++c_idx) {
                auto acc_va = acc_o.template select<8, 1>(c_idx * 128 + 0).read();
                acc_o.template select<8, 1>(c_idx * 128 + 0) = esimd::xmx::dpas<8, 8, float, float, TypeEngine::bf16_t, TypeEngine::bf16_t>(
                    acc_va, v_vnni.template select<128, 1>(0).read(), p_sub);
                
                auto acc_vb = acc_o.template select<8, 1>(c_idx * 128 + 64).read();
                acc_o.template select<8, 1>(c_idx * 128 + 64) = esimd::xmx::dpas<8, 8, float, float, TypeEngine::bf16_t, TypeEngine::bf16_t>(
                    acc_vb, v_vnni.template select<128, 1>(128).read(), p_sub);
            }
        }
    }

    #pragma unroll
    for(int r=0; r<8; ++r) {
        if (m_start + r < seq_len) {
            esimd::simd<float, 64> fin = acc_o.template select<64, 1>(r*64).read();
            float den = l_curr[r];
            esimd::block_store<float, 64>(o_ptr + r*64, fin / den);
        }
    }
}

class AttentionKernel {
public:
    double benchmark(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t out, int b, int h, int s, int d, int iter) {
        queue qs{gpu_selector_v};
        int wgs = ((s + 7) / 8) * h * b;
        
        for(int i=0; i<5; ++i) {
            qs.submit([&](handler& cgh) {
                cgh.parallel_for(nd_range<1>(range<1>(wgs), range<1>(1)), [=](nd_item<1> it) {
                    attention_kernel_xe2((TypeEngine::bf16_t*)q, (TypeEngine::bf16_t*)k, (TypeEngine::bf16_t*)v, (float*)out, s, h, it);
                });
            });
        }
        qs.wait();

        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<iter; ++i) {
            qs.submit([&](handler& cgh) {
                cgh.parallel_for(nd_range<1>(range<1>(wgs), range<1>(1)), [=](nd_item<1> it) {
                    attention_kernel_xe2((TypeEngine::bf16_t*)q, (TypeEngine::bf16_t*)k, (TypeEngine::bf16_t*)v, (float*)out, s, h, it);
                });
            });
        }
        qs.wait();
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() / iter;
    }
};

PYBIND11_MODULE(attention_kernel, m) {
    pybind11::class_<AttentionKernel>(m, "AttentionKernel").def(pybind11::init<>()).def("benchmark", &AttentionKernel::benchmark);
}