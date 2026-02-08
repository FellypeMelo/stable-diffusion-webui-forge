#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <memory>

using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

// ======================================================================================
// 🚀 V5.7: Streaming DPAS Flash-Attention (Scalar Cast Fix)
// ======================================================================================

static constexpr int SG_SZ = 16;
static constexpr int HEAD_DIM = 64;
static constexpr int TILE_SZ = 16;

using T_Half = sycl::half;

template <typename T, int N>
ESIMD_INLINE esimd::simd<T, N> load_linear(T* ptr) {
    return esimd::block_load<T, N>(ptr);
}

template <typename T, int N>
ESIMD_INLINE void store_linear(T* ptr, esimd::simd<T, N> val) {
    esimd::block_store(ptr, val);
}

extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void attention_stream_esimd(
    T_Half* q_base, T_Half* k_base, T_Half* v_base,
    T_Half* out_base,
    int seq_len_q, int seq_len_kv, int num_heads, float scale,
    sycl::nd_item<1> item, sycl::local_accessor<float, 1> slm_scratch_f
) {
    const int group_id = item.get_group(0);
    const int chunks_per_q = (seq_len_q + TILE_SZ - 1) / TILE_SZ;
    if (chunks_per_q == 0) return;
    const int total_blocks = chunks_per_q * num_heads;
    
    int batch_id = group_id / total_blocks;
    int rem = group_id % total_blocks;
    int h_id = rem / chunks_per_q;
    int q_blk_id = rem % chunks_per_q;

    const int m_start = q_blk_id * TILE_SZ;
    if (m_start >= seq_len_q) return;

    long stride_h = (long)seq_len_q * HEAD_DIM;
    long stride_b = (long)num_heads * stride_h;
    long stride_kv_h = (long)seq_len_kv * HEAD_DIM;
    long stride_kv_b = (long)num_heads * stride_kv_h;

    T_Half* q_ptr = q_base + batch_id * stride_b + h_id * stride_h + q_blk_id * TILE_SZ * HEAD_DIM;
    T_Half* out_ptr = out_base + batch_id * stride_b + h_id * stride_h + q_blk_id * TILE_SZ * HEAD_DIM;
    T_Half* k_start = k_base + batch_id * stride_kv_b + h_id * stride_kv_h;
    T_Half* v_start = v_base + batch_id * stride_kv_b + h_id * stride_kv_h;

    // 1. Load Q
    esimd::simd<T_Half, 256> q_regs[4];
    #pragma unroll
    for(int d=0; d<4; ++d) {
        auto q_2d = q_regs[d].bit_cast_view<T_Half, 16, 16>();
        for(int r=0; r<16; ++r) {
             q_2d.row(r) = esimd::block_load<T_Half, 16>(q_ptr + r * HEAD_DIM + d * 16);
        }
    }

    esimd::simd<float, 256> out_acc[4];
    #pragma unroll
    for(int i=0; i<4; ++i) out_acc[i] = 0.0f;

    esimd::simd<float, 16> m_curr = -1e30f;
    esimd::simd<float, 16> l_curr = 0.0f;

    float* slm_ptr_f = slm_scratch_f.get_multi_ptr<access::decorated::no>().get();
    T_Half* slm_ptr_h = reinterpret_cast<T_Half*>(slm_ptr_f);

    // 2. Loop
    for (int k_off = 0; k_off < seq_len_kv; k_off += TILE_SZ) {
        
        esimd::simd<float, 256> s_acc = 0.0f;

        #pragma unroll
        for(int d=0; d<4; ++d) {
            T_Half* k_curr = k_start + k_off * HEAD_DIM + d * 16;
            
            esimd::simd<T_Half, 256> k_sub;
            auto k_2d = k_sub.bit_cast_view<T_Half, 16, 16>();
            for(int r=0; r<16; ++r) {
                if (k_off + r < seq_len_kv)
                     k_2d.row(r) = esimd::block_load<T_Half, 16>(k_curr + r * HEAD_DIM);
                else
                     k_2d.row(r) = 0;
            }

            esimd::simd<float, 128> c_top = s_acc.select<128, 1>(0);
            esimd::simd<float, 128> c_bot = s_acc.select<128, 1>(128);
            
            esimd::simd<T_Half, 128> q_top = q_regs[d].select<128, 1>(0);
            esimd::simd<T_Half, 128> q_bot = q_regs[d].select<128, 1>(128);

            c_top = xmx::dpas<8, 8, float, float, T_Half, T_Half>(c_top, k_sub, q_top);
            c_bot = xmx::dpas<8, 8, float, float, T_Half, T_Half>(c_bot, k_sub, q_bot);
            
            s_acc.select<128, 1>(0) = c_top;
            s_acc.select<128, 1>(128) = c_bot;
        }

        s_acc = s_acc * scale;
        
        esimd::simd<float, 16> m_new = -1e30f;
        auto s_2d = s_acc.bit_cast_view<float, 16, 16>();
        for(int r=0; r<16; ++r) {
             float max_val = -1e30f;
             for(int c=0; c<16; ++c) {
                 if (k_off + c < seq_len_kv) {
                     float val = s_2d.row(r)[c];
                     if(val > max_val) max_val = val;
                 }
             }
             m_new[r] = max_val;
        }

        esimd::simd<float, 16> m_prev = m_curr;
        m_curr = esimd::max(m_prev, m_new);

        esimd::simd<float, 16> alpha = 0.0f;
        for(int r=0; r<16; ++r) {
             float diff = m_prev[r] - m_curr[r];
             if (diff > -64.0f)
                 alpha[r] = sycl::exp((float)(diff));
        }

        #pragma unroll
        for(int i=0; i<4; ++i) {
            auto acc_view = out_acc[i].bit_cast_view<float, 16, 16>();
            for(int r=0; r<16; ++r) {
                // FIXED: Explicit cast from simd_view to float
                float alpha_scalar = (float)alpha[r]; 
                
                esimd::simd<float, 16> row = acc_view.row(r);
                row = row * alpha_scalar;
                acc_view.row(r) = row;
            }
        }

        l_curr = l_curr * alpha;
        
        esimd::simd<T_Half, 256> p_half;
        auto p_view = p_half.bit_cast_view<T_Half, 16, 16>();

        for(int r=0; r<16; ++r) {
             float row_sum = 0.0f;
             for(int c=0; c<16; ++c) {
                 if (k_off + c < seq_len_kv) {
                     float val = s_2d.row(r)[c];
                     float p = sycl::exp((float)(val - m_curr[r]));
                     p_view.row(r)[c] = (T_Half)p;
                     row_sum += p;
                 } else {
                     p_view.row(r)[c] = 0;
                 }
             }
             l_curr[r] += row_sum;
        }

        #pragma unroll
        for(int d=0; d<4; ++d) {
             T_Half* v_curr = v_start + k_off * HEAD_DIM + d * 16; 

             esimd::simd<T_Half, 256> v_reg;
             auto v_2d = v_reg.bit_cast_view<T_Half, 16, 16>();
             for(int r=0; r<16; ++r) {
                 if (k_off + r < seq_len_kv)
                     v_2d.row(r) = esimd::block_load<T_Half, 16>(v_curr + r * HEAD_DIM);
                 else
                     v_2d.row(r) = 0;
             }
             
             esimd::block_store(slm_ptr_h, v_reg);
             
             esimd::simd<T_Half, 256> v_trans;
             auto v_t_2d = v_trans.bit_cast_view<T_Half, 16, 16>();
             
             esimd::simd<uint32_t, 16> offsets;
             for(int i=0; i<16; ++i) offsets[i] = i * 16 * sizeof(T_Half);

             #pragma unroll
             for(int c=0; c<16; ++c) {
                  v_t_2d.row(c) = esimd::gather<T_Half, 16>(slm_ptr_h, offsets + c * sizeof(T_Half));
             }
             
             esimd::simd<float, 128> c_top = out_acc[d].select<128, 1>(0);
             esimd::simd<float, 128> c_bot = out_acc[d].select<128, 1>(128);
             
             esimd::simd<T_Half, 128> p_top = p_half.select<128, 1>(0);
             esimd::simd<T_Half, 128> p_bot = p_half.select<128, 1>(128);

             c_top = xmx::dpas<8, 8, float, float, T_Half, T_Half>(c_top, v_trans, p_top);
             c_bot = xmx::dpas<8, 8, float, float, T_Half, T_Half>(c_bot, v_trans, p_bot);
             
             out_acc[d].select<128, 1>(0) = c_top;
             out_acc[d].select<128, 1>(128) = c_bot;
        }
    }

    #pragma unroll
    for(int d=0; d<4; ++d) {
         auto acc_view = out_acc[d].bit_cast_view<float, 16, 16>();
         for(int r=0; r<16; ++r) {
             float inv = 1.0f / (l_curr[r] + 1e-6f);
             esimd::simd<float, 16> row = acc_view.row(r);
             row = row * inv;
             acc_view.row(r) = row;
         }

         esimd::simd<T_Half, 256> res_h = esimd::convert<T_Half>(out_acc[d]);
         auto res_2d = res_h.bit_cast_view<T_Half, 16, 16>();
         
         for(int r=0; r<16; ++r) {
             if (m_start + r < seq_len_q) {
                 esimd::block_store(out_ptr + d*16 + r*HEAD_DIM, res_2d.row(r));
             }
         }
    }
}

class AttentionKernelV5 {
public: // Public
    std::unique_ptr<queue> q_inst;
    AttentionKernelV5() {
        try { q_inst = std::make_unique<queue>(gpu_selector_v); } 
        catch (...) { q_inst = std::make_unique<queue>(); }
    }

    void run(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t out, int seq_q, int seq_kv, int num_heads, int dim_head, float scale, int batch_size) {
        int chunks_q = (seq_q + 16 - 1) / 16;
        size_t total_wgs = chunks_q * num_heads * batch_size;

        q_inst->submit([&](handler& cgh) {
            local_accessor<float, 1> slm(1024, cgh); 
            cgh.parallel_for(nd_range<1>(range<1>(total_wgs * 16), range<1>(16)), [=](nd_item<1> it) {
                attention_stream_esimd(
                    (T_Half*)q, (T_Half*)k, (T_Half*)v, (T_Half*)out,
                    seq_q, seq_kv, num_heads, scale, it, slm
                );
            });
        }).wait();
    }
};

PYBIND11_MODULE(attention_kernel_v5_dpas, m) {
    pybind11::class_<AttentionKernelV5>(m, "AttentionKernel")
        .def(pybind11::init<>())
        .def("run", &AttentionKernelV5::run);
}
