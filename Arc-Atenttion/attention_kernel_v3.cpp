#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>

namespace esimd = sycl::ext::intel::esimd;
namespace esimd_exp = sycl::ext::intel::experimental::esimd;
using namespace sycl;

// Hardware Constants for Arc Battlemage (Xe2) / Alchemist (Xe1)
static constexpr int SIMD_WIDTH = 16;       // Execution width
static constexpr int HEAD_DIM = 64;         // Fixed for SD 1.5/XL (standard)
static constexpr int Q_BLK = 16;            // Rows of Q processed per thread-block (Reduced for JIT Stability)
static constexpr int KV_BLK = 64;           // Cols of K/V processed per block (XMX friendly)

struct TypeEngine {
    // Switch to Native Half (FP16) as requested by User
    using T = sycl::half; 
};

// Helper: Naive implementation of exp/max
// Robust safe_exp with clamping to avoid Inf/NaN
static inline float safe_exp(float x) {
    // Clamp input to avoid overflow. exp(88.0) is near float max.
    // For stability, we clamp slightly lower.
    if (x > 80.0f) x = 80.0f; 
    if (x < -80.0f) x = -80.0f; // exp(-80) is effectively 0
    return std::exp(x);
}

// ======================================================================================
// 🧠 Fused Flash-Cross-Attention Kernel (SLM Cached K/V)
// ======================================================================================
extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void attention_fused_xe2(
    TypeEngine::T* q_base, TypeEngine::T* k_base, TypeEngine::T* v_base,
    TypeEngine::T* out_base,
    int seq_len_q, int seq_len_kv, int num_heads, float scale,
    sycl::nd_item<1> item, sycl::local_accessor<TypeEngine::T, 1> slm_kv
) {
    // ----------------------------------------------------------------------
    // 1. Coordinates & Scheduling
    // ----------------------------------------------------------------------
    const int group_id = item.get_group(0); // Flattened Group ID
    
    // Grid: [Seq_Q_Chunks, Heads] * Batch
    // We assume the caller maps this 1D grid to logical dims
    // Simplest mapping: h_id varies fastest
    const int h_id = group_id % num_heads;
    const int flattened_chunk = group_id / num_heads;
    const int q_chunk_idx = flattened_chunk % ((seq_len_q + Q_BLK - 1) / Q_BLK);
    const int b_id = flattened_chunk / ((seq_len_q + Q_BLK - 1) / Q_BLK);

    // Row Start for this Q-Block
    const int m_start = q_chunk_idx * Q_BLK;
    if (m_start >= seq_len_q) return;

    // Pointers for this Batch/Head
    // Stride assumptions: [Batch, Head, Seq, Dim] contiguous surfaces (re-arranged by Python wrapper)
    long q_offset = ((long)b_id * num_heads * seq_len_q + (long)h_id * seq_len_q + m_start) * HEAD_DIM;
    long k_offset = ((long)b_id * num_heads * seq_len_kv + (long)h_id * seq_len_kv) * HEAD_DIM;
    long v_offset = ((long)b_id * num_heads * seq_len_kv + (long)h_id * seq_len_kv) * HEAD_DIM;
    long out_offset = q_offset; // Output matches Q layout

    TypeEngine::T* q_ptr = q_base + q_offset;
    TypeEngine::T* k_ptr = k_base + k_offset;
    TypeEngine::T* v_ptr = v_base + v_offset;
    TypeEngine::T* o_ptr = out_base + out_offset;

    // ----------------------------------------------------------------------
    // 2. SLM Pre-Fetch (Cache K & V)
    // ----------------------------------------------------------------------
    
    // Calculate offsets in SLM
    uint32_t kv_size_elems = seq_len_kv * HEAD_DIM;
    // SYCL 2020 Replacement for get_pointer()
    auto slm_base = slm_kv.get_multi_ptr<sycl::access::decorated::no>().get();
    TypeEngine::T* slm_k_ptr = slm_base; 
    TypeEngine::T* slm_v_ptr = slm_base + kv_size_elems;

    // Vectorized Load of K/V into SLM
    // We load in chunks of 128 elements (256 bytes)
    for (int i = 0; i < kv_size_elems; i += 128) {
        int remain = kv_size_elems - i;
        if (remain >= 128) {
            esimd::block_store(slm_k_ptr + i, esimd::block_load<TypeEngine::T, 128>(k_ptr + i));
            esimd::block_store(slm_v_ptr + i, esimd::block_load<TypeEngine::T, 128>(v_ptr + i));
        } else {
            // Tail handling
            for(int j=0; j<remain; ++j) {
                slm_k_ptr[i+j] = k_ptr[i+j];
                slm_v_ptr[i+j] = v_ptr[i+j];
            }
        }
    }
    // esimd::barrier(); // Not needed if WG=1

    // ----------------------------------------------------------------------
    // 3. Load Q-Tile (Registers)
    // ----------------------------------------------------------------------
    // Load 32 rows x 64 head_dim
    esimd::simd<TypeEngine::T, Q_BLK * HEAD_DIM> q_reg;
    auto q_2d = q_reg.bit_cast_view<TypeEngine::T, Q_BLK, HEAD_DIM>();

    #pragma unroll
    for (int r = 0; r < Q_BLK; ++r) {
        if (m_start + r < seq_len_q) {
            // Load row 'r' via block load (assuming contiguous row due to layout transpose)
            q_2d.row(r) = esimd::block_load<TypeEngine::T, HEAD_DIM>(q_ptr + r * HEAD_DIM);
        } else {
            q_2d.row(r) = 0; // Pad with zeros
        }
    }
    
    // Scale Q immediately (1/sqrt(d))
    // Convert to Float for Math
    esimd::simd<TypeEngine::T, Q_BLK * HEAD_DIM> q_scaled;
    auto q_f = esimd::convert<float>(q_reg);
    q_f = q_f * scale; 
    q_scaled = esimd::convert<TypeEngine::T>(q_f);
    auto q_scaled_view = q_scaled.bit_cast_view<TypeEngine::T, Q_BLK, HEAD_DIM>();

    // ----------------------------------------------------------------------
    // 4. Attention Loop (Score -> Softmax -> Output)
    // ----------------------------------------------------------------------
    
    // Accumulators for Output (Always Float32 for Precision)
    esimd::simd<float, Q_BLK * HEAD_DIM> acc_out = 0.0f;
    
    // Online Softmax State
    // Use finite min for fast-math compatibility
    esimd::simd<float, Q_BLK> max_score = -3.4028235e38f;
    esimd::simd<float, Q_BLK> sum_exp = 0.0f;

    // Iterate over K/V blocks (from SLM)
    const int K_ITER_BLK = 16; // MATCH Q_BLK for simplicity and register pressure 
    
    for (int k_off = 0; k_off < seq_len_kv; k_off += K_ITER_BLK) {
        int k_curr = (k_off + K_ITER_BLK > seq_len_kv) ? (seq_len_kv - k_off) : K_ITER_BLK;
        
        // 4a. Compute Scores = Q * K^T
        esimd::simd<float, Q_BLK * K_ITER_BLK> scores = 0.0f;
        auto scores_view = scores.bit_cast_view<float, Q_BLK, K_ITER_BLK>();
        
        // Naive MatMul (can be upgraded to DPAS if K is layout-transformed)
        for(int r=0; r<Q_BLK; ++r) {
            for(int c=0; c<k_curr; ++c) {
                // Loading K from SLM
                esimd::simd<TypeEngine::T, HEAD_DIM> k_vec = 
                    esimd::block_load<TypeEngine::T, HEAD_DIM>(slm_k_ptr + (k_off + c) * HEAD_DIM);
                
                // Explicit conversion to float for dot product
                auto prod = esimd::convert<float>(q_scaled_view.row(r).read()) * esimd::convert<float>(k_vec);
                float dot = esimd::reduce<float>(prod, std::plus<>());
                
                scores_view.select<1, 1, 1, 1>(r, c) = dot;
            }
        }
        
        // 4b. Online Softmax Update
        for(int r=0; r<Q_BLK; ++r) {
            // Find max in this new chunk
            // Use finite min
            float local_max = -3.4028235e38f;
            for(int c=0; c<k_curr; ++c) {
                float val = scores_view.select<1,1,1,1>(r,c).read()[0];
                if (val > local_max) local_max = val;
            }
            
            // Update global max/sum
            float old_max = max_score[r];
            float new_max = (local_max > old_max) ? local_max : old_max;
            
            float alpha = safe_exp(old_max - new_max); 
            float beta = safe_exp(local_max - new_max); 
            
            sum_exp[r] = sum_exp[r] * alpha;
            auto row_out = acc_out.bit_cast_view<float, Q_BLK, HEAD_DIM>().row(r);
            row_out = row_out * alpha;
            acc_out.bit_cast_view<float, Q_BLK, HEAD_DIM>().row(r) = row_out;

            // 4c. Accumulate V (Prob * V)
            for(int c=0; c<k_curr; ++c) {
                float s = scores_view.select<1,1,1,1>(r,c).read()[0];
                float p = safe_exp(s - new_max);
                sum_exp[r] += p; 
                
                // Load V vector from SLM
                esimd::simd<TypeEngine::T, HEAD_DIM> v_vec = 
                    esimd::block_load<TypeEngine::T, HEAD_DIM>(slm_v_ptr + (k_off + c) * HEAD_DIM);
                
                auto v_f = esimd::convert<float>(v_vec);
                acc_out.bit_cast_view<float, Q_BLK, HEAD_DIM>().row(r) += v_f * p;
            }
            max_score[r] = new_max;
        }
    }

    // ----------------------------------------------------------------------
    // 5. Final Normalization & Store
    // ----------------------------------------------------------------------
    auto out_view = acc_out.bit_cast_view<float, Q_BLK, HEAD_DIM>();
    
    #pragma unroll
    for(int r=0; r<Q_BLK; ++r) {
        if (m_start + r < seq_len_q) {
            // FIX: Add Epsilon to prevent division by zero (NaN)
            float inv_sum = 1.0f / (sum_exp[r] + 1e-6f);
            
            auto res_f = out_view.row(r) * inv_sum;
            
            esimd::simd<TypeEngine::T, HEAD_DIM> res = esimd::convert<TypeEngine::T>(res_f);
            esimd::block_store(o_ptr + r * HEAD_DIM, res);
        }
    }
}

class AttentionKernel {
    std::unique_ptr<queue> q_inst;
    
public:
    AttentionKernel() {
        try {
            q_inst = std::make_unique<queue>(gpu_selector_v);
        } catch (...) {
            q_inst = std::make_unique<queue>();
        }
    }

    void run(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t out, int seq_q, int seq_kv, int num_heads, int dim_head, float scale) {
        // Grid Calculation
        int num_chunks = (seq_q + Q_BLK - 1) / Q_BLK;
        int total_wgs = num_chunks * num_heads; 
        
        // SLM Size: (Seq_KV * Head_Dim) * 2 (K+V) * 2 (Bytes)
        // Ensure this fits! 77 * 64 * 2 * 2 = ~20KB. Fine.
        int slm_elements = 2 * seq_kv * dim_head; 

        q_inst->submit([&](handler& cgh) {
            local_accessor<TypeEngine::T, 1> slm(slm_elements, cgh);
            
            cgh.parallel_for(nd_range<1>(range<1>(total_wgs), range<1>(1)), [=](nd_item<1> it) {
                attention_fused_xe2(
                    (TypeEngine::T*)q, 
                    (TypeEngine::T*)k, 
                    (TypeEngine::T*)v, 
                    (TypeEngine::T*)out, 
                    seq_q, seq_kv, num_heads, scale,
                    it, slm
                );
            });
        }).wait();
    }
    
    // Benchmark stub
    double benchmark(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t out, int b, int h, int s, int d, int iter) {
        return 0.0f; 
    }
};

PYBIND11_MODULE(attention_kernel_v3, m) {
    pybind11::class_<AttentionKernel>(m, "AttentionKernel")
        .def(pybind11::init<>())
        .def("run", &AttentionKernel::run)
        .def("benchmark", &AttentionKernel::benchmark);
}