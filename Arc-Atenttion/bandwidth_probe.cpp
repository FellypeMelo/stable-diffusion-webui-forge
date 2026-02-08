// bandwidth_probe.cpp
// ESIMD Bandwidth Probe Kernel for Intel Arc B580 (Battlemage/Xe2)
// Target: >350 GB/s sustained on 456 GB/s theoretical

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>

namespace esimd = sycl::ext::intel::esimd;
using namespace sycl;

// Architecture-tuned constants for Xe2/Battlemage
constexpr int SIMD_WIDTH = 16;           // Native SIMD16 execution
constexpr int BYTES_PER_REG = 64;        // 512-bit GRF = 64 bytes
constexpr int BYTES_PER_THREAD = 64;     // Full GRF utilization per lane
constexpr int BYTES_PER_SIMD = SIMD_WIDTH * BYTES_PER_THREAD;  // 1024 bytes/SIMD

// LSC block load/store work best with 64-byte aligned, 64-256 byte transactions
// For maximum bandwidth: use 256-byte (4 GRFs) per lane = full cache line
constexpr int LSC_BLOCK_SIZE = 256;      // Optimal LSC transaction size
constexpr int REGS_PER_BLOCK = LSC_BLOCK_SIZE / BYTES_PER_THREAD;  // 4 registers

// Tile dimensions - must be large enough to saturate memory pipeline
// B580 has 20 Xe-cores, each with 16 EUs @ SIMD16 = 320 concurrent SIMDs
// We want ~10-20MB working set to blow through caches and measure raw BW
constexpr int TILE_M = 4096;             // Rows per tile
constexpr int TILE_K_BYTES = 4096;       // Bytes per row (must be 64-byte aligned)

// Verify alignment constraints
static_assert(TILE_K_BYTES % 64 == 0, "Row size must be 64-byte aligned");
static_assert(LSC_BLOCK_SIZE % 64 == 0, "LSC block must be cache-line aligned");

// ESIMD Bandwidth Probe Kernel
// Pure memory copy: Global -> Registers -> Global
// Uses explicit LSC block loads/stores with streaming hints
template <typename T>
SYCL_EXTERNAL void bandwidth_probe_kernel(
    T* __restrict__ src,
    T* __restrict__ dst,
    int num_rows,
    int bytes_per_row,
    sycl::item<1> item
) {
    // Global ID is the row index directly
    const int row = item.get_id(0);
    
    if (row < num_rows) {
        const uint64_t row_offset = static_cast<uint64_t>(row) * bytes_per_row;
        const T* row_src = src + (row_offset / sizeof(T));
        T* row_dst = dst + (row_offset / sizeof(T));
        
        // Number of LSC blocks per row
        const int blocks_per_row = bytes_per_row / LSC_BLOCK_SIZE;
        
        // Linear process of all blocks in the row
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const uint64_t block_offset = static_cast<uint64_t>(block_idx) * LSC_BLOCK_SIZE;
            const T* load_addr = reinterpret_cast<const T*>(
                reinterpret_cast<const uint8_t*>(row_src) + block_offset
            );
            T* store_addr = reinterpret_cast<T*>(
                reinterpret_cast<uint8_t*>(row_dst) + block_offset
            );
            
            // Explicit LSC block load with streaming hint (non-temporal)
            // This bypasses L1/L2 cache pollution for bandwidth-bound workloads
            esimd::properties load_props{
                esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                esimd::cache_hint_L2<esimd::cache_hint::uncached>
            };
            auto loaded = esimd::block_load<
                T, 
                LSC_BLOCK_SIZE / sizeof(T)
            >(load_addr, load_props);
            
            // Explicit LSC block store with write-back hint
            // Uses write-combining buffer for efficient memory subsystem utilization
            esimd::properties store_props{
                esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                esimd::cache_hint_L2<esimd::cache_hint::write_back>
            };
            esimd::block_store<
                T,
                LSC_BLOCK_SIZE / sizeof(T)
            >(store_addr, loaded, store_props);
        }
    }
}

// Python-exposed wrapper
class BandwidthProbe {
public:
    BandwidthProbe() : q_(sycl::default_selector_v, sycl::property::queue::enable_profiling()) {
        auto dev = q_.get_device();
        if (!dev.has(sycl::aspect::ext_intel_esimd)) {
            throw std::runtime_error("Device does not support ESIMD");
        }
    }
    
    float run_probe(uint64_t src_addr, uint64_t dst_addr, int num_rows, int num_cols, int iterations) {
        const size_t total_bytes = num_rows * num_cols * sizeof(float);
        
        if (total_bytes % 64 != 0) {
            throw std::runtime_error("Total size must be 64-byte aligned");
        }
        
        // Get raw pointers from integer addresses
        float* src_ptr = reinterpret_cast<float*>(src_addr);
        float* dst_ptr = reinterpret_cast<float*>(dst_addr);
        
        // Launch one ESIMD thread per row
        // Each thread handles 4KB (or more) of data
        // For 8192 rows, we have 8192 threads.
        // ESIMD threads are heavyweight (register rich).
        
        q_.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::range<1>(num_rows),
                [=](sycl::item<1> item) {
                    bandwidth_probe_kernel(src_ptr, dst_ptr, num_rows, num_cols * sizeof(float), item);
                }
            );
        }).wait();
        
        // Timed iterations
        std::vector<sycl::event> events;
        events.reserve(iterations);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            auto e = q_.submit([&](sycl::handler& h) {
                h.parallel_for(
                    sycl::range<1>(num_rows),
                    [=](sycl::item<1> item) {
                        bandwidth_probe_kernel(src_ptr, dst_ptr, num_rows, num_cols * sizeof(float), item);
                    }
                );
            });
            events.push_back(e);
        }
        
        q_.wait();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate bandwidth using GPU event timing (more accurate)
        float total_gpu_time_ms = 0.0f;
        for (auto& e : events) {
            auto start_ns = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end_ns = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_gpu_time_ms += (end_ns - start_ns) / 1e6f;
        }
        
        // Each iteration: read total_bytes + write total_bytes = 2x total_bytes
        const double total_gb = (2.0 * total_bytes * iterations) / (1024.0 * 1024.0 * 1024.0);
        const double avg_time_s = total_gpu_time_ms / (iterations * 1000.0);
        const double bandwidth_gbps = total_gb / avg_time_s;
        
        return static_cast<float>(bandwidth_gbps);
    }
    
private:
    sycl::queue q_;
};

PYBIND11_MODULE(bandwidth_probe, m) {
    m.doc() = "ESIMD Bandwidth Probe for Intel Arc B580";
    pybind11::class_<BandwidthProbe>(m, "BandwidthProbe")
        .def(pybind11::init<>())
        .def("run_probe", &BandwidthProbe::run_probe, "Run bandwidth probe",
             pybind11::arg("src_addr"), pybind11::arg("dst_addr"), 
             pybind11::arg("num_rows"), pybind11::arg("num_cols"), 
             pybind11::arg("iterations") = 10);
}
