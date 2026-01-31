#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <ranges>
#include <stdexcept>

// TT-Lock device locking (required)
extern "C" void tt_lock_init();
extern "C" void tt_lock_cleanup();

// TT-Metal APIs
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/span.hpp>

// TT-Metal logging control
#include "umd/device/logging/config.hpp"

inline bool nearly_equal(const bfloat16 a_bf16, const bfloat16 b_bf16, float abs_tol = 1e-6f, float rel_tol = 1e-5f) {
    const auto a = static_cast<float>(a_bf16);
    const auto b = static_cast<float>(b_bf16);

    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }

    const float diff = std::fabs(a - b);
    if (diff <= abs_tol) {
        return true;
    }

    const float max_mag = std::max(std::fabs(a), std::fabs(b));
    return diff <= rel_tol * max_mag;
}

int main(int argc, char* argv[]) {
    // Suppress TT-Metal device info/warning messages
    tt::umd::logging::set_level(tt::umd::logging::level::error);

    // Initialize TT-Lock (acquire device locks)
    tt_lock_init();
    std::cout << "TT-Lock: Device locking enabled" << std::endl;



    // Part1: Device initialization && Program setup
    // Initialize Mesh Device
    constexpr int device_id = 0;
    if (tt::tt_metal::GetNumAvailableDevices() == 0) {
        throw std::runtime_error("No device found");
    }
    const auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

    // Create Program
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device -> mesh_command_queue();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();



    // Part 2: Create Buffers in DRAM and SRAM
    // Compute Buffer Size
    constexpr uint32_t num_tiles = 50;
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
    constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;
    if (tt::constants::TILE_WIDTH != 32 || tt::constants::TILE_HEIGHT != 32) {
        throw std::runtime_error("Tile size should be 32 x 32");
    }

    // On-chip Buffer Allocation
    tt::tt_metal::distributed::DeviceLocalBufferConfig l1_config {
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::L1
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig l1_buffer_config {
        .size = tile_size_bytes
    };
    auto l1_buffer = tt::tt_metal::distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());

    // Off-chip Buffer Allocation
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config {
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig dram_buffer_config {
        .size = dram_buffer_size
    };
    auto input_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto output_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());


    // Part3: Sending Data to DRAM
    std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
    std::mt19937 rng(std::random_device{}());   
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    for (auto& val : input_vec) {
        val = bfloat16(distribution(rng));
    }
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, false);

    constexpr tt::tt_metal::CoreCoord core = {0, 0};
    std::vector<uint32_t> dram_copy_compile_time_args;
    const auto& input_buffer = *(input_dram_buffer -> get_backing_buffer)();
    const auto& output_buffer = *(output_dram_buffer -> get_backing_buffer)();
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(dram_copy_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(dram_copy_compile_time_args);

    tt::tt_metal::DataMovementConfig dataflow_config {
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = dram_copy_compile_time_args
    };
    tt::tt_metal::KernelHandle dram_copy_kernel_id = CreateKernel(
        program,
        "dram_loopback/kernels/dataflow/loopback_dram_copy.cpp",
        core,
        dataflow_config
    );

    // Setting Runtime Arguments
    const std::vector runtime_args = {
        static_cast<uint32_t>((l1_buffer -> address)()),
        static_cast<uint32_t>((input_dram_buffer -> address)()),
        static_cast<uint32_t>((output_dram_buffer -> address)()),
        static_cast<uint32_t>(num_tiles)
    };
    tt::tt_metal::SetRuntimeArgs(program, dram_copy_kernel_id, core, ttsl::make_const_span(runtime_args));

    // Running the Program
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt::tt_metal::distributed::Finish(cq);

    // Download and Verity the Result
    std::vector<bfloat16> result_vec;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, result_vec, output_dram_buffer, true);

    bool pass = false;
    if (input_vec.size() == result_vec.size()) {
        pass = true;
        if (std::memcmp(input_vec.data(), result_vec.data(), input_vec.size() * sizeof(bfloat16)) == 0) {
            std::cout << "\033[0;31m" << "Correct (Bitwise)" << "\033[0m" << std::endl;
        } else if (std::ranges::equal(input_vec, result_vec)) {
            std::cout << "\033[0;31m" << "Correct (Exact)" << "\033[0m" << std::endl;
        } else if (std::ranges::equal(input_vec, result_vec, [&](auto x, auto y){ return nearly_equal(x, y); })) {
            std::cout << "\033[0;31m" << "Correct (Approx)" << "\033[0m" << std::endl;
        } else {
            pass = false;
        }
    }
    if (pass == false) {
        std::cout << "\033[0;31m" << "Incorrect" << "\033[0m" << std::endl;
    }

    if ((mesh_device -> close)() == false) {
        throw std::runtime_error("Device close failed");
    }


    // Clean up and release device locks
    tt_lock_cleanup();

    return 0;
}
