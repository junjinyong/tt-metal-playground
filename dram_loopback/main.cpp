#include <functional>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// TT-Lock device locking (required)
extern "C" void tt_lock_init();
extern "C" void tt_lock_cleanup();

// Common types for datatype and fidelity
#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/types.hpp>

// TT-Metal logging control
#include "umd/device/logging/config.hpp"

#include "dram_loopback.cpp"


int main(int argc, char* argv[]) {
    // Suppress TT-Metal device info/warning messages
    tt::umd::logging::set_level(tt::umd::logging::level::error);
    // tt::umd::logging::set_level(tt::umd::logging::level::debug);

    // Initialize TT-Lock (acquire device locks)
    tt_lock_init();
    std::cout << "TT-Lock: Device locking enabled" << std::endl;

    std::cout << "Hello, World!" << ' ' << g() << std::endl;

    // Clean up and release device locks
    tt_lock_cleanup();

    return 0;
}