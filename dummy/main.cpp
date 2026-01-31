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

// TT-Metal Host APIs
#include <tt-metalium/host_api.hpp>

// TT-Metal logging control
#include "umd/device/logging/config.hpp"


int main(int argc, char* argv[]) {
    // Suppress TT-Metal device info/warning messages
    tt::umd::logging::set_level(tt::umd::logging::level::error);

    // Initialize TT-Lock (acquire device locks)
    tt_lock_init();
    std::cout << "TT-Lock: Device locking enabled" << std::endl;
    
    std::cout << "\033[0;31m" << "Hello, World!" << "\033[0m" << std::endl;
    std::cout << "\033[0;31m" << "Number of Devices: " << tt::tt_metal::GetNumAvailableDevices() << "\033[0m" << std::endl;
    std::cout << "\033[0;31m" << "Number of PCIe Devices: " << tt::tt_metal::GetNumPCIeDevices() << "\033[0m" << std::endl;

    // Clean up and release device locks
    tt_lock_cleanup();

    return 0;
}
