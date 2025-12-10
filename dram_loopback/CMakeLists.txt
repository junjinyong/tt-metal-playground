# Add TT-Metal cmake directory to search path
list(APPEND CMAKE_PREFIX_PATH "$ENV{TT_METAL_HOME}/build/lib/cmake")

# Find tt-metalium package
if(NOT TARGET TT::Metalium)
    find_package(tt-metalium REQUIRED)
endif()

# Create library
add_library(dram_loopback
    dram_loopback.cpp
)

target_include_directories(dram_loopback PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(dram_loopback PUBLIC
    TT::Metalium
)

# Set C++17 standard
target_compile_features(dram_loopback PUBLIC cxx_std_17)