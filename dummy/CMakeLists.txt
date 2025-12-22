# Add TT-Metal cmake directory to search path
list(APPEND CMAKE_PREFIX_PATH "$ENV{TT_METAL_HOME}/build/lib/cmake")

# Find tt-metalium package
if(NOT TARGET TT::Metalium)
    find_package(tt-metalium REQUIRED)
endif()

# Create library
add_library(dummy
    dummy.cpp
)

target_include_directories(dummy PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(dummy PUBLIC
    TT::Metalium
)

# Set C++17 standard
target_compile_features(dummy PUBLIC cxx_std_17)