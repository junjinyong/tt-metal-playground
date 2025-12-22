cmake_minimum_required(VERSION 3.16)

project(main LANGUAGES CXX)

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME environment variable is not set. Please set it to the TT-Metal installation directory.")
endif()

if(NOT DEFINED ENV{TT_LOCK_SO})
    message(FATAL_ERROR "TT_LOCK_SO environment variable is not set. Please set it to the path of libtt_lock.so")
endif()

if(NOT EXISTS "$ENV{TT_LOCK_SO}")
    message(FATAL_ERROR "TT_LOCK_SO file does not exist: $ENV{TT_LOCK_SO}")
endif()

message(STATUS "TT_METAL_HOME: $ENV{TT_METAL_HOME}")
message(STATUS "TT_LOCK_SO: $ENV{TT_LOCK_SO}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

# Add custom cmake modules (for xtensor)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake/xtensor")

# Add TT-Metal cmake directory to search path
list(APPEND CMAKE_PREFIX_PATH "$ENV{TT_METAL_HOME}/build/lib/cmake")

# Find OpenMP (required for parallel validation)
find_package(OpenMP REQUIRED)

# Find tt-metalium package
if(NOT TARGET TT::Metalium)
    find_package(tt-metalium REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR})


# Add tt-lock library (required)
add_library(tt_lock SHARED IMPORTED)
set_target_properties(tt_lock PROPERTIES
    IMPORTED_LOCATION $ENV{TT_LOCK_SO}
)

# Executables
add_executable(main
    main.cpp
)

target_link_libraries(main PUBLIC
    TT::Metalium
)

target_link_libraries(main PUBLIC
    tt_lock
)

target_include_directories(main PUBLIC
    ${CMAKE_SOURCE_DIR}
)

# Set C++17 standard
target_compile_features(main PUBLIC cxx_std_17)