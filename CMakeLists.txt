cmake_minimum_required(VERSION 3.16)
project(tt_metal_playground LANGUAGES CXX)

# --- Check required environment variables ---
if(NOT DEFINED ENV{TT_METAL_HOME})
  message(FATAL_ERROR "TT_METAL_HOME is not set.")
endif()

if(NOT DEFINED ENV{TT_LOCK_SO})
  message(FATAL_ERROR "TT_LOCK_SO is not set.")
endif()

if(NOT EXISTS "$ENV{TT_LOCK_SO}")
  message(FATAL_ERROR "TT_LOCK_SO file does not exist: $ENV{TT_LOCK_SO}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common optimization flags
add_compile_options(-O3)

# Add custom CMake modules (for xtensor), if needed
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake/xtensor")

# Add TT-Metal CMake package directory to the search path
list(APPEND CMAKE_PREFIX_PATH "$ENV{TT_METAL_HOME}/build/lib/cmake")

# Find OpenMP (used by some examples)
find_package(OpenMP REQUIRED)

# Find tt-metalium package (exports TT::Metalium target)
find_package(tt-metalium REQUIRED)

# Import tt_lock shared library (required by TT runtime)
add_library(tt_lock SHARED IMPORTED)
set_target_properties(tt_lock PROPERTIES
  IMPORTED_LOCATION "$ENV{TT_LOCK_SO}"
)

# Bundle common dependencies into a single INTERFACE target
add_library(tt_example_deps INTERFACE)
target_link_libraries(tt_example_deps INTERFACE
  TT::Metalium
  tt_lock
  OpenMP::OpenMP_CXX
)
target_compile_features(tt_example_deps INTERFACE cxx_std_20)

# Helper function to add an example executable with common deps
function(add_tt_example target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE tt_example_deps)
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR})
endfunction()

# Add each example subdirectory
add_subdirectory(dummy)
add_subdirectory(dram_loopback)
