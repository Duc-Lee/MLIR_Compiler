cmake_minimum_required(VERSION 3.20)
project(MLIR_Compiler C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Portable search paths for LLVM and MLIR packages
if(DEFINED ENV{LLVM_DIR})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{LLVM_DIR}")
endif()
if(DEFINED ENV{MLIR_DIR})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{MLIR_DIR}")
endif()
if(DEFINED ENV{HOME})
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{HOME}/.local/llvm-18/usr/lib/llvm-18"
    "$ENV{HOME}/.local/llvm-19/usr/lib/llvm-19"
  )
endif()
list(APPEND CMAKE_PREFIX_PATH
  "/usr/lib/llvm-18"
  "/usr/lib/llvm-19"
  "/usr/local"
)

# Project root include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(MLIR CONFIG QUIET)
find_package(LLVM CONFIG QUIET)

if(MLIR_FOUND AND LLVM_FOUND)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at: ${LLVM_DIR}")
  message(STATUS "Found MLIR at: ${MLIR_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  link_directories(${LLVM_BUILD_LIBRARY_DIR})

  add_definitions(${LLVM_DEFINITIONS})

  add_subdirectory(lib)
  add_subdirectory(tools)
else()
  message(WARNING "MLIR development packages not found on system.")
  message(STATUS "To install on Ubuntu/Debian: sudo apt install -y llvm-18-dev libmlir-18-dev mlir-18-tools clang-18")
  message(STATUS "Continuing with runtime library and benchmark kernels...")
endif()

# Always build runtime library and benchmark suites
add_subdirectory(runtime)
add_subdirectory(benchmarks)
