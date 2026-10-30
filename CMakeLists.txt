cmake_minimum_required(VERSION 3.20)
project(MLIR_Compiler C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# uu tien tim kiem LLVM va MLIR tai .local neu he thong chua cai global
list(PREPEND CMAKE_PREFIX_PATH
  "/home/duc-le/.local/llvm-18/usr/lib/llvm-18"
  "/usr/lib/llvm-18"
)

# them include goc cua du an
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(MLIR CONFIG QUIET)
find_package(LLVM CONFIG QUIET)

if(MLIR_FOUND AND LLVM_FOUND)
  message(STATUS "Da tim thay LLVM ${LLVM_PACKAGE_VERSION} tai: ${LLVM_DIR}")
  message(STATUS "Da tim thay MLIR tai: ${MLIR_DIR}")

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
  message(WARNING "Chua tim thay MLIR dev package tren may.")
  message(STATUS "Huong dan cai dat tren Ubuntu/Debian: sudo apt install -y llvm-18-dev libmlir-18-dev mlir-18-tools clang-18")
  message(STATUS "Van tiep tuc build runtime C va benchmark kernel AVX2...")
endif()

# luon build runtime va benchmarks
add_subdirectory(runtime)
