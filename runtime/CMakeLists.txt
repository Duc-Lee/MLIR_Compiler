add_library(vision_runtime STATIC
  src/vision_runtime.c
)

target_include_directories(vision_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_options(vision_runtime PRIVATE
  -O3 -Wall -Wextra
)
