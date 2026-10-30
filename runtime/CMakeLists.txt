add_library(vision_runtime STATIC
  src/vision_runtime.c
)

target_include_directories(vision_runtime PUBLIC
  include
)

target_compile_options(vision_runtime PRIVATE
  -O3 -Wall -Wextra
)

# benchmark thuc thi conv2d
add_executable(bench_conv
  ../benchmarks/bench_conv.c
)

target_link_libraries(bench_conv PRIVATE
  vision_runtime
  m
)

target_include_directories(bench_conv PRIVATE
  include
)

target_compile_options(bench_conv PRIVATE
  -O3 -mavx2 -mfma -Wall -Wextra
)

# benchmark thuc thi neural network mlp
add_executable(bench_mlp
  ../benchmarks/bench_mlp.c
)

target_link_libraries(bench_mlp PRIVATE
  vision_runtime
  m
)

target_include_directories(bench_mlp PRIVATE
  include
)

target_compile_options(bench_mlp PRIVATE
  -O3 -mavx2 -mfma -Wall -Wextra
)
