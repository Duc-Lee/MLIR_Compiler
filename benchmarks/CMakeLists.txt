add_executable(bench_conv
  bench_conv.c
)

target_link_libraries(bench_conv PRIVATE
  vision_runtime
  m
)

target_include_directories(bench_conv PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
)

target_compile_options(bench_conv PRIVATE
  -O3 -mavx2 -mfma -Wall -Wextra
)

add_executable(bench_mlp
  bench_mlp.c
)

target_link_libraries(bench_mlp PRIVATE
  vision_runtime
  m
)

target_include_directories(bench_mlp PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
)

target_compile_options(bench_mlp PRIVATE
  -O3 -mavx2 -mfma -Wall -Wextra
)
