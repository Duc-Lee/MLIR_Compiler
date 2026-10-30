add_mlir_tool(vision-opt
  vision-opt.cpp
)

target_link_libraries(vision-opt PRIVATE
  VisionTransforms
  VisionPipeline
  MLIR
)
