add_mlir_tool(vision-translate
  vision-translate.cpp
)

target_link_libraries(vision-translate PRIVATE
  MLIR
)
