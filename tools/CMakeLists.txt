add_subdirectory(vision-opt)
add_subdirectory(vision-translate)

# Ensure vision-opt and vision-translate are built with the default 'all' target
add_custom_target(vision_tools ALL
  DEPENDS
    vision-opt
    vision-translate
)
