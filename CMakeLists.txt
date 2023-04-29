cmake_minimum_required(VERSION 3.20.1)

project(iwp-gunrock)

add_executable(${PROJECT_NAME})

set(SOURCE_LIST
    ${PROJECT_NAME}.cu
)
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_LIST})

add_subdirectory(externals/gunrock)

target_include_directories(${PROJECT_NAME} PUBLIC externals/gunrock/include)

target_link_libraries(${PROJECT_NAME} PRIVATE essentials)
get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.
