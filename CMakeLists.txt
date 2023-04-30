cmake_minimum_required(VERSION 3.20.1)

project(iwp-gunrock)

enable_language(CUDA)

add_subdirectory(externals/gunrock)

# ###################################################
# ############# SET CXX & CUDA FLAGS ################
# ###################################################
set(CXX_FLAGS
    $<$<CXX_COMPILER_ID:MSVC>:
    /W4
    >
    $<$<CXX_COMPILER_ID:GNU>:
    -Wall

    # -Wextra
    -Wno-unused-result
    -Wno-unused-local-typedefs
    -Wno-strict-aliasing
    -Wno-unused-function
    -Wno-format-security

    # -Werror
    # -vvv
    >
)

set(CUDA_RELEASE_FLAGS
    --expt-extended-lambda
    --expt-relaxed-constexpr
    --use_fast_math
)

set(CUDA_DEBUG_FLAGS
    --expt-extended-lambda
    --expt-relaxed-constexpr
    --ptxas-options -v
    --debug # Host debug
    --device-debug # Device debug
)

add_executable(${PROJECT_NAME} iwp-gunrock.cu)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} externals/gunrock/include)

target_link_libraries(${PROJECT_NAME}
    PRIVATE ${OpenCV_LIBS}
    PRIVATE essentials
)

get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY bin
    CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
)
