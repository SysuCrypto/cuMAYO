cmake_minimum_required(VERSION 3.18)
project(mayo_tests LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT CMAKE_CUDA_ARCHITECTURES)
    #set(CMAKE_CUDA_ARCHITECTURES 86) #3090
    set(CMAKE_CUDA_ARCHITECTURES 89) #4090
endif ()
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

set(SRC_DIR     ${CMAKE_CURRENT_LIST_DIR}/../src)
set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include)

set(MAYO_SOURCES
        ${SRC_DIR}/fips202.cu
        ${SRC_DIR}/aes128_ctr.cu
        ${SRC_DIR}/keypair.cu
        ${SRC_DIR}/gf16.cu
        ${SRC_DIR}/api.cu
        ${SRC_DIR}/expandsk.cu
        ${SRC_DIR}/expandpk.cu
        ${SRC_DIR}/sign.cu
        ${SRC_DIR}/verify.cu
)

function(add_mayo_test_all_versions base_target)
    cmake_parse_arguments(ARG "" "" "SRCS" ${ARGN})

    foreach(ver 1 2 3 5)
        set(target "${base_target}_mayo${ver}")

        add_executable(${target} ${ARG_SRCS} ${MAYO_SOURCES})
        target_include_directories(${target} PRIVATE ${INCLUDE_DIR})

        target_compile_definitions(${target} PRIVATE MAYO_VERSION=${ver})

        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
        )

        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endforeach()
endfunction()

if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/bench_stream.cu)
    add_mayo_test_all_versions(bench_stream SRCS ${CMAKE_CURRENT_LIST_DIR}/bench_stream.cu)
endif ()
