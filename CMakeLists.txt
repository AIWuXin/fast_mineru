cmake_minimum_required(VERSION 3.21)
project(fast_mineru LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# RTX 40系列: 89, RTX 50系列: 120。改这里适配目标卡。
set(CMAKE_CUDA_ARCHITECTURES 89 120)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# ============================================
# 1. Python(Development 组件提供 Python.h) —— 优先项目内 .venv
# ============================================
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv")
    set(Python3_ROOT_DIR ${CMAKE_SOURCE_DIR}/.venv)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

# ============================================
# 2. pybind11(用 pip 安装的 cmake 配置)
# ============================================
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR_AUTO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(EXISTS "${pybind11_DIR_AUTO}")
    set(pybind11_DIR ${pybind11_DIR_AUTO})
endif()
find_package(pybind11 CONFIG REQUIRED)

# ============================================
# 3. CUDA
# ============================================
find_package(CUDAToolkit REQUIRED)

# ============================================
# 4. pybind11 模块(csrc CUDA kernel + binding)
# ============================================
file(GLOB_RECURSE FAST_MINERU_CU  ${CMAKE_SOURCE_DIR}/fast_mineru/csrc/kernels/*.cu)
pybind11_add_module(_fast_mineru_core SHARED
    ${CMAKE_SOURCE_DIR}/fast_mineru/csrc/binding.cpp
    ${FAST_MINERU_CU}
)
target_include_directories(_fast_mineru_core PRIVATE
    ${CMAKE_SOURCE_DIR}/fast_mineru/csrc/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(_fast_mineru_core PRIVATE
    pybind11::module
    CUDA::cudart
)
set_target_properties(_fast_mineru_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "_fast_mineru_core"
)

# ============================================
# 5. 后处理：把 .pyd + cudart dll 拷进包内(照抄 nemotron_infer)
# ============================================
set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/fast_mineru/csrc")
file(GLOB CUDA_RUNTIME_DLL "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
add_custom_command(TARGET _fast_mineru_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:_fast_mineru_core>" "${OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDA_RUNTIME_DLL} "${OUTPUT_DIR}"
    COMMENT "Copying _fast_mineru_core.pyd + CUDA runtime DLLs to ${OUTPUT_DIR}"
    VERBATIM
)
