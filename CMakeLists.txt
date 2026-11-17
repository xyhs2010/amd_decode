cmake_minimum_required(VERSION 3.24)
project(amd_decode_ops LANGUAGES CXX HIP)

set(CMAKE_HIP_ARCHITECTURES gfx906)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ops MODULE src/amd_decode/with_hip/hip/int4_gemv.hip)
install(TARGETS _ops LIBRARY DESTINATION amd_decode/with_hip)
