cmake_minimum_required(VERSION 3.1)
project(async_sparse)

find_package(pybind11 REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)

include_directories(include include/eigen)

add_library(libasync_sparse STATIC src/conv2d.cpp)

pybind11_add_module(async_sparse src/bindings.cpp)

target_link_libraries(async_sparse PRIVATE libasync_sparse pybind11::embed)
