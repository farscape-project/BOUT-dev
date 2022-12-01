cmake_minimum_required(VERSION 3.13)

project(hasegawa-wakatani LANGUAGES CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Development NumPy)
#find_package(PythonLibs 3.8 REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})

if (NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()

bout_add_example(hasegawa-wakatani SOURCES hw.cxx)

target_link_libraries(hasegawa-wakatani Python::Python Python::NumPy util)

