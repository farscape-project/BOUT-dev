cmake_minimum_required(VERSION 3.13)

project(hw3d LANGUAGES CXX C)

find_package(Python 3.8 REQUIRED COMPONENTS Development NumPy)
#find_package(PythonLibs 3.8 REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})

if (NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()

bout_add_example(hw3d SOURCES hw.cxx)

if(BOUT_HAS_CUDA)
  set_source_files_properties(hw.cxx PROPERTIES LANGUAGE CUDA)
endif()

target_link_libraries(hw3d Python::Python Python::NumPy util)
