cmake_minimum_required(VERSION 2.8)
project(liblda)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-std=c++11 -Ofast -march=native -fopenmp")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -Ofast -march=native -fopenmp")

include_directories(${PROJECT_SOURCE_DIR}/src/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)

file(GLOB SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*/*.cpp")
file(GLOB INCLUDE_FILES "${PROJECT_SOURCE_DIR}/src/include/*/*.h")

set(SRCS ${SOURCE_FILES} ${INCLUDE_FILES})

add_executable(main ${SRCS})

add_library(liblda SHARED ${SRCS})