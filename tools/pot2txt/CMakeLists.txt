cmake_minimum_required(VERSION 3.10)

project(pot2txt)

set(CMAKE_CXX_STANDARD 11)
# 开启调试信息
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

SET(PROJECT_SRC
        ./pot2txt.c
        )

add_executable(${PROJECT_NAME} ${PROJECT_SRC})

