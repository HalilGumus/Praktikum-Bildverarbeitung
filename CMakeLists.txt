cmake_minimum_required(VERSION 3.16)
project(P1)

if(UNIX)
	set(royale_DIR /opt/libroyale-4.10.0.162-LINUX-x86-64Bit/share)
endif()

find_package(OpenCV REQUIRED)
find_package(royale REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(${PROJECT_NAME} PRIVATE ${royale_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${royale_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${royale_LIBS})