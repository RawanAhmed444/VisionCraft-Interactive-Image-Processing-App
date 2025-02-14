cmake_minimum_required(VERSION 3.14)
project(Task1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure MSVC is used
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(STATUS "Using MSVC as the compiler")
endif()

# Set CMake Prefix Path for dependencies
set(CMAKE_PREFIX_PATH "C:/Qt/6.5.0/msvc2019_64" "C:/vcpkg/installed/x64-windows")
#======================= INCLUSION OF Qt =======================#
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

#======================= FIND OpenCV =======================#
set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4") # Adjust as needed
find_package(OpenCV REQUIRED)

#======================= FIND Protobuf =======================#
set(Protobuf_DIR "C:/vcpkg/installed/x64-windows/share/protobuf") # Adjust as needed
find_package(Protobuf REQUIRED)

#======================= FIND Vulkan =======================#


#=================== INCLUSION OF Project Files ====================#
set(UI_DIR "${CMAKE_SOURCE_DIR}/ui")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

include_directories(${UI_DIR} ${INCLUDE_DIR} ${SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} ${Vulkan_INCLUDE_DIR})

file(GLOB_RECURSE SOURCES
    "${UI_DIR}/*.ui"
    "${UI_DIR}/*.qrc"
    "${INCLUDE_DIR}/*.h"
    "${SOURCE_DIR}/*.cpp"
)

# Add the executable
add_executable(Task1 WIN32 ${SOURCES})

# Link Libraries
target_link_libraries(Task1
    PRIVATE Qt6::Widgets
    PRIVATE ${OpenCV_LIBS}
    PRIVATE ${Protobuf_LIBRARIES}
)
