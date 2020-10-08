cmake_minimum_required(VERSION 3.8)

project(comp_graph)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH "D:\\Qt\\Qt5.12.9\\5.12.9\\msvc2017_64")

set(SOURCE_FILES main.cpp)

find_package( Qt5Core REQUIRED )
find_package( Qt5Widgets REQUIRED )
find_package( Qt5Gui REQUIRED )
find_package( Qt5OpenGL REQUIRED )

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

include_directories(${Qt5Widgets_INCLUDES})
include_directories(${Qt5Core_INCLUDES})
include_directories(${Qt5Gui_INCLUDES})
include_directories(${Qt5OpenGL_INCLUDES})

add_definitions(${Qt5Widgets_DEFINITIONS})
add_definitions(${Qt5Core_DEFINITIONS})
add_definitions(${Qt5Gui_DEFINITIONS})
add_definitions(${Qt5OpenGL_DEFINITIONS})

add_executable(comp_graph ${SOURCE_FILES})
target_link_libraries(comp_graph opengl32 glu32 Qt5::Widgets Qt5::Core Qt5::Gui Qt5::OpenGL)
