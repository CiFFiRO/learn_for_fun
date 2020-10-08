cmake_minimum_required(VERSION 3.8)

project(comp_graph)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
SET (GLEW_VERSION    2.1.0)
SET (GLEW_DIRECTORY  ${EXTERNAL_LIBS}/glew/${GLEW_VERSION})
SET (GLEW_INCLUDES   ${GLEW_DIRECTORY}/include)
SET (GLEW_LIBS       ${GLEW_DIRECTORY}/win/${SYSTEM_ARC}/lib)

set(SOURCE_FILES main.cpp)

include_directories(GL)
include_directories(GLFW)


add_executable(comp_graph ${SOURCE_FILES})
target_link_libraries(comp_graph opengl32 glu32 glfw3 D:\\workspace\\dss\\comp_graph\\glew32.lib)
