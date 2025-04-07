cmake_minimum_required(VERSION 3.16)
project(TDOA_PROJECT)

# Указываем стандарт C++
set(CMAKE_CXX_STANDARD 17)

# Подключаем vcpkg
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Создаём исполняемый файл
add_executable(my_program TDOA_2.cpp)

# Связываем Eigen с проектом (если нужно)
target_link_libraries(my_program Eigen3::Eigen)