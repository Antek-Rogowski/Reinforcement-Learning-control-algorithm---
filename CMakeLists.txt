cmake_minimum_required(VERSION 3.12...3.27)
project(TankSimulatorRL)

set(pybind11_DIR "C:/Users/anton/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/pybind11/share/cmake/pybind11")

find_package(pybind11 REQUIRED)

pybind11_add_module(tank_sim tanks.cpp)

# Bezpieczne i wydajne przypisanie flag kompilacji dla MSVC (Windows) oraz GCC/Clang
if(MSVC)
    target_compile_options(tank_sim PRIVATE /O2 /fp:fast)
else()
    target_compile_options(tank_sim PRIVATE -O3 -ffast-math)
endif()