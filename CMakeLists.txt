cmake_minimum_required(VERSION 3.12...3.27)
project(Pybind)

# Najpierw znajdujemy Pythona w systemie
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Pytamy Pythona dynamicznie o ścieżkę do modułu CMake od pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Teraz CMake na 100% znajdzie pakiet, niezależnie od tego jak Windows namiesza w folderach
find_package(pybind11 REQUIRED)

pybind11_add_module(tank_sim tanks.cpp)

# Bezpieczne i wydajne przypisanie flag kompilacji dla MSVC (Windows) oraz GCC/Clang
if(MSVC)
    target_compile_options(tank_sim PRIVATE /O2 /fp:fast)
else()
    target_compile_options(tank_sim PRIVATE -O3 -ffast-math)
endif()