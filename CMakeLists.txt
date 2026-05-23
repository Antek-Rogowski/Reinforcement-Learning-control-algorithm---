cmake_minimum_required(VERSION 3.12...3.27)
project(TankSimulatorRL)

# Szukamy pakietu pybind11. 
# Zadbaj, by mieć go zainstalowanego w swoim środowisku: pip install pybind11
find_package(pybind11 REQUIRED)

# Tworzymy moduł rozszerzenia Pythona.
# "tank_sim" to nazwa modułu, która MUSI zgadzać się z tą z makra PYBIND11_MODULE w C++.
pybind11_add_module(tank_sim tanks.cpp)

# Opcjonalnie: flaga agresywnej optymalizacji, żeby pokazać pełną moc C++
target_compile_options(tank_sim PRIVATE -O3 -ffast-math)