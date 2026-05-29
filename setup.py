from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "tank_sim",          
        ["tanks.cpp"],       
        extra_compile_args=["/O2", "/fp:fast"], 
    ),
]

setup(
    name="tank_sim",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)