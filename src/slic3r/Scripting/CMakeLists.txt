# libbambu_api — the embedded pyslic3r Python runtime host.
#
# Kept as its own static target (additive-and-isolated invariant): the only
# core edits are the one init/shutdown hook pair in GUI_App.cpp and the
# wiring in the parent CMakeLists. Mutually dependent with libslic3r_gui
# (the bindings call Plater; GUI_App calls host_init) — a static-library
# cycle CMake resolves by repeating archives on the link line.

add_library(libbambu_api STATIC
    PyHost.hpp
    PyHost.cpp
    PyBindings.hpp
    PyObjectModel.cpp
    PyDeviceHost.cpp
    M0SelfTest.cpp
)

# Orca open device plane: bound over the PrintHost HTTP layer
# (PyDeviceHost.cpp), not Bambu MQTT. The Bambu PyDevice.cpp is carried
# but not compiled here.

# Same include roots as libslic3r_gui: src/ is the include root
# ("slic3r/GUI/GUI_App.hpp", "libslic3r/Model.hpp"). Utils/ is on the path
# because GUI headers use "../GUI/Foo.hpp"-style relative includes that only
# resolve through it (libslic3r_gui does the same).
target_include_directories(libbambu_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_include_directories(libbambu_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(libbambu_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../Utils)

target_compile_definitions(libbambu_api PRIVATE BBS_PY_RUNTIME=1)

# Plain (non-keyword) signature to stay compatible with the parent scope's
# usage on libslic3r_gui; wxWidgets flags/includes arrive via the directory-
# scope include(${wxWidgets_USE_FILE}) in src/CMakeLists.txt. Deliberately
# does NOT list ${wxWidgets_LIBRARIES}: repeating the wx archives here
# perturbs the final static link order (wx webrequest ends up after the last
# libcurl occurrence -> undefined curl_version_info); wx symbols resolve via
# the libslic3r_gui cycle instead.
target_link_libraries(libbambu_api libslic3r cereal imgui pybind11::embed)

encoding_check(libbambu_api)
