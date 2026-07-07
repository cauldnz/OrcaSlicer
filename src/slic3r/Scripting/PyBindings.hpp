#ifndef slic3r_Scripting_PyBindings_hpp_
#define slic3r_Scripting_PyBindings_hpp_

#include <pybind11/pybind11.h>

// The pyslic3r embedded module is registered once (PYBIND11_EMBEDDED_MODULE in
// PyHost.cpp). Its read-only object-model surface (M1) is defined separately in
// PyObjectModel.cpp and attached here, so the module macro stays a thin
// composition point and the binding surface can grow per-milestone.
namespace pyslic3r {
void register_object_model(pybind11::module_ &m);
// M4 cloud device plane — read-only (login state, bound-printer enumeration,
// status). Kept in its own TU so the device layer stays strictly isolated
// (a cloud outage must degrade printing only, never model/slice work).
void register_device(pybind11::module_ &m);
} // namespace pyslic3r

#endif // slic3r_Scripting_PyBindings_hpp_
