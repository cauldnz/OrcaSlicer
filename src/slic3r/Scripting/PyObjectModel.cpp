// pyslic3r read-only object model (M1).
//
// Exposes Application / Document / Model tree / Config / Plates as an
// inspection-only surface. Invariants:
//   - Read-only: every member is a getter; nothing here mutates the document,
//     config, or presets.
//   - UI-parity: everything routes through Plater and the objects it owns
//     (Model, PartPlateList, DynamicPrintConfig, PresetBundle) — never a
//     parallel data path.
//   - Main-thread-guarded: every accessor asserts the wx main thread, so
//     off-thread callers must come through run_on_main_blocking().
//   - Handles are indices, re-resolved per call and bounds-checked, so a stale
//     Python handle raises rather than dereferencing a freed pointer if the
//     model changed between calls.

#include "PyBindings.hpp"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <wx/app.h>
#include <wx/thread.h>

#include "libslic3r/libslic3r.h"
#include "libslic3r/Model.hpp"
#include "libslic3r/Shape/TextShape.hpp"   // load_text_shape / TextResult (editable text)
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/Config.hpp"
#include "libslic3r/Preset.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/Print.hpp"                 // Print, PrintStatistics
#include "libslic3r/GCode/GCodeProcessor.hpp"  // GCodeProcessorResult
#include "libslic3r/Format/bbs_3mf.hpp"        // LoadStrategy
#include "libslic3r/CustomGCode.hpp"      // colour-change-by-height
#include "slic3r/GUI/GUI_App.hpp"
#include <pybind11/eval.h>                  // py::exec (download helper)
#include "slic3r/GUI/Plater.hpp"
#include "libslic3r/CutUtils.hpp"   // Cut class + ModelObjectCutAttribute (reworked cut)
#include "slic3r/GUI/GLCanvas3D.hpp"   // canvas3D()->get_selection()
#include "slic3r/GUI/Selection.hpp"     // Selection::add_object (headless select)
#include "slic3r/GUI/PartPlate.hpp"
#include "slic3r/GUI/Tab.hpp"
#include "slic3r/GUI/BackgroundSlicingProcess.hpp"   // job.cancel() -> stop()

#include <boost/filesystem/path.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>   // std::sort
#include <map>

#include <wx/utils.h>   // wxMilliSleep

namespace py = pybind11;
using namespace Slic3r;

namespace pyslic3r {
namespace {

// ---- guards + resolvers ---------------------------------------------------

void main_thread(const char *what)
{
    if (!wxThread::IsMain())
        throw std::runtime_error(std::string(what) +
            " must be accessed on the wx main thread; use the marshalling primitive");
}

GUI::Plater *plater_or_throw(const char *what)
{
    main_thread(what);
    auto *p = GUI::wxGetApp().plater();
    if (p == nullptr)
        throw std::runtime_error("no active document");
    return p;
}

Model &model_or_throw(const char *what) { return plater_or_throw(what)->model(); }

ModelObject *object_at(size_t idx, const char *what)
{
    Model &m = model_or_throw(what);
    if (idx >= m.objects.size())
        throw std::runtime_error("object index out of range (model changed?)");
    return m.objects[idx];
}

// ---- small value converters ----------------------------------------------

py::tuple vec3(const Vec3d &v) { return py::make_tuple(v.x(), v.y(), v.z()); }

const char *volume_type_str(ModelVolumeType t)
{
    switch (t) {
    case ModelVolumeType::MODEL_PART:         return "model_part";
    case ModelVolumeType::NEGATIVE_VOLUME:    return "negative_volume";
    case ModelVolumeType::PARAMETER_MODIFIER: return "modifier";
    case ModelVolumeType::SUPPORT_BLOCKER:    return "support_blocker";
    case ModelVolumeType::SUPPORT_ENFORCER:   return "support_enforcer";
    default:                                  return "invalid";
    }
}

// ---- handle types ---------------------------------------------------------
// Each holds indices only; the referenced C++ object is re-resolved per call.

struct PyApp {};
struct PyDocument {};
struct PyModel {};
struct PyObject   { size_t idx; };
struct PyVolume   { size_t obj_idx; size_t vol_idx; };
struct PyText     { size_t obj_idx; size_t vol_idx; };
struct PyPlateList {};
struct PyPlate    { int idx; };

// Which config a PyConfig fronts.
enum class ConfigSource { Global, Print, Filament, Printer, Plate, Object, Volume };
struct PyConfig { ConfigSource source; int plate_idx = 0; int vol_idx = 0; };

// M3 slicing handles.
struct PySliceJob { int plate_idx; };
struct PySliceResult
{
    bool                    success = false;
    long                    print_time_s = 0;
    long                    layer_count = 0;
    double                  total_weight = 0;               // grams (sum of filament_g)
    double                  total_cost = 0;                 // currency units
    double                  total_wipe_tower_filament = 0;  // mm
    int                     total_toolchanges = 0;
    std::map<int, double>   filament_g;    // per extruder/slot
    std::map<int, double>   filament_mm;   // per extruder/slot (from volume)
    std::string             gcode_path;
    std::string             error;         // reason when !success
};

// The PresetCollection backing a preset source, or nullptr for Global/Plate.
PresetCollection *preset_collection(ConfigSource s)
{
    auto *pb = GUI::wxGetApp().preset_bundle;
    switch (s) {
    case ConfigSource::Print:    return &pb->prints;
    case ConfigSource::Filament: return &pb->filaments;
    case ConfigSource::Printer:  return &pb->printers;
    default:                     return nullptr;
    }
}

Preset::Type preset_type(ConfigSource s)
{
    switch (s) {
    case ConfigSource::Print:    return Preset::TYPE_PRINT;
    case ConfigSource::Filament: return Preset::TYPE_FILAMENT;
    case ConfigSource::Printer:  return Preset::TYPE_PRINTER;
    default:                     return Preset::TYPE_INVALID;
    }
}

GUI::PartPlate *plate_or_throw(int plate_idx, const char *what)
{
    auto &list = plater_or_throw(what)->get_partplate_list();
    if (plate_idx < 0 || plate_idx >= list.get_plate_count())
        throw std::runtime_error("plate index out of range");
    GUI::PartPlate *plate = list.get_plate(plate_idx);
    if (plate == nullptr) throw std::runtime_error("plate gone");
    return plate;
}

// Read view. For preset sources this is the *edited* preset — the working copy
// the GUI reads and writes — so a value set via Config.set reads straight back.
const ConfigBase *resolve_config(const PyConfig &c, const char *what)
{
    switch (c.source) {
    case ConfigSource::Global: {
        const DynamicPrintConfig *cfg = plater_or_throw(what)->config();
        if (cfg == nullptr) throw std::runtime_error("no global config");
        return cfg;
    }
    case ConfigSource::Print:
    case ConfigSource::Filament:
    case ConfigSource::Printer: {
        main_thread(what);
        return &preset_collection(c.source)->get_edited_preset().config;
    }
    case ConfigSource::Plate: {
        GUI::PartPlate *plate = plate_or_throw(c.plate_idx, what);
        if (plate->config() == nullptr) throw std::runtime_error("no plate config");
        return plate->config();
    }
    case ConfigSource::Volume: {
        // per-volume overrides (ModelVolume::config); plate_idx=object, vol_idx=volume.
        ModelObject *obj = object_at(c.plate_idx, what);
        if (size_t(c.vol_idx) >= obj->volumes.size())
            throw std::runtime_error("volume index out of range");
        return &obj->volumes[c.vol_idx]->config.get();
    }
    case ConfigSource::Object: {
        // per-object overrides (ModelObject::config); plate_idx = object index.
        return &object_at(c.plate_idx, what)->config.get();
    }
    }
    throw std::runtime_error("unknown config source");
}

ModelVolume *volume_at(const PyVolume &v, const char *what)
{
    ModelObject *obj = object_at(v.obj_idx, what);
    if (v.vol_idx >= obj->volumes.size())
        throw std::runtime_error("volume index out of range (model changed?)");
    return obj->volumes[v.vol_idx];
}

// Resolve a text handle -> ModelVolume, asserting it is editable text.
ModelVolume *text_at(const PyText &t, const char *what)
{
    ModelVolume *v = volume_at(PyVolume{t.obj_idx, t.vol_idx}, what);
    if (!v->is_text())
        throw std::runtime_error(std::string(what) + ": volume is not editable text");
    return v;
}

// Map Orca text_configuration -> load_text_shape() params. Depth (thickness)
// is taken from the current mesh Z extent (Orca stores no depth on the style).
static void _orca_text_params(ModelVolume *v, std::string &text, std::string &font,
                              float &size, float &depth, bool &bold, bool &italic)
{
    const TextConfiguration &tc = *v->text_configuration;
    const FontProp &fp = tc.style.prop;
    text = tc.text;
    font = fp.face_name.has_value() ? *fp.face_name : tc.style.name;   // actual face, not style name
    if (font.empty()) font = "NotoSans";               // OCCT fallback
    size = fp.size_in_mm;
    depth = (float) v->mesh().bounding_box().size().z();
    if (!(depth > 0.f)) depth = 1.f;
    bold = (fp.weight.has_value() && fp.weight->find("bold") != std::string::npos)
        || (fp.boldness.has_value() && *fp.boldness > 0.f);
    italic = (fp.style.has_value() && fp.style->find("italic") != std::string::npos)
        || (fp.skew.has_value() && (*fp.skew < -1e-3f || *fp.skew > 1e-3f));
}

// Read slice stats off a sliced plate into a PySliceResult. Main thread.
void fill_slice_result(int plate_idx, PySliceResult &res)
{
    auto &list = plater_or_throw("SliceResult")->get_partplate_list();
    GUI::PartPlate *plate = list.get_plate(plate_idx);
    if (plate == nullptr) return;

    res.gcode_path = plate->get_tmp_gcode_path();

    const GCodeProcessorResult *gr = plate->get_slice_result();
    if (gr != nullptr) {
        const auto &st   = gr->print_statistics;
        const auto &mode = st.modes[0];   // 0 = Normal
        res.print_time_s = long(mode.time + 0.5f);
        { long _lc = 0; for (const auto &mv : gr->moves) if (long(mv.layer_id) > _lc) _lc = long(mv.layer_id);
          res.layer_count = gr->moves.empty() ? 0L : _lc + 1; }   // Mode has no layers_times

        // total_volumes_per_extruder is mm^3 of extruded filament per extruder.
        // length mm  = volume / filament cross-section (Ø1.75 mm default).
        // weight g   = volume(cm^3) * density(g/cm^3), density per extruder.
        const double area = M_PI * (1.75 / 2.0) * (1.75 / 2.0);   // mm^2
        for (const auto &kv : st.total_volumes_per_extruder) {
            const int    e       = int(kv.first);
            const double vol_mm3 = kv.second;
            res.filament_mm[e] = area > 0 ? vol_mm3 / area : 0.0;
            const double density = (e >= 0 && e < int(gr->filament_densities.size()))
                                       ? double(gr->filament_densities[e]) : 1.24;  // PLA fallback
            res.filament_g[e]  = vol_mm3 / 1000.0 * density;      // mm^3 -> cm^3 * g/cm^3
        }
        if (Print *_pr = plate->fff_print()) {
            const PrintStatistics &_ps = _pr->print_statistics();
            res.total_weight              = _ps.total_weight;
            res.total_cost                = _ps.total_cost;
            res.total_toolchanges         = _ps.total_toolchanges;
            res.total_wipe_tower_filament = _ps.total_wipe_tower_filament;
        }
        if (res.total_weight <= 0.0)   // fallback: sum per-extruder filament_g
            for (const auto &_kv : res.filament_g) res.total_weight += _kv.second;
    }
}

// Point the GUI selection at a single object (headless) — the seam the
// selection-based Plater ops (fill_bed, split) bind to. Mirrors a click.
static void _select_only(size_t obj_idx, const char *what)
{
    (void) object_at(obj_idx, what);   // bounds-check
    GUI::Selection &sel = plater_or_throw(what)->canvas3D()->get_selection();
    sel.clear();
    sel.add_object((unsigned int) obj_idx, true);
}

} // anonymous namespace

// ---------------------------------------------------------------------------

void register_object_model(py::module_ &m)
{
    // ---- Config -----------------------------------------------------------
    py::class_<PyConfig>(m, "Config")
        .def("has", [](const PyConfig &c, const std::string &key) {
            return resolve_config(c, "Config.has")->has(key);
        })
        .def("get", [](const PyConfig &c, const std::string &key) -> py::object {
            const ConfigBase *cfg = resolve_config(c, "Config.get");
            if (!cfg->has(key))
                return py::none();
            return py::str(cfg->opt_serialize(key));   // serialized string form
        })
        .def("keys", [](const PyConfig &c) {
            return resolve_config(c, "Config.keys")->keys();   // -> list[str]
        })
        .def_property_readonly("is_dirty", [](const PyConfig &c) {
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr) return false;   // global/plate: not preset-dirty-tracked
            main_thread("Config.is_dirty");
            return col->current_is_dirty();
        })
        // ---- M2 mutation --------------------------------------------------
        .def("set", [](const PyConfig &c, const std::string &key, const std::string &value) {
            // GUI-parity: edit the working config the app watches, mark dirty,
            // and let Plater invalidate slicing exactly as an on-screen edit.
            if (c.source == ConfigSource::Global)
                throw std::runtime_error(
                    "the global config is derived and read-only; set on "
                    "print_config / filament_config / printer_config, or a plate config");
            auto *plater = plater_or_throw("Config.set");
            if (c.source == ConfigSource::Plate) {
                GUI::PartPlate *plate = plate_or_throw(c.plate_idx, "Config.set");
                DynamicPrintConfig *cfg = plate->config();
                if (cfg == nullptr) throw std::runtime_error("no plate config");
                cfg->set_deserialize_strict(key, value);
                plater->schedule_background_process();
                return;
            }
            if (c.source == ConfigSource::Volume) {
                ModelObject *obj = object_at(c.plate_idx, "Config.set");
                if (size_t(c.vol_idx) >= obj->volumes.size())
                    throw std::runtime_error("volume index out of range");
                ConfigSubstitutionContext ctx(ForwardCompatibilitySubstitutionRule::EnableSilent);
                obj->volumes[c.vol_idx]->config.set_deserialize(key, value, ctx);
                plater->changed_object(int(c.plate_idx));
                return;
            }
            if (c.source == ConfigSource::Object) {
                ModelObject *obj = object_at(c.plate_idx, "Config.set");
                ConfigSubstitutionContext ctx(ForwardCompatibilitySubstitutionRule::EnableSilent);
                obj->config.set_deserialize(key, value, ctx);
                plater->changed_object(int(c.plate_idx));
                return;
            }
            PresetCollection *col = preset_collection(c.source);
            DynamicPrintConfig &cfg = col->get_edited_preset().config;
            if (!cfg.has(key))
                throw std::runtime_error("unknown config key for this preset: " + key);
            cfg.set_deserialize_strict(key, value);
            col->update_dirty();                 // mark preset dirty like the GUI
            plater->on_config_change(cfg);        // diff + schedule reslice
        }, py::arg("key"), py::arg("value"))
        .def("presets", [](const PyConfig &c) {
            main_thread("Config.presets");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr)
                throw std::runtime_error("presets() only on print/filament/printer config");
            py::list out;
            for (const Preset &p : col->get_presets())
                if (p.is_visible && p.is_compatible) out.append(p.name);   // selectable set
            return out;
        })
        .def_property_readonly("selected_preset", [](const PyConfig &c) {
            main_thread("Config.selected_preset");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr)
                throw std::runtime_error("selected_preset only on print/filament/printer config");
            return col->get_selected_preset_name();
        })
        .def("apply_preset", [](const PyConfig &c, const std::string &name, bool force) {
            // Select a named preset via the Tab — the same path the preset
            // dropdown uses (compatibility checks, dependent tabs, dirty, and
            // the Plater cascade). force=True force-selects, which also allows
            // selecting a system preset that isn't yet "installed"/visible —
            // the end state of installing it through the wizard.
            if (preset_type(c.source) == Preset::TYPE_INVALID)
                throw std::runtime_error("apply_preset only on print/filament/printer config");
            main_thread("Config.apply_preset");
            // Headless-safe: select via the PresetBundle directly. The GUI Tab
            // path (Tab::select_preset) SIGSEGVs offscreen; this mirrors Orca's
            // own undo/redo restore (PresetCollection::select_preset_by_name),
            // then resolves compatible print/filament and pushes to the Plater.
            auto *plater = plater_or_throw("Config.apply_preset");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr) throw std::runtime_error("no preset collection for this config");
            col->select_preset_by_name(name, force);
            PresetBundle *pb = GUI::wxGetApp().preset_bundle;
            pb->update_compatible(PresetSelectCompatibleType::Always);
            col->update_dirty();
            plater->on_config_change(pb->full_config());
            // Verify the END STATE, not select_preset_by_name's return: it returns
            // false when the preset is already selected, and update_compatible reverts
            // an incompatible one — so the real test is whether the target is selected.
            if (!force && col->get_selected_preset_name() != name)
                throw std::runtime_error("preset not applied (not found or incompatible): " + name);
        }, py::arg("name"), py::arg("force") = false);

    // ---- Volume -----------------------------------------------------------
    py::class_<PyVolume>(m, "Volume")
        .def_property_readonly("name", [](const PyVolume &v) {
            return volume_at(v, "Volume.name")->name;
        })
        .def_property_readonly("type", [](const PyVolume &v) {
            return std::string(volume_type_str(volume_at(v, "Volume.type")->type()));
        })
        .def_property_readonly("is_model_part", [](const PyVolume &v) {
            return volume_at(v, "Volume.is_model_part")->is_model_part();
        })
        .def_property_readonly("config", [](const PyVolume &v) {
            (void) volume_at(v, "Volume.config");   // bounds-check
            return PyConfig{ConfigSource::Volume, int(v.obj_idx), int(v.vol_idx)};
        });

    // ---- Text (editable emboss text; Orca text_configuration + load_text_shape) ----
    py::class_<PyText>(m, "Text")
        .def_property_readonly("text", [](const PyText &t) {
            return text_at(t, "Text.text")->text_configuration->text;
        })
        .def_property_readonly("font_name", [](const PyText &t) {
            { const EmbossStyle &st = text_at(t, "Text.font_name")->text_configuration->style; return st.prop.face_name.has_value() ? *st.prop.face_name : st.name; }
        })
        .def_property_readonly("font_size", [](const PyText &t) {
            return text_at(t, "Text.font_size")->text_configuration->style.prop.size_in_mm;
        })
        .def_property_readonly("width", [](const PyText &t) {
            ModelVolume *v = text_at(t, "Text.width");
            std::string text, font; float size, depth; bool bold, italic;
            _orca_text_params(v, text, font, size, depth, bold, italic);
            TextResult r;
            load_text_shape(text.c_str(), font.c_str(), size, depth, bold, italic, r);
            return r.text_width;
        })
        .def("set_text", [](const PyText &t, const std::string &new_text,
                            bool fit, py::object max_width) -> py::object {
            main_thread("Text.set_text");
            ModelObject *obj = object_at(t.obj_idx, "Text.set_text");
            ModelVolume *vol = text_at(t, "Text.set_text");
            std::string text, font; float size, depth; bool bold, italic;
            _orca_text_params(vol, text, font, size, depth, bold, italic);

            double target_w;
            if (!max_width.is_none()) {
                target_w = max_width.cast<double>();
            } else {
                TextResult orig;
                load_text_shape(text.c_str(), font.c_str(), size, depth, bold, italic, orig);
                target_w = orig.text_width;
            }
            TextResult r;
            load_text_shape(new_text.c_str(), font.c_str(), size, depth, bold, italic, r);
            bool fitted = false;
            if (fit && r.text_width > target_w && r.text_width > 0.0) {
                size = (float)(size * (target_w / r.text_width));
                load_text_shape(new_text.c_str(), font.c_str(), size, depth, bold, italic, r);
                fitted = true;
            }
            if (r.text_mesh.empty())
                throw std::runtime_error("Text.set_text: mesh generation failed (font missing?)");

            TextConfiguration tc = *vol->text_configuration;   // copy + update
            tc.text = new_text;
            tc.style.prop.size_in_mm = size;

            GUI::wxGetApp().plater()->take_snapshot("Edit Text");
            Geometry::Transformation tran = vol->get_transformation();
            ModelVolume *nv = obj->add_volume(r.text_mesh, false);
            nv->calculate_convex_hull();
            nv->set_transformation(tran.get_matrix());
            nv->text_configuration = tc;
            nv->name = vol->name;
            nv->set_type(vol->type());
            nv->config.apply(vol->config);
            std::swap(obj->volumes[t.vol_idx], obj->volumes.back());
            obj->delete_volume(obj->volumes.size() - 1);
            obj->invalidate_bounding_box();
            GUI::wxGetApp().plater()->changed_object(int(t.obj_idx));

            py::dict d;
            d["text"] = new_text; d["font_size"] = size; d["width"] = r.text_width;
            d["fit_target"] = target_w; d["fitted"] = fitted;
            return d;
        }, py::arg("text"), py::arg("fit") = true, py::arg("max_width") = py::none());

    // ---- Object -----------------------------------------------------------
    py::class_<PyObject>(m, "Object")
        .def_property_readonly("name", [](const PyObject &o) {
            return object_at(o.idx, "Object.name")->name;
        })
        .def_property_readonly("index", [](const PyObject &o) { return o.idx; })
        .def_property_readonly("instance_count", [](const PyObject &o) {
            return object_at(o.idx, "Object.instance_count")->instances.size();
        })
        .def_property_readonly("volumes", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.volumes");
            py::list out;
            for (size_t i = 0; i < obj->volumes.size(); ++i)
                out.append(PyVolume{o.idx, i});
            return out;
        })
        .def("texts", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.texts");
            py::list out;
            for (size_t i = 0; i < obj->volumes.size(); ++i)
                if (obj->volumes[i]->is_text())
                    out.append(PyText{o.idx, i});
            return out;
        })
        .def_property_readonly("config", [](const PyObject &o) {
            (void) object_at(o.idx, "Object.config");   // bounds-check
            return PyConfig{ConfigSource::Object, int(o.idx)};
        })
        .def("bounding_box", [](const PyObject &o) {
            const BoundingBoxf3 &bb = object_at(o.idx, "Object.bounding_box")->bounding_box_approx();
            py::dict d;
            d["min"]    = vec3(bb.min);
            d["max"]    = vec3(bb.max);
            d["size"]   = vec3(bb.size());
            d["center"] = vec3(bb.center());
            return d;
        })
        // ---- M2 mutation (UI-parity: snapshot + Plater refresh) -----------
        .def("translate", [](const PyObject &o, double dx, double dy, double dz) {
            auto *plater = plater_or_throw("Object.translate");
            ModelObject *obj = object_at(o.idx, "Object.translate");
            GUI::Plater::TakeSnapshot snap(plater, "API: move object");
            obj->translate_instances(Vec3d(dx, dy, dz));  // moves the placed instances
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));           // same refresh the GUI uses
        }, py::arg("dx"), py::arg("dy"), py::arg("dz"))
        .def("delete", [](const PyObject &o) {
            auto *plater = plater_or_throw("Object.delete");
            (void) object_at(o.idx, "Object.delete");     // bounds-check
            plater->delete_object_from_model(o.idx);      // snapshots internally
        })
        // ---- transforms (UI-parity: the rotate / scale / mirror gizmos and
        //      the "set number of instances" action) --------------------------
        .def("rotate", [](const PyObject &o, double rx, double ry, double rz) {
            auto *plater = plater_or_throw("Object.rotate");
            ModelObject *obj = object_at(o.idx, "Object.rotate");
            GUI::Plater::TakeSnapshot snap(plater, "API: rotate object");
            if (rx != 0.0) obj->rotate(rx * M_PI / 180.0, X);
            if (ry != 0.0) obj->rotate(ry * M_PI / 180.0, Y);
            if (rz != 0.0) obj->rotate(rz * M_PI / 180.0, Z);
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("rx") = 0.0, py::arg("ry") = 0.0, py::arg("rz") = 0.0)
        .def("scale", [](const PyObject &o, double x, py::object y, py::object z) {
            const double sx = x;
            const double sy = y.is_none() ? x : y.cast<double>();
            const double sz = z.is_none() ? x : z.cast<double>();
            if (sx <= 0.0 || sy <= 0.0 || sz <= 0.0)
                throw std::runtime_error("scale factors must be > 0");
            auto *plater = plater_or_throw("Object.scale");
            ModelObject *obj = object_at(o.idx, "Object.scale");
            GUI::Plater::TakeSnapshot snap(plater, "API: scale object");
            obj->scale(Vec3d(sx, sy, sz));
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("x"), py::arg("y") = py::none(), py::arg("z") = py::none())
        .def("mirror", [](const PyObject &o, const std::string &axis) {
            Axis a;
            if      (axis == "x" || axis == "X") a = X;
            else if (axis == "y" || axis == "Y") a = Y;
            else if (axis == "z" || axis == "Z") a = Z;
            else throw std::runtime_error("axis must be 'x', 'y', or 'z'");
            auto *plater = plater_or_throw("Object.mirror");
            ModelObject *obj = object_at(o.idx, "Object.mirror");
            GUI::Plater::TakeSnapshot snap(plater, "API: mirror object");
            obj->mirror(a);
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("axis"))
        .def("set_instances", [](const PyObject &o, int n) {
            if (n < 1) throw std::runtime_error("instance count must be >= 1");
            auto *plater = plater_or_throw("Object.set_instances");
            ModelObject *obj = object_at(o.idx, "Object.set_instances");
            GUI::Plater::TakeSnapshot snap(plater, "API: set instances");
            while (int(obj->instances.size()) < n) {
                ModelInstance *ni = obj->add_instance(*obj->instances.front());
                ni->set_offset(ni->get_offset() + Vec3d(5.0, 5.0, 0.0));  // arrange() separates
            }
            while (int(obj->instances.size()) > n)
                obj->delete_last_instance();
            plater->changed_object(int(o.idx));
        }, py::arg("n"))
        .def("transform", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.transform");
            const BoundingBoxf3 bb = obj->bounding_box_approx();
            py::dict d;
            d["size"]   = vec3(bb.size());
            d["center"] = vec3(bb.center());
            if (!obj->instances.empty())
                d["position"] = vec3(obj->instances.front()->get_offset());
            d["instances"] = obj->instances.size();
            return d;
        })
        // ---- geometry finish (UI-parity: drop-to-bed, scale-to-fit, rename) --
        .def("split", [](const PyObject &o) {
            main_thread("Object.split");
            _select_only(o.idx, "Object.split");
            plater_or_throw("Object.split")->split_object();
            return model_or_throw("Object.split").objects.size();
        })
        .def("cut", [](const PyObject &o, double z, bool keep_upper, bool keep_lower) {
            main_thread("Object.cut");
            if (!keep_upper && !keep_lower)
                throw std::runtime_error("cut: keep_upper and keep_lower are both false");
            auto *plater = plater_or_throw("Object.cut");
            ModelObject *obj = object_at(o.idx, "Object.cut");
            if (obj->instances.empty())
                throw std::runtime_error("cut: object has no instances");
            const Vec3d off = obj->instances[0]->get_offset();
            ModelObjectCutAttributes attrs =
                only_if(keep_upper, ModelObjectCutAttribute::KeepUpper) |
                only_if(keep_lower, ModelObjectCutAttribute::KeepLower);
            Transform3d cut_matrix = Geometry::translation_transform(Vec3d(0.0, 0.0, z - off.z()));
            Slic3r::Cut cut(obj, 0, cut_matrix, attrs);
            const ModelObjectPtrs &new_objects = cut.perform_with_plane();
            plater->apply_cut_object_to_model(o.idx, new_objects);
            return model_or_throw("Object.cut").objects.size();
        }, py::arg("z"), py::arg("keep_upper") = true, py::arg("keep_lower") = true)
        .def("place_on_bed", [](const PyObject &o) {
            auto *plater = plater_or_throw("Object.place_on_bed");
            ModelObject *obj = object_at(o.idx, "Object.place_on_bed");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: place on bed"));
            obj->ensure_on_bed();
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        })
        .def("scale_to_fit", [](const PyObject &o, double x, double y, double z) {
            if (x <= 0.0 || y <= 0.0 || z <= 0.0)
                throw std::runtime_error("fit size must be > 0");
            auto *plater = plater_or_throw("Object.scale_to_fit");
            ModelObject *obj = object_at(o.idx, "Object.scale_to_fit");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: scale to fit"));
            obj->scale_to_fit(Vec3d(x, y, z));
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("rename", [](const PyObject &o, const std::string &name) {
            auto *plater = plater_or_throw("Object.rename");
            ModelObject *obj = object_at(o.idx, "Object.rename");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: rename object"));
            obj->name = name;
            plater->changed_object(int(o.idx));
        }, py::arg("name"))
        // ---- paint-by-height: per-Z-band config overrides ------------------
        // UI-parity: mirrors the object-list "Height range Modifier". Writes
        // ModelObject::layer_config_ranges[{min_z,max_z}] and refreshes via
        // changed_object, exactly like the GUI. `overrides` maps config keys to
        // values (e.g. {"extruder": "2", "layer_height": "0.12"}). Re-slice after.
        .def("add_height_range", [](const PyObject &o, double min_z, double max_z,
                                    py::dict overrides) {
            if (!(max_z > min_z))
                throw std::runtime_error("max_z must be greater than min_z");
            auto *plater = plater_or_throw("Object.add_height_range");
            ModelObject *obj = object_at(o.idx, "Object.add_height_range");
            GUI::Plater::TakeSnapshot snap(plater, "API: add height range");

            const t_layer_height_range range{ min_z, max_z };
            ModelConfig &cfg = obj->layer_config_ranges[range];   // creates entry

            // Seed a well-formed range config the way the GUI does: a
            // layer_height (from the object/preset) and extruder=0 ("use object").
            double lh = 0.2;
            {
                const DynamicPrintConfig &oc = obj->config.get();
                auto *pb = GUI::wxGetApp().preset_bundle;
                if (oc.has("layer_height"))
                    lh = oc.opt_float("layer_height");
                else if (pb && pb->prints.get_edited_preset().config.has("layer_height"))
                    lh = pb->prints.get_edited_preset().config.opt_float("layer_height");
            }
            cfg.set_key_value("layer_height", new ConfigOptionFloat(lh));
            cfg.set_key_value("extruder",     new ConfigOptionInt(0));

            // Apply caller overrides (serialized-string path, like Config.set).
            for (auto kv : overrides) {
                const std::string key = kv.first.cast<std::string>();
                const std::string val = py::str(kv.second).cast<std::string>();
                ConfigSubstitutionContext ctx{ ForwardCompatibilitySubstitutionRule::Disable };
                cfg.set_deserialize(key, val, ctx);
            }
            plater->changed_object(int(o.idx));
            return py::make_tuple(min_z, max_z);
        }, py::arg("min_z"), py::arg("max_z"), py::arg("overrides") = py::dict())
        .def("clear_height_ranges", [](const PyObject &o) {
            auto *plater = plater_or_throw("Object.clear_height_ranges");
            ModelObject *obj = object_at(o.idx, "Object.clear_height_ranges");
            GUI::Plater::TakeSnapshot snap(plater, "API: clear height ranges");
            obj->layer_config_ranges.clear();
            plater->changed_object(int(o.idx));
        })
        .def("height_ranges", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.height_ranges");
            py::list out;
            for (const auto &kv : obj->layer_config_ranges) {
                py::dict d;
                d["min_z"] = kv.first.first;
                d["max_z"] = kv.first.second;
                py::dict ov;
                const DynamicPrintConfig &c = kv.second.get();
                for (const std::string &k : c.keys())
                    ov[py::str(k)] = c.opt_serialize(k);
                d["overrides"] = ov;
                out.append(d);
            }
            return out;
        });

    // ---- Model ------------------------------------------------------------
    py::class_<PyModel>(m, "Model")
        .def_property_readonly("object_count", [](const PyModel &) {
            return model_or_throw("Model.object_count").objects.size();
        })
        .def_property_readonly("objects", [](const PyModel &) {
            Model &mo = model_or_throw("Model.objects");
            py::list out;
            for (size_t i = 0; i < mo.objects.size(); ++i)
                out.append(PyObject{i});
            return out;
        })
        // ---- M2 mutation --------------------------------------------------
        .def("add", [](const PyModel &, const std::string &path) {
            // Import geometry as object(s) via the same Plater path as GUI
            // "Add" (LoadModel, no LoadConfig, Silence), under a snapshot.
            auto *plater = plater_or_throw("Model.add");
            const size_t before = plater->model().objects.size();
            GUI::Plater::TakeSnapshot snap(plater, "API: add model");
            std::vector<boost::filesystem::path> paths{ boost::filesystem::path(path) };
            const auto strategy = LoadStrategy::LoadModel |
                                  LoadStrategy::AddDefaultInstances |
                                  LoadStrategy::Silence;
            std::vector<size_t> idxs = plater->load_files(paths, strategy, /*ask_multi=*/false);
            const size_t after = plater->model().objects.size();
            if (after <= before)
                throw std::runtime_error("no object added from: " + path);
            return PyObject{ idxs.empty() ? after - 1 : idxs.back() };
        }, py::arg("path"))
        .def("remove", [](const PyModel &, const PyObject &o) {
            auto *plater = plater_or_throw("Model.remove");
            (void) object_at(o.idx, "Model.remove");       // bounds-check
            plater->delete_object_from_model(o.idx);        // snapshots internally
        }, py::arg("object"))
        // ---- clear the scene (UI-parity: Edit -> Delete All) ---------------
        .def("clear", [](const PyModel &) {
            auto *plater = plater_or_throw("Model.clear");
            plater->delete_all_objects_from_model();   // one-shot; snapshots internally
        });

    // ---- Plate / PlateList ------------------------------------------------
    py::class_<PyPlate>(m, "Plate")
        .def_property_readonly("index", [](const PyPlate &p) { return p.idx; })
        .def_property_readonly("object_count", [](const PyPlate &p) {
            auto &list = plater_or_throw("Plate.object_count")->get_partplate_list();
            GUI::PartPlate *plate = list.get_plate(p.idx);
            if (plate == nullptr) throw std::runtime_error("plate gone");
            return plate->get_objects_on_this_plate().size();
        })
        .def_property_readonly("is_sliceable", [](const PyPlate &p) {
            auto &list = plater_or_throw("Plate.is_sliceable")->get_partplate_list();
            GUI::PartPlate *plate = list.get_plate(p.idx);
            if (plate == nullptr) throw std::runtime_error("plate gone");
            return plate->can_slice();
        })
        .def_property_readonly("config", [](const PyPlate &p) {
            return PyConfig{ConfigSource::Plate, p.idx};
        })
        // ---- paint-by-height: colour changes (the multi-colour primitive) --
        // UI-parity: mirrors the preview vertical-slider "add colour change".
        // Writes Model::plates_custom_gcodes[plate] with ColorChange items
        // (extruder is 1-based = AMS slot), then invalidates the slice result
        // exactly as the slider handler does. Re-slice after (doc.slice()).
        // `changes`: list of dicts, each {"z": <print_z>, "extruder": <1-based>,
        //  optional "color": "#RRGGBB"} (colour auto-filled from the filament
        //  preset colour when omitted).
        .def("set_color_changes", [](const PyPlate &p, py::list changes) {
            auto *plater = plater_or_throw("Plate.set_color_changes");
            auto &list = plater->get_partplate_list();
            GUI::PartPlate *plate = list.get_plate(p.idx);
            if (plate == nullptr) throw std::runtime_error("plate gone");

            std::vector<std::string> ext_colors =
                plater->get_extruder_colors_from_plater_config();  // 0-based

            CustomGCode::Info info;
            info.mode = CustomGCode::Undef;
            for (auto handle : changes) {
                py::dict d = handle.cast<py::dict>();
                CustomGCode::Item item;
                item.type = CustomGCode::ToolChange;
                if (!d.contains("z"))
                    throw std::runtime_error("each colour change needs a 'z' (print_z)");
                item.print_z = d["z"].cast<double>();
                if (!d.contains("extruder"))
                    throw std::runtime_error("each colour change needs an 'extruder' (1-based)");
                item.extruder = d["extruder"].cast<int>();
                if (item.extruder < 1)
                    throw std::runtime_error("extruder is 1-based (AMS slot); must be >= 1");
                if (d.contains("color") && !d["color"].is_none()) {
                    item.color = d["color"].cast<std::string>();
                } else {
                    const int i0 = item.extruder - 1;
                    item.color = (i0 >= 0 && i0 < int(ext_colors.size()))
                                     ? ext_colors[i0] : std::string("#FFFFFF");
                }
                info.gcodes.push_back(std::move(item));
            }
            std::sort(info.gcodes.begin(), info.gcodes.end());
            CustomGCode::check_mode_for_custom_gcode_per_print_z(info);

            GUI::Plater::TakeSnapshot snap(plater, "API: set colour changes");
            plater->model().plates_custom_gcodes[plate->get_index()] = std::move(info);
            plate->update_slice_result_valid_state(false);
            plater->schedule_background_process();
        }, py::arg("changes"))
        .def("clear_color_changes", [](const PyPlate &p) {
            auto *plater = plater_or_throw("Plate.clear_color_changes");
            auto &list = plater->get_partplate_list();
            GUI::PartPlate *plate = list.get_plate(p.idx);
            if (plate == nullptr) throw std::runtime_error("plate gone");
            GUI::Plater::TakeSnapshot snap(plater, "API: clear colour changes");
            plater->model().plates_custom_gcodes.erase(plate->get_index());
            plate->update_slice_result_valid_state(false);
            plater->schedule_background_process();
        })
        .def("color_changes", [](const PyPlate &p) {
            auto *plater = plater_or_throw("Plate.color_changes");
            auto &list = plater->get_partplate_list();
            GUI::PartPlate *plate = list.get_plate(p.idx);
            if (plate == nullptr) throw std::runtime_error("plate gone");
            py::list out;
            auto &m = plater->model().plates_custom_gcodes;
            auto it = m.find(plate->get_index());
            if (it == m.end()) return out;
            for (const CustomGCode::Item &item : it->second.gcodes) {
                py::dict d;
                d["z"]        = item.print_z;
                d["extruder"] = item.extruder;
                d["color"]    = item.color;
                d["type"]     = int(item.type);
                out.append(d);
            }
            return out;
        });

    py::class_<PyPlateList>(m, "PlateList")
        .def_property_readonly("count", [](const PyPlateList &) {
            return plater_or_throw("PlateList.count")->get_partplate_list().get_plate_count();
        })
        .def("__len__", [](const PyPlateList &) {
            return plater_or_throw("PlateList.__len__")->get_partplate_list().get_plate_count();
        })
        .def("__getitem__", [](const PyPlateList &, int i) {
            int n = plater_or_throw("PlateList[]")->get_partplate_list().get_plate_count();
            if (i < 0 || i >= n) throw py::index_error("plate index out of range");
            return PyPlate{i};
        })
        // ---- M2 mutation --------------------------------------------------
        .def("arrange", [](const PyPlateList &, bool wait) {
            // Auto-arrange, same as the toolbar button. Asynchronous — starts a
            // background ArrangeJob (which snapshots itself). With wait=True,
            // pump the event loop (GIL released) until no job is running, so a
            // script can rely on the finished layout (same technique as
            // SliceJob.wait; not a nested modal loop).
            auto *plater = plater_or_throw("PlateList.arrange");
            plater->arrange();
            if (!wait) return;
            main_thread("PlateList.arrange(wait=True)");
            py::gil_scoped_release nogil;
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            for (;;) {
                if (wxTheApp != nullptr) wxTheApp->Yield(true);
                if (plater->get_ui_job_worker().is_idle()) break;
                if (clock::now() - t0 > std::chrono::seconds(120)) break;
                wxMilliSleep(40);
            }
        }, py::arg("wait") = false);

    // ---- SliceResult ------------------------------------------------------
    py::class_<PySliceResult>(m, "SliceResult")
        .def_property_readonly("success",       [](const PySliceResult &r) { return r.success; })
        .def_property_readonly("print_time_s",  [](const PySliceResult &r) { return r.print_time_s; })
        .def_property_readonly("layer_count",   [](const PySliceResult &r) { return r.layer_count; })
        .def_property_readonly("total_weight",  [](const PySliceResult &r) { return r.total_weight; })
        .def_property_readonly("total_cost",    [](const PySliceResult &r) { return r.total_cost; })
        .def_property_readonly("total_toolchanges", [](const PySliceResult &r) { return r.total_toolchanges; })
        .def_property_readonly("total_wipe_tower_filament", [](const PySliceResult &r) { return r.total_wipe_tower_filament; })
        .def_property_readonly("filament_g",    [](const PySliceResult &r) { return r.filament_g; })
        .def_property_readonly("filament_mm",   [](const PySliceResult &r) { return r.filament_mm; })
        .def_property_readonly("gcode_3mf_path",[](const PySliceResult &r) { return r.gcode_path; })
        .def_property_readonly("error",         [](const PySliceResult &r) { return r.error; });

    // ---- SliceJob ---------------------------------------------------------
    py::class_<PySliceJob>(m, "SliceJob")
        .def_property_readonly("done", [](const PySliceJob &j) {
            auto *plater = plater_or_throw("SliceJob.done");
            GUI::PartPlate *plate = plate_or_throw(j.plate_idx, "SliceJob.done");
            return plate->is_slice_result_valid() ||
                   (!plater->is_background_process_slicing() &&
                    !plater->is_background_process_update_scheduled());
        })
        .def_property_readonly("progress", [](const PySliceJob &) {
            // Pollable progress placeholder; push progress is M5. Percent is
            // not exposed publicly off the process, so report slicing state.
            auto *plater = plater_or_throw("SliceJob.progress");
            const char *stage = plater->is_background_process_slicing() ? "slicing" : "idle";
            return py::make_tuple(py::none(), std::string(stage));
        })
        .def("cancel", [](const PySliceJob &) {
            plater_or_throw("SliceJob.cancel")->get_ui_job_worker().cancel_all();
        })
        .def("wait", [](const PySliceJob &j, py::object timeout) -> PySliceResult {
            // Runs on the wx main thread (Python's thread in this model). Pump
            // the event loop so the slicing worker's completion/progress events
            // are delivered, polling the plate's validity — a controlled event
            // pump, NOT a nested modal loop. GIL is released while pumping.
            main_thread("SliceJob.wait");
            auto *plater = plater_or_throw("SliceJob.wait");
            const double timeout_s = timeout.is_none() ? 300.0 : timeout.cast<double>();

            PySliceResult res;
            {
                py::gil_scoped_release nogil;
                using clock = std::chrono::steady_clock;
                const auto t0 = clock::now();
                const auto start_grace = std::chrono::milliseconds(6000);
                GUI::PartPlate *plate = plater->get_partplate_list().get_plate(j.plate_idx);
                bool ever_busy = false;
                for (;;) {
                    if (wxTheApp != nullptr) wxTheApp->Yield(true);  // deliver events
                    if (plate != nullptr && plate->is_slice_result_valid()) {
                        res.success = true; break;
                    }
                    const bool busy = plater->is_background_process_slicing() ||
                                      plater->is_background_process_update_scheduled();
                    if (busy) ever_busy = true;
                    const auto elapsed = clock::now() - t0;
                    if (ever_busy && !busy) {
                        // It ran and stopped without producing a valid result.
                        // Give the completion event a few more turns to land.
                        for (int k = 0; k < 10 && !plate->is_slice_result_valid(); ++k) {
                            if (wxTheApp != nullptr) wxTheApp->Yield(true);
                            wxMilliSleep(50);
                        }
                        if (plate->is_slice_result_valid()) { res.success = true; }
                        else { res.success = false;
                               res.error = "slicing stopped without a valid result (slice error)"; }
                        break;
                    }
                    if (!ever_busy && elapsed > start_grace) {
                        res.success = false;
                        res.error = "slicing never started — plate not sliceable "
                                    "(invalid/empty config or objects off the plate)";
                        break;
                    }
                    if (elapsed > std::chrono::duration<double>(timeout_s)) {
                        res.success = false;
                        res.error = "timeout waiting for slice";
                        break;
                    }
                    wxMilliSleep(40);
                }
            }
            if (res.success) fill_slice_result(j.plate_idx, res);
            return res;
        }, py::arg("timeout") = py::none());

    // ---- Document ---------------------------------------------------------
    py::class_<PyDocument>(m, "Document")
        // Kept from M0 for continuity:
        .def_property_readonly("object_count", [](const PyDocument &) {
            return model_or_throw("Document.object_count").objects.size();
        })
        .def_property_readonly("model", [](const PyDocument &) { return PyModel{}; })
        .def_property_readonly("plates", [](const PyDocument &) { return PyPlateList{}; })
        .def_property_readonly("config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Global};
        })
        .def_property_readonly("print_config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Print};
        })
        .def_property_readonly("filament_config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Filament};
        })
        .def_property_readonly("printer_config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Printer};
        })
        // ---- M3 slicing ---------------------------------------------------
        // Set up N project filaments with the given colours (UI-parity: the
        // Sidebar "+" add-filament path). Colours are "#RRGGBB". Enables
        // multi-colour prints; build_ams_mapping matches these to AMS slots.
        .def("set_filaments", [](const PyDocument &, py::list colors) {
            auto *plater = plater_or_throw("Document.set_filaments");
            auto *pb = GUI::wxGetApp().preset_bundle;
            const unsigned int n = (unsigned int) colors.size();
            if (n < 1 || n > 16) throw std::runtime_error("filament count must be 1..16");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: set filaments"));
            pb->set_num_filaments(n);
            std::vector<std::string> cols;
            for (auto c : colors) {
                std::string h = c.cast<std::string>();
                if (!h.empty() && h[0] != '#') h = "#" + h;
                cols.push_back(h);
            }
            if (auto *opt = pb->project_config.option<ConfigOptionStrings>("filament_colour"))
                opt->values = cols;
            plater->on_filament_count_change(n);
            plater->on_config_change(pb->full_config());   // schedule reslice (not via Tab)
            return int(n);
        }, py::arg("colors"))
        .def("slice", [](const PyDocument &, py::object plate) {
            auto *plater = plater_or_throw("Document.slice");
            auto &list = plater->get_partplate_list();
            int idx;
            if (plate.is_none()) {
                idx = list.get_curr_plate_index();
            } else {
                idx = plate.cast<int>();
                if (idx < 0 || idx >= list.get_plate_count())
                    throw std::runtime_error("plate index out of range");
                list.select_plate(idx);   // reslice targets the current plate
            }
            plater->reslice();            // starts the background slicing worker
            return PySliceJob{idx};
        }, py::arg("plate") = py::none())
        // Copy the current plate's last sliced G-code to `path`.
        .def("save_gcode", [](const PyDocument &, const std::string &path) {
            namespace fs = boost::filesystem;
            auto *plater = plater_or_throw("Document.save_gcode");
            auto &list = plater->get_partplate_list();
            GUI::PartPlate *plate = list.get_plate(list.get_curr_plate_index());
            if (plate == nullptr) throw std::runtime_error("no current plate");
            fs::path src(plate->get_tmp_gcode_path());
            if (src.empty() || !fs::exists(src))
                throw std::runtime_error("no sliced G-code yet; call slice().wait() first");
            fs::copy_file(src, fs::path(path), fs::copy_option::overwrite_if_exists);
            return path;
        }, py::arg("path"))
        // Save the project as a .3mf (UI-parity: Save Project).
        .def("save_3mf", [](const PyDocument &, const std::string &path) {
            namespace fs = boost::filesystem;
            auto *plater = plater_or_throw("Document.save_3mf");
            plater->export_3mf(fs::path(path));
            if (!fs::exists(fs::path(path)) || fs::file_size(fs::path(path)) == 0)
                throw std::runtime_error("3mf not written: " + path);
            return path;
        }, py::arg("path"));

    // ---- Application ------------------------------------------------------
    py::class_<PyApp>(m, "Application")
        .def_property_readonly("version", [](const PyApp &) {
            main_thread("app.version");
            return std::string(SLIC3R_VERSION);
        })
        .def_property_readonly("name", [](const PyApp &) {
            main_thread("app.name");
            return std::string(SLIC3R_APP_NAME);   // BambuStudio / OrcaSlicer / PrusaSlicer
        })
        .def_property_readonly("active_document", [](const PyApp &) -> py::object {
            main_thread("app.active_document");
            if (GUI::wxGetApp().plater() == nullptr)
                return py::none();
            return py::cast(PyDocument{});
        })
        .def_property_readonly("selected_printer", [](const PyApp &) {
            main_thread("app.selected_printer");
            return GUI::wxGetApp().preset_bundle->printers.get_selected_preset_name();
        })
        .def("printers", [](const PyApp &) {
            main_thread("app.printers");
            // Mirror the GUI dropdown: visible printer presets only.
            std::vector<std::string> out;
            for (const Preset &p : GUI::wxGetApp().preset_bundle->printers.get_presets())
                if (p.is_visible)
                    out.push_back(p.name);
            return out;
        })
        // Cloud device plane (M4). Registered in PyDevice.cpp; exposed here as
        // app.device. Account/app-level (one logged-in account), so it hangs
        // off Application, not Document.
        .def_property_readonly("device", [](const PyApp &) {
            return py::module_::import("pyslic3r").attr("_device_singleton");
        });

    m.attr("app") = py::cast(PyApp{});

    // MakerWorld model download — Python (urllib) transport to navigate
    // Cloudflare (which fingerprints/blocks libcurl). Reuses the C++ auth
    // primitive device.request_bind_ticket(). Exposed as pyslic3r.download_model.
    {
        py::dict ns;
        py::exec(R"PYSRC(
def download_model(url_or_id, instance_id=None, out_dir=None, retries=4):
    """Download a MakerWorld model's 3mf via the logged-in Bambu account.
    url_or_id: MakerWorld URL (…/models/<design>…#profileId-<instance>) or design id.
    Returns {name, path, size, design_id, instance_id}. Navigates Cloudflare via
    Python's TLS (native libcurl is blocked); retries transient 403/429/5xx."""
    import pyslic3r as _ps, urllib.request, urllib.parse, http.cookiejar, json, os, re, tempfile, time
    dev = _ps.app.device
    if not dev.is_logged_in:
        raise RuntimeError("download_model: not logged in to a Bambu account")
    design_id, inst = None, instance_id
    m = re.search(r"/models/(\d+)", str(url_or_id))
    if m:
        design_id = m.group(1)
    m = re.search(r"profileId-(\d+)", str(url_or_id))
    if m and inst is None:
        inst = m.group(1)
    if design_id is None:
        design_id = str(url_or_id)
    if inst is None:
        raise RuntimeError("download_model: need instance_id= or a URL containing #profileId-<id>")
    host = "https://makerworld.com/"
    hdrs = {"User-Agent": "BambuStudio", "accept": "*/*"}
    TRANSIENT = (403, 429, 500, 502, 503, 504)
    err = None
    for attempt in range(max(1, retries)):
        try:
            ticket = dev.request_bind_ticket()
            if not ticket:
                raise RuntimeError("request_bind_ticket failed (login/session?)")
            f3mf = host + "api/v1/design-service/instance/%s/f3mf" % inst
            signin = host + "api/sign-in/ticket?to=" + urllib.parse.quote(f3mf, safe="") + "&ticket=" + ticket
            cj = http.cookiejar.CookieJar()
            op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
            meta = op.open(urllib.request.Request(signin, headers=hdrs), timeout=60).read()
            j = json.loads(meta)
            cdn = j.get("url")
            name = j.get("name") or ("model_%s.3mf" % inst)
            if not cdn:
                raise RuntimeError("could not resolve file url: %r" % meta[:160])
            data = op.open(urllib.request.Request(cdn, headers=hdrs), timeout=180).read()
            if not data:
                raise RuntimeError("downloaded file is empty")
            d = out_dir or tempfile.gettempdir()
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, name)
            with open(path, "wb") as f:
                f.write(data)
            return {"name": name, "path": path, "size": len(data),
                    "design_id": design_id, "instance_id": inst}
        except urllib.error.HTTPError as e:
            err = e
            if e.code in TRANSIENT and attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt)); continue
            raise RuntimeError("download_model: HTTP %s from MakerWorld (Cloudflare?)" % e.code)
        except Exception as e:
            err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt)); continue
            raise
    raise RuntimeError("download_model failed after %d attempts: %s" % (retries, err))
)PYSRC", ns, ns);
        m.attr("download_model") = ns["download_model"];
    }
}

} // namespace pyslic3r
