// pyslic3r cloud device plane (M4) — READ-ONLY surface.
//
// Login state, account-bound printer enumeration, selection, and status
// read-back, routed through the same NetworkAgent / DeviceManager the GUI's
// Device page uses. Invariants:
//   - Read-only here: no send/print/config — those are gated behind explicit
//     user intent and a later slice (M4 write half).
//   - Strictly isolated: everything degrades gracefully if the network plugin
//     isn't loaded or the cloud is unreachable (null checks, never throw the
//     whole app down). A cloud outage must not affect model/slice work.
//   - Main-thread-guarded: NetworkAgent/DeviceManager live on the wx main
//     thread; off-thread callers must come through run_on_main_blocking.
//   - Handles are dev_id strings, re-resolved per call — a stale handle
//     returns None/raises rather than dereferencing a freed MachineObject.

#include "PyBindings.hpp"

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <pybind11/stl.h>

#include <wx/app.h>
#include <wx/thread.h>
#include <wx/utils.h>   // wxMilliSleep

#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/Utils/NetworkAgent.hpp"
#include "slic3r/GUI/DeviceCore/DevManager.h"
#include "slic3r/GUI/DeviceManager.hpp"
#include "slic3r/GUI/DeviceCore/DevBed.h"
#include "slic3r/GUI/DeviceCore/DevExtruderSystem.h"
#include "slic3r/GUI/DeviceCore/DevChamber.h"
#include "slic3r/GUI/DeviceCore/DevHMS.h"
#include "slic3r/GUI/Plater.hpp"
#include "slic3r/GUI/PartPlate.hpp"
#include "slic3r/GUI/Jobs/PrintJob.hpp"      // PrintPrepareData
#include "slic3r/Utils/bambu_networking.hpp" // BBL::PrintParams
#include "libslic3r/ProjectTask.hpp"            // Slic3r::FilamentInfo
#include "libslic3r/PresetBundle.hpp"           // PresetBundle
#include "slic3r/GUI/DeviceCore/DevMapping.h"   // DevMappingUtil (auto-mapper)
#include "slic3r/GUI/DeviceCore/DevDefs.h"      // VIRTUAL_TRAY_MAIN_ID / DEPUTY
#include "slic3r/GUI/BitmapCache.hpp"           // BitmapCache::parse_color4
#include <cstdio>                               // std::snprintf

namespace py = pybind11;
using namespace Slic3r;

namespace pyslic3r {
namespace {

void dev_main_thread(const char *what)
{
    if (!wxThread::IsMain())
        throw std::runtime_error(std::string(what) +
            " must be accessed on the wx main thread; use the marshalling primitive");
}

// NetworkAgent is null until the signed network plugin is loaded. Return it
// (possibly null) — callers treat null as "device plane unavailable".
NetworkAgent *agent(const char *what)
{
    dev_main_thread(what);
    return GUI::wxGetApp().getAgent();
}

DeviceManager *devmgr(const char *what)
{
    dev_main_thread(what);
    return GUI::wxGetApp().getDeviceManager();
}

GUI::Plater *plater_dev(const char *what)
{
    dev_main_thread(what);
    GUI::Plater *p = GUI::wxGetApp().plater();
    if (p == nullptr) throw std::runtime_error("no active document");
    return p;
}

bool logged_in(const char *what)
{
    NetworkAgent *a = agent(what);
    return a != nullptr && a->is_user_login();
}

// dev_id -> MachineObject*, or nullptr if gone.
MachineObject *machine(const std::string &dev_id, const char *what)
{
    DeviceManager *dm = devmgr(what);
    return dm == nullptr ? nullptr : dm->get_my_machine(dev_id);
}

// Merged account-bound + local machine list (dev_id -> MachineObject*).
std::map<std::string, MachineObject *> all_machines(const char *what)
{
    DeviceManager *dm = devmgr(what);
    std::map<std::string, MachineObject *> out;
    if (dm == nullptr) return out;
    out = dm->get_my_machine_list();                       // local + mine
    for (auto &kv : dm->get_user_machinelist())            // account-bound (cloud)
        if (kv.second != nullptr) out.emplace(kv.first, kv.second);
    return out;
}

// Build the full AMS mapping table (v0 array + v1 {ams_id,slot_id} array +
// info table) via the app's own auto-mapper. A dispatch that sends only a bare
// v0 mapping makes the printer fail with "Failed to get AMS mapping table"
// (HMS 0700-8012). Single-nozzle only; mirrors SelectMachineDialog
// do_ams_mapping + get_ams_mapping_result (FROM_NORMAL). Fills
// params.ams_mapping / ams_mapping2 / ams_mapping_info. Returns false if no
// valid mapping was produced (caller then falls back to the manual string).
bool build_ams_mapping(MachineObject *mo, GUI::Plater *plater, int plate_idx,
                       bool use_ams, BBL::PrintParams &params)
{
    if (mo == nullptr || plater == nullptr) return false;
    PresetBundle *pb = GUI::wxGetApp().preset_bundle;
    if (pb == nullptr) return false;

    auto &list = plater->get_partplate_list();
    GUI::PartPlate *pp = list.get_plate(plate_idx);
    if (pp == nullptr) return false;

    // Per-filament preset metadata, indexed to match filament_presets.
    struct PInfo { std::string filament_id, setting_id, type; };
    std::vector<PInfo> preset_info;
    for (const auto &name : pb->filament_presets) {
        PInfo pi;
        Preset *preset = pb->filaments.find_preset(name);
        if (preset != nullptr) {
            pi.filament_id = preset->filament_id;
            pi.setting_id  = preset->setting_id;
            std::string disp;
            pi.type = preset->get_filament_type(disp);
        }
        preset_info.push_back(std::move(pi));
    }

    // Input filament list from the plate's used filaments (1-based -> 0-based).
    std::vector<int> used = pp->get_used_filaments();
    std::vector<FilamentInfo> in_filaments;
    for (size_t i = 0; i < used.size(); ++i) {
        int u = used[i] - 1;
        if (u < 0 || u >= (int)preset_info.size()) continue;
        std::string colour = pb->project_config.opt_string("filament_colour",
                                                           (unsigned int)u);
        unsigned char rgba[4] = {0, 0, 0, 255};
        GUI::BitmapCache::parse_color4(colour, rgba);
        char cbuf[10];
        std::snprintf(cbuf, sizeof(cbuf), "#%02X%02X%02X%02X",
                      rgba[0], rgba[1], rgba[2], rgba[3]);
        FilamentInfo info;
        info.id          = u;
        info.type        = preset_info[u].type;
        info.filament_id = preset_info[u].filament_id;
        info.setting_id  = preset_info[u].setting_id;
        info.color       = cbuf;
        in_filaments.push_back(std::move(info));
    }
    if (in_filaments.empty()) return false;

    if (!mo->HasAms()) use_ams = false;
    std::vector<bool> map_opt = use_ams
        ? std::vector<bool>{false, true, false, false}   // USE_RIGHT_AMS
        : std::vector<bool>{false, false, false, true};  // USE_RIGHT_EXT

    std::vector<FilamentInfo> result;
    if (DevMappingUtil::ams_filament_mapping(mo, in_filaments, result, map_opt) != 0)
        return false;
    if (!DevMappingUtil::is_valid_mapping_result(mo, result))
        return false;

    nlohmann::json v0   = nlohmann::json::array();
    nlohmann::json v1   = nlohmann::json::array();
    nlohmann::json info = nlohmann::json::array();
    for (int i = 0; i < (int)pb->filament_presets.size(); ++i) {
        int tray_id = -1;
        nlohmann::json item_v1;
        item_v1["ams_id"]  = 0xff;
        item_v1["slot_id"] = 0xff;
        nlohmann::json item;
        item["ams"]          = tray_id;
        item["targetColor"]  = "";
        item["filamentId"]   = "";
        item["filamentType"] = "";
        for (int k = 0; k < (int)result.size(); ++k) {
            if (result[k].id != i) continue;
            tray_id              = result[k].tray_id;
            item["ams"]          = tray_id;
            item["filamentType"] = result[k].type;
            item["filamentId"]   = result[k].filament_id;
            item["sourceColor"]  = in_filaments[k].color;
            item["targetColor"]  = result[k].color;
            if (tray_id == VIRTUAL_TRAY_MAIN_ID || tray_id == VIRTUAL_TRAY_DEPUTY_ID)
                tray_id = -1;
            try {
                if (result[k].ams_id.empty() || result[k].slot_id.empty()) {
                    item_v1["ams_id"]  = VIRTUAL_TRAY_MAIN_ID;
                    item_v1["slot_id"] = VIRTUAL_TRAY_MAIN_ID;
                } else {
                    item_v1["ams_id"]  = std::stoi(result[k].ams_id);
                    item_v1["slot_id"] = std::stoi(result[k].slot_id);
                }
            } catch (...) {}
            break;
        }
        v0.push_back(tray_id);
        v1.push_back(item_v1);
        info.push_back(item);
    }
    params.ams_mapping      = v0.dump();
    params.ams_mapping2     = v1.dump();
    params.ams_mapping_info = info.dump();
    return true;
}

const char *hms_level_str(Slic3r::HMSMessageLevel l)
{
    switch (l) {
    case Slic3r::HMS_FATAL:   return "fatal";
    case Slic3r::HMS_SERIOUS: return "serious";
    case Slic3r::HMS_COMMON:  return "common";
    case Slic3r::HMS_INFO:    return "info";
    default:                  return "unknown";
    }
}

struct PyDevice {};
struct PyBoundPrinter { std::string dev_id; };

} // anonymous namespace

void register_device(py::module_ &m)
{
    // ---- BoundPrinter -----------------------------------------------------
    py::class_<PyBoundPrinter>(m, "BoundPrinter")
        .def_property_readonly("dev_id", [](const PyBoundPrinter &p) { return p.dev_id; })
        .def_property_readonly("name", [](const PyBoundPrinter &p) {
            MachineObject *mo = machine(p.dev_id, "BoundPrinter.name");
            return mo ? mo->get_dev_name() : std::string();
        })
        .def_property_readonly("online", [](const PyBoundPrinter &p) {
            MachineObject *mo = machine(p.dev_id, "BoundPrinter.online");
            return mo ? mo->is_online() : false;
        })
        .def_property_readonly("connection_type", [](const PyBoundPrinter &p) {
            MachineObject *mo = machine(p.dev_id, "BoundPrinter.connection_type");
            return mo ? mo->connection_type() : std::string();
        });

    // ---- Device -----------------------------------------------------------
    py::class_<PyDevice>(m, "Device")
        // Is the signed network plugin loaded at all?
        .def_property_readonly("available", [](const PyDevice &) {
            return agent("Device.available") != nullptr;
        })
        .def_property_readonly("is_logged_in", [](const PyDevice &) {
            return logged_in("Device.is_logged_in");
        })
        .def_property_readonly("user_id", [](const PyDevice &) -> py::object {
            NetworkAgent *a = agent("Device.user_id");
            if (a == nullptr || !a->is_user_login()) return py::none();
            return py::str(a->get_user_id());
        })
        .def_property_readonly("user_name", [](const PyDevice &) -> py::object {
            NetworkAgent *a = agent("Device.user_name");
            if (a == nullptr || !a->is_user_login()) return py::none();
            std::string n = a->get_user_nickanme();          // (sic — upstream typo)
            if (n.empty()) n = a->get_user_name();
            return py::str(n);
        })
        .def("printers", [](const PyDevice &) {
            // Account-bound (+ local) printers, from the cached list. Call
            // refresh() first to fetch it from the cloud (a fresh instance
            // hasn't). Empty list is a valid result (nothing bound).
            py::list out;
            for (auto &kv : all_machines("Device.printers"))
                out.append(PyBoundPrinter{kv.first});
            return out;
        })
        .def("refresh", [](const PyDevice &) {
            // Fetch the account's bound-printer list from the cloud (what the
            // GUI does on the Device page). update_user_machine_list_info()
            // makes a synchronous HTTP call then defers the JSON parse via
            // CallAfter — so pump the event loop until the list populates,
            // then return the count. Returns 0 if not logged in / unavailable.
            DeviceManager *dm = devmgr("Device.refresh");
            if (dm == nullptr) return size_t(0);
            NetworkAgent *a = GUI::wxGetApp().getAgent();
            if (a == nullptr || !a->is_user_login()) return size_t(0);

            py::gil_scoped_release nogil;
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            // Re-issue the fetch periodically once the server MQTT is up: a
            // fresh instance's first fetch can race the login/connect, and the
            // JSON parse is deferred (CallAfter). Stop as soon as the list
            // populates. (An empty result is legitimate — nothing bound.)
            clock::time_point last_fetch{};   // epoch => fetch on the first eligible turn
            for (;;) {
                if (wxTheApp != nullptr) wxTheApp->Yield(true);   // run the deferred parse
                if (a->is_server_connected() &&
                    clock::now() - last_fetch > std::chrono::seconds(3)) {
                    dm->update_user_machine_list_info();
                    last_fetch = clock::now();
                }
                if (!dm->get_user_machinelist().empty()) break;  // populated
                if (clock::now() - t0 > std::chrono::seconds(15)) break;
                wxMilliSleep(120);
            }
            return dm->get_user_machinelist().size();
        })
        .def_property_readonly("selected", [](const PyDevice &) -> py::object {
            DeviceManager *dm = devmgr("Device.selected");
            if (dm == nullptr) return py::none();
            MachineObject *mo = dm->get_selected_machine();
            if (mo == nullptr) return py::none();
            return py::cast(PyBoundPrinter{mo->get_dev_id()});
        })
        .def("select", [](const PyDevice &, const std::string &dev_id) {
            DeviceManager *dm = devmgr("Device.select");
            if (dm == nullptr || !dm->set_selected_machine(dev_id))
                throw std::runtime_error("could not select device: " + dev_id);
        }, py::arg("dev_id"))
        .def("status", [](const PyDevice &, double wait) -> py::object {
            // Live status for the selected printer. None if nothing selected /
            // unavailable. With wait>0, establish the telemetry: a headless
            // offscreen app is never "studio active", so its app-subscribe
            // never fires and no status flows — force the subscribe, request a
            // full push, and pump the loop until the push lands (or timeout).
            DeviceManager *dm = devmgr("Device.status");
            if (dm == nullptr) return py::none();
            MachineObject *mo = dm->get_selected_machine();
            if (mo == nullptr) return py::none();

            if (wait > 0.0) {
                NetworkAgent *a = GUI::wxGetApp().getAgent();
                if (a != nullptr) a->start_subscribe("app");
                mo->reset_update_time();
                mo->command_request_push_all(true);
                py::gil_scoped_release nogil;
                using clock = std::chrono::steady_clock;
                const auto t0 = clock::now();
                for (;;) {
                    if (wxTheApp != nullptr) wxTheApp->Yield(true);
                    // connected AND a full status push received (is_connecting
                    // = connected but push_count==0):
                    if (mo->is_connected() && !mo->is_connecting()) break;
                    if (clock::now() - t0 > std::chrono::duration<double>(wait)) break;
                    mo->command_request_push_all(false);   // throttled to 3s internally
                    wxMilliSleep(120);
                }
            }

            py::dict d;
            d["dev_id"]        = mo->get_dev_id();
            d["online"]        = mo->is_online();
            d["connected"]     = mo->is_connected();
            d["awaiting_push"] = mo->is_connecting();   // connected, no full status yet
            d["print_status"]  = mo->print_status;      // RUNNING/PAUSE/FINISH/...
            d["stage"]         = std::string(mo->get_curr_stage().ToUTF8().data());
            d["progress"]      = mo->mc_print_percent;  // 0..100
            d["current_layer"] = mo->curr_layer;
            d["total_layers"]  = mo->total_layers;
            d["remaining_s"]   = mo->mc_left_time;
            d["subtask_name"]  = mo->subtask_name;

            if (DevBed *bed = mo->GetBed()) {
                d["bed_temp"]        = bed->GetBedTemp();
                d["bed_temp_target"] = bed->GetBedTempTarget();
            }
            if (DevExtderSystem *ex = mo->GetExtderSystem()) {
                py::list nozzles;
                const int n = ex->GetTotalExtderCount();
                for (int i = 0; i < n; ++i) {
                    py::dict nz;
                    nz["current"] = ex->GetNozzleTempCurrent(i);
                    nz["target"]  = ex->GetNozzleTempTarget(i);
                    nozzles.append(nz);
                }
                d["nozzles"] = nozzles;
            }
            if (std::shared_ptr<DevChamber> ch = mo->GetChamber()) {
                if (ch->HasChamber()) {
                    d["chamber_temp"]        = ch->GetChamberTemp();
                    d["chamber_temp_target"] = ch->GetChamberTempTarget();
                }
            }
            if (DevHMS *hms = mo->GetHMS()) {
                py::list items;
                for (const auto &it : hms->GetHMSItems()) {
                    py::dict h;
                    h["level"] = std::string(hms_level_str(it.get_level()));
                    h["code"]  = it.get_long_error_code();
                    items.append(h);
                }
                d["hms"] = items;
            }
            return d;
        }, py::arg("wait") = 0.0)

        // ---- camera (read-only) -------------------------------------------
        .def("camera_url", [](const PyDevice &, double timeout) -> py::object {
            // The liveview stream URL for the selected printer (bambu:///tutk|
            // agora|rtsp|local). READ-ONLY. Async: the plugin invokes a
            // callback with the URL; pump until it fires. Decoding an actual
            // frame from this stream is a separate media-pipeline task (needs
            // libBambuSource + ffmpeg) — deferred; this returns the URL an
            // external player/decoder can consume.
            NetworkAgent *a = agent("Device.camera_url");
            DeviceManager *dm = devmgr("Device.camera_url");
            if (a == nullptr || dm == nullptr) return py::none();
            MachineObject *mo = dm->get_selected_machine();
            if (mo == nullptr) throw std::runtime_error("no printer selected (device.select first)");

            auto url = std::make_shared<std::string>();
            auto got = std::make_shared<std::atomic<bool>>(false);
            a->get_camera_url(mo->get_dev_id(), [url, got](std::string u) {
                *url = std::move(u);
                got->store(true);
            });
            {
                py::gil_scoped_release nogil;
                using clock = std::chrono::steady_clock;
                const auto t0 = clock::now();
                while (!got->load()) {
                    if (wxTheApp != nullptr) wxTheApp->Yield(true);
                    if (clock::now() - t0 > std::chrono::duration<double>(timeout)) break;
                    wxMilliSleep(60);
                }
            }
            if (!got->load() || url->empty()) return py::none();
            return py::str(*url);
        }, py::arg("timeout") = 10.0)

        // ---- send (WRITE — dispatch a sliced plate to the printer) ---------
        // dry_run=True does everything EXCEPT the dispatch (export the sliced
        // 3mf, build the print params) — safe to call anytime. dry_run=False
        // actually starts the print on the physical printer and MUST only be
        // called with the user's explicit, per-print approval.
        .def("send", [](const PyDevice &, py::object plate, bool dry_run,
                        const std::string &project_name, bool use_ams,
                        const std::string &ams_mapping, bool bed_leveling,
                        bool flow_cali, bool timelapse) -> py::dict {
            GUI::Plater *plater = plater_dev("Device.send");
            DeviceManager *dm = devmgr("Device.send");
            NetworkAgent *a = GUI::wxGetApp().getAgent();
            if (dm == nullptr || a == nullptr)
                throw std::runtime_error("device plane unavailable");
            MachineObject *mo = dm->get_selected_machine();
            if (mo == nullptr)
                throw std::runtime_error("no printer selected (device.select first)");

            auto &list = plater->get_partplate_list();
            const int idx = plate.is_none() ? list.get_curr_plate_index()
                                            : plate.cast<int>();
            if (idx < 0 || idx >= list.get_plate_count())
                throw std::runtime_error("plate index out of range");
            GUI::PartPlate *pp = list.get_plate(idx);
            if (pp == nullptr || !pp->is_slice_result_valid())
                throw std::runtime_error("plate not sliced — call doc.slice() first");

            // Export the sliced 3mf (+ config 3mf for cloud) — same as GUI send.
            if (!plate.is_none()) list.select_plate(idx);
            const int rc = plater->send_gcode(idx, nullptr);
            if (rc != 0)
                throw std::runtime_error("send_gcode export failed rc=" + std::to_string(rc));
            plater->export_config_3mf(idx, nullptr);

            GUI::PrintPrepareData jd;
            plater->get_print_job_data(&jd);

            BBL::PrintParams params;
            params.dev_id          = mo->get_dev_id();
            params.dev_name        = mo->get_dev_name();
            params.connection_type = mo->connection_type();
            params.dev_ip          = mo->get_dev_ip();
            params.username        = "bblp";
            params.password        = mo->get_access_code();
            params.ftp_folder      = mo->get_ftp_folder();
            params.filename        = jd._3mf_path.string();
            params.config_filename = jd._3mf_config_path.string();
            params.plate_index     = idx + 1;   // 1-based on the wire
            params.project_name    = project_name;
            params.task_use_ams    = use_ams;
            // Full AMS mapping table via the app's auto-mapper (a bare v0
            // string alone -> "Failed to get AMS mapping table"). A non-empty
            // manual ams_mapping still wins only if the auto-map can't run.
            if (!(use_ams && build_ams_mapping(mo, plater, idx, use_ams, params)))
                params.ams_mapping = ams_mapping;
            params.task_bed_leveling     = bed_leveling;
            params.task_flow_cali        = flow_cali;
            params.task_record_timelapse = timelapse;
            params.auto_bed_leveling     = bed_leveling ? 2 : 0;
            params.auto_flow_cali        = flow_cali ? 2 : 0;

            py::dict out;
            out["dev_id"]      = params.dev_id;
            out["gcode_3mf"]   = params.filename;
            out["config_3mf"]  = params.config_filename;
            out["plate_index"] = params.plate_index;
            out["connection"]  = params.connection_type;

            if (dry_run) {
                out["dry_run"]    = true;
                out["dispatched"] = false;
                out["ready"]      = !params.filename.empty();
                return out;
            }

            // === REAL DISPATCH — physical print starts here. ===============
            // Run start_print on a worker thread and pump the main loop, so the
            // agent's MQTT confirmation (polled by wait_fn) can be delivered —
            // mirrors the GUI's PrintJob (a background job).
            std::atomic<bool> done{false};
            std::atomic<int>  sret{-999};
            std::mutex        info_mtx;
            std::string       info_snap;
            int               stage_snap = -1;
            auto update_fn = [&](int stage, int /*code*/, std::string info) {
                std::lock_guard<std::mutex> lk(info_mtx);
                stage_snap = stage; info_snap = std::move(info);
            };
            auto cancel_fn = []() { return false; };
            auto wait_fn   = [](int, std::string) { return true; };

            std::thread worker([&]() {
                const int r = a->start_print(params, update_fn, cancel_fn, wait_fn);
                sret.store(r);
                done.store(true);
            });
            {
                py::gil_scoped_release nogil;
                while (!done.load()) {
                    if (wxTheApp != nullptr) wxTheApp->Yield(true);
                    wxMilliSleep(80);
                }
            }
            worker.join();

            out["dry_run"]     = false;
            out["dispatched"]  = (sret.load() == 0);
            out["result_code"] = sret.load();
            {
                std::lock_guard<std::mutex> lk(info_mtx);
                out["stage"] = stage_snap;
                out["info"]  = info_snap;
            }
            return out;
        }, py::arg("plate") = py::none(), py::arg("dry_run") = false,
           py::arg("project_name") = std::string("pyslic3r"),
           py::arg("use_ams") = false, py::arg("ams_mapping") = std::string(),
           py::arg("bed_leveling") = false, py::arg("flow_cali") = false,
           py::arg("timelapse") = false)

        // ---- stage (WRITE — upload a sliced plate to the printer's storage
        // WITHOUT starting it) ----------------------------------------------
        // Mirrors the GUI's "Send to Printer" (SendJob -> start_send_gcode_to_
        // sdcard): the sliced 3mf is placed on the printer so the user can
        // start it themselves from the printer screen or Bambu Handy. This
        // NEVER issues a "start print" command — the human presses Print. So
        // it honours the per-print-approval rule by construction, and for a
        // LAN printer it is fully local (FTP, no cloud round-trip at all).
        // dry_run=True exports + prepares only (no upload).
        .def("stage", [](const PyDevice &, py::object plate, bool dry_run,
                         const std::string &project_name, bool use_ams,
                         const std::string &ams_mapping) -> py::dict {
            GUI::Plater *plater = plater_dev("Device.stage");
            DeviceManager *dm = devmgr("Device.stage");
            NetworkAgent *a = GUI::wxGetApp().getAgent();
            if (dm == nullptr || a == nullptr)
                throw std::runtime_error("device plane unavailable");
            MachineObject *mo = dm->get_selected_machine();
            if (mo == nullptr)
                throw std::runtime_error("no printer selected (device.select first)");

            auto &list = plater->get_partplate_list();
            const int idx = plate.is_none() ? list.get_curr_plate_index()
                                            : plate.cast<int>();
            if (idx < 0 || idx >= list.get_plate_count())
                throw std::runtime_error("plate index out of range");
            GUI::PartPlate *pp = list.get_plate(idx);
            if (pp == nullptr || !pp->is_slice_result_valid())
                throw std::runtime_error("plate not sliced — call doc.slice() first");

            // Export the sliced 3mf (+ config 3mf) — same as GUI send/stage.
            if (!plate.is_none()) list.select_plate(idx);
            const int rc = plater->send_gcode(idx, nullptr);
            if (rc != 0)
                throw std::runtime_error("send_gcode export failed rc=" + std::to_string(rc));
            plater->export_config_3mf(idx, nullptr);

            GUI::PrintPrepareData jd;
            plater->get_print_job_data(&jd);

            BBL::PrintParams params;
            params.dev_id          = mo->get_dev_id();
            params.dev_name        = mo->get_dev_name();
            params.connection_type = mo->connection_type();
            params.dev_ip          = mo->get_dev_ip();
            params.username        = "bblp";
            params.password        = mo->get_access_code();
            params.ftp_folder      = mo->get_ftp_folder();
            params.filename        = jd._3mf_path.string();
            params.config_filename = jd._3mf_config_path.string();
            params.plate_index     = idx + 1;   // 1-based on the wire
            params.project_name    = project_name;
            params.task_use_ams    = use_ams;
            // Full AMS mapping table via the app's auto-mapper (a bare v0
            // string alone -> "Failed to get AMS mapping table"). A non-empty
            // manual ams_mapping still wins only if the auto-map can't run.
            if (!(use_ams && build_ams_mapping(mo, plater, idx, use_ams, params)))
                params.ams_mapping = ams_mapping;

            py::dict out;
            out["dev_id"]          = params.dev_id;
            out["gcode_3mf"]       = params.filename;
            out["config_3mf"]      = params.config_filename;
            out["plate_index"]     = params.plate_index;
            out["connection"]      = params.connection_type;
            out["supports_sdcard"] = mo->is_support_send_to_sdcard;

            if (dry_run) {
                out["dry_run"] = true;
                out["staged"]  = false;
                out["ready"]   = !params.filename.empty();
                return out;
            }

            // Upload to the printer's storage on a worker thread while pumping
            // the main loop (mirrors the GUI's SendJob). NO print is started.
            std::atomic<bool> done{false};
            std::atomic<int>  sret{-999};
            std::mutex        info_mtx;
            std::string       info_snap;
            int               stage_snap = -1;
            auto update_fn = [&](int stage, int /*code*/, std::string info) {
                std::lock_guard<std::mutex> lk(info_mtx);
                stage_snap = stage; info_snap = std::move(info);
            };
            auto cancel_fn = []() { return false; };
            auto wait_fn   = [](int, std::string) { return true; };

            std::thread worker([&]() {
                const int r = a->start_send_gcode_to_sdcard(params, update_fn,
                                                            cancel_fn, wait_fn);
                sret.store(r);
                done.store(true);
            });
            {
                py::gil_scoped_release nogil;
                while (!done.load()) {
                    if (wxTheApp != nullptr) wxTheApp->Yield(true);
                    wxMilliSleep(80);
                }
            }
            worker.join();

            out["dry_run"]     = false;
            out["staged"]      = (sret.load() == 0);
            out["result_code"] = sret.load();
            {
                std::lock_guard<std::mutex> lk(info_mtx);
                out["stage"] = stage_snap;
                out["info"]  = info_snap;
            }
            return out;
        }, py::arg("plate") = py::none(), py::arg("dry_run") = false,
           py::arg("project_name") = std::string("pyslic3r"),
           py::arg("use_ams") = false, py::arg("ams_mapping") = std::string());

    m.attr("_device_singleton") = py::cast(PyDevice{});
}

} // namespace pyslic3r
