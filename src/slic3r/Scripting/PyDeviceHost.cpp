///|/ pyslic3r — open device plane for OrcaSlicer (PrintHost transport).
///|/
///|/ Bambu's device plane is Bambu-cloud MQTT (PyDevice.cpp, compiled out on the
///|/ Orca port). Orca's *open* device plane is the PrintHost HTTP layer that
///|/ backs the GUI's "Physical Printer -> Test / Send" flow — OctoPrint,
///|/ Moonraker, PrusaLink, &c. This TU exposes that path as pyslic3r's `device`
///|/ so the same façade drives an open printer (and its digital twin) the way
///|/ MQTT drives a Bambu one. UI-parity: every operation here is something a
///|/ user does through Orca's Physical Printer settings and Send button.
///|/
#include "PyBindings.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/stl.h>

#include <wx/string.h>

#include "libslic3r/Config.hpp"
#include "libslic3r/Preset.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/Utils/PrintHost.hpp"
#include "slic3r/Utils/Http.hpp"

#include <boost/filesystem/path.hpp>

namespace py = pybind11;
using namespace Slic3r;

namespace pyslic3r {
namespace {

// The host fields (host_type / print_host / printhost_apikey) live in the
// printer preset config — exactly what the Physical Printer dialog edits.
//
// The embedded script runs ON the wx main thread (PyHost dispatches it via
// CallAfter), so these reads are already main-thread-affine; the blocking curl
// underneath test()/upload()/status is a sub-second op for a normal upload. We
// drop the GIL around the network I/O so a long transfer can't wedge Python.
DynamicPrintConfig &host_config()
{
    return GUI::wxGetApp().preset_bundle->printers.get_edited_preset().config;
}

std::unique_ptr<PrintHost> make_host()
{
    std::unique_ptr<PrintHost> host(PrintHost::get_print_host(&host_config()));
    if (host == nullptr)
        throw std::runtime_error(
            "no print host configured — call device.configure(host_type, url, apikey) first");
    return host;
}

struct PyHostDevice
{
    // Mirror the Physical Printer dialog: stamp host_type / URL / api-key onto
    // the edited printer preset. host_type is the serialized enum string
    // ("octoprint", "moonraker", "prusalink", ...).
    void configure(const std::string &host_type, const std::string &url, const std::string &apikey)
    {
        DynamicPrintConfig &cfg = host_config();
        ConfigSubstitutionContext ctx{ ForwardCompatibilitySubstitutionRule::Disable };
        cfg.set_deserialize("host_type", host_type, ctx);   // enum string -> coEnum
        cfg.set_key_value("print_host", new ConfigOptionString(url));
        cfg.set_key_value("printhost_apikey", new ConfigOptionString(apikey));
    }

    // The resolved host label (mirrors what the dialog shows).
    std::string host() { return make_host()->get_name(); }

    // "Test" button — connectivity / auth check.
    bool test()
    {
        auto host = make_host();
        wxString msg;
        py::gil_scoped_release nogil;
        return host->test(msg);
    }

    // "Send" (start=False) / "Send and print" (start=True). Uploads a sliced
    // G-code through the same PrintHost::upload the GUI print-host queue uses.
    // Returns "" on success, else the host's error string.
    std::string send(const std::string &gcode_path, bool start)
    {
        namespace fs = boost::filesystem;
        auto host = make_host();

        PrintHostUpload up;
        up.source_path = fs::path(gcode_path);
        up.upload_path = fs::path(gcode_path).filename();
        up.post_action = start ? PrintHostPostUploadAction::StartPrint
                               : PrintHostPostUploadAction::None;

        std::string err;
        bool ok = false;
        {
            py::gil_scoped_release nogil;
            ok = host->upload(
                std::move(up),
                [](Http::Progress, bool & /*cancel*/) {},               // progress
                [&err](wxString e) { err = e.ToStdString(); },          // error
                [](wxString /*tag*/, wxString /*status*/) {});          // info
        }
        if (ok)
            return {};
        return err.empty() ? std::string("upload failed") : err;
    }

    // Read-back (the bidirectional invariant): the raw job status body the host
    // reports. Returned as a string so the caller parses it — keeps this TU free
    // of a JSON dependency. Uses Orca's own Http primitive (the GUI's transport).
    std::string status_raw(const std::string &path)
    {
        DynamicPrintConfig &cfg = host_config();
        std::string base   = cfg.opt_string("print_host");
        std::string apikey = cfg.opt_string("printhost_apikey");
        if (base.empty())
            throw std::runtime_error("no print host configured");
        if (base.back() != '/')
            base += '/';
        const std::string url = base + path;

        std::string body, error;
        {
            py::gil_scoped_release nogil;
            auto http = Http::get(url);
            if (!apikey.empty())
                http.header("X-Api-Key", apikey);
            http.on_complete([&](std::string b, unsigned /*status*/) { body = std::move(b); })
                .on_error([&](std::string b, std::string e, unsigned /*status*/) {
                    error = e.empty() ? std::move(b) : std::move(e);
                })
                .perform_sync();
        }
        if (body.empty() && !error.empty())
            throw std::runtime_error("status query failed: " + error);
        return body;
    }
};

} // namespace

void register_device(py::module_ &m)
{
    py::class_<PyHostDevice>(m, "Device")
        .def("configure", &PyHostDevice::configure,
             py::arg("host_type"), py::arg("url"), py::arg("apikey") = std::string(),
             "Point the device at a print host (Physical Printer settings): "
             "host_type is 'octoprint' / 'moonraker' / 'prusalink' / ...")
        .def_property_readonly("host", &PyHostDevice::host)
        .def("test", &PyHostDevice::test, "Connectivity/auth check (the Test button).")
        .def("send", &PyHostDevice::send,
             py::arg("gcode_path"), py::arg("start") = false,
             "Upload a sliced G-code to the host; start=True also begins the print. "
             "Returns '' on success or the error string.")
        .def("status", &PyHostDevice::status_raw,
             py::arg("path") = std::string("api/job"),
             "Raw status body from the host (default OctoPrint 'api/job'); caller parses.");

    m.attr("device") = PyHostDevice{};
}

} // namespace pyslic3r
