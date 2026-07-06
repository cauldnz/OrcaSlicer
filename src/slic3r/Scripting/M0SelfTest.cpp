// M0 self-test: prove the embed + marshal round-trip, then exit the app.
//
// Armed only when PYSLIC3R_M0_TEST=1. Runs after host_init (end of
// GUI_App::post_init, i.e. after command-line input files are loaded):
//   path (a) — main thread: import pyslic3r, read version + object_count.
//   path (b) — background std::thread: the same two reads, N iterations,
//              every call marshalled through run_on_main_blocking().
// Writes PASS/FAIL + timings to $PYSLIC3R_M0_RESULT (default M0-RESULT.md),
// echoes "PYSLIC3R_M0: PASS|FAIL" on stdout, then closes the app.

#include "PyHost.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/embed.h>

#include <wx/app.h>
#include <wx/dialog.h>
#include <wx/timer.h>
#include <wx/toplevel.h>
#include <wx/version.h>

#include <boost/log/trivial.hpp>

#include "libslic3r/libslic3r.h"
#include "libslic3r/Format/bbs_3mf.hpp"   // LoadStrategy
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/MainFrame.hpp"
#include "slic3r/GUI/Plater.hpp"

namespace py = pybind11;

namespace pyslic3r {

namespace {

struct M0Results
{
    std::vector<std::string> checks;    // "PASS|FAIL — description"
    bool        failed = false;
    std::string version;                // app.version, from path (a)
    std::string py_version;             // Py_GetVersion(), captured before finalize
    long        object_count = -1;      // from path (a)
    double      path_a_ms = 0.0;
    double      path_b_total_ms = 0.0;
    int         path_b_iters_done = 0;
    int         path_b_iters_target = 0;
    double      finalize_ms = 0.0;      // explicit host_shutdown() duration
};

M0Results   s_results;
std::thread s_worker;

// Headless modal auto-dismisser. Run offscreen, cloud/device operations can
// pop informational MessageDialogs (e.g. GUI_App::process_network_msg emits
// "update_studio"/"wait_info" because a from-source build isn't a recognised
// official Studio version) — a ShowModal with no human to click it wedges the
// app. This repeating timer fires even inside a nested ShowModal loop and ends
// any stray modal, so automated/headless runs never hang on one. (A real
// deployment would instead capture such messages; here we just dismiss them.)
class ModalDismissTimer : public wxTimer
{
public:
    void Notify() override
    {
        for (wxWindow *w : wxTopLevelWindows)
            if (auto *dlg = dynamic_cast<wxDialog *>(w))
                if (dlg->IsModal()) {
                    BOOST_LOG_TRIVIAL(warning) << "pyslic3r: auto-dismissing headless modal";
                    dlg->EndModal(wxID_OK);
                }
    }
};
// Heap-allocated at Start() time (NOT a static global — a wxTimer constructed
// before wxApp exists doesn't register with the event loop).
ModalDismissTimer *s_modal_dismisser = nullptr;

void start_modal_dismisser()
{
    if (s_modal_dismisser == nullptr)
        s_modal_dismisser = new ModalDismissTimer();
    s_modal_dismisser->Start(200);
}

void check(bool ok, const std::string &what)
{
    s_results.checks.push_back(std::string(ok ? "PASS" : "FAIL") + " — " + what);
    if (!ok)
        s_results.failed = true;
    BOOST_LOG_TRIVIAL(info) << "pyslic3r M0 check: " << (ok ? "PASS" : "FAIL") << " — " << what;
}

std::string env_or(const char *name, const std::string &fallback)
{
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

void write_result_and_quit()
{
    const bool pass = !s_results.failed;
    const std::string path = env_or("PYSLIC3R_M0_RESULT", "M0-RESULT.md");

    const std::string label = env_or("PYSLIC3R_TEST_LABEL", "M0");

    std::ostringstream md;
    md << "# " << label << "-RESULT — pyslic3r self-test\n\n";
    md << "RESULT: " << (pass ? "PASS" : "FAIL") << "\n\n";

    std::time_t now = std::time(nullptr);
    char timebuf[64];
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S UTC", std::gmtime(&now));
    md << "- Run: " << timebuf << "\n";
    md << "- Fixture object_count read: " << s_results.object_count
       << " (expected " << env_or("PYSLIC3R_M0_EXPECT_OBJECTS", "2") << ")\n";
    md << "- app.version read: \"" << s_results.version << "\"\n\n";

    md << "## Checks\n\n";
    for (const auto &c : s_results.checks)
        md << "- " << c << "\n";

    md << "\n## Timings\n\n";
    md << "| what | value |\n|---|---|\n";
    md << "| interpreter init (Py_Initialize + import pyslic3r) | "
       << interpreter_init_ms() << " ms |\n";
    md << "| path (a): main-thread import + 2 reads | " << s_results.path_a_ms << " ms |\n";
    md << "| path (b): " << s_results.path_b_iters_done << "/" << s_results.path_b_iters_target
       << " iterations x 2 marshalled reads | " << s_results.path_b_total_ms << " ms total";
    if (s_results.path_b_iters_done > 0)
        md << ", " << (s_results.path_b_total_ms / (2.0 * s_results.path_b_iters_done))
           << " ms/round-trip avg";
    md << " |\n";

    md << "| interpreter finalize (explicit host_shutdown) | "
       << s_results.finalize_ms << " ms |\n";

    md << "\n## Environment\n\n";
    md << "- BambuStudio: " << SLIC3R_VERSION << "\n";
    md << "- Python: " << s_results.py_version << "\n";
    md << "- pybind11: " << PYBIND11_VERSION_MAJOR << "." << PYBIND11_VERSION_MINOR
       << "." << PYBIND11_VERSION_PATCH << "\n";
    md << "- wxWidgets: " << wxVERSION_NUM_DOT_STRING << "\n";
#if defined(__VERSION__)
    md << "- Compiler: " << __VERSION__ << "\n";
#endif

    std::ofstream out(path, std::ios::trunc);
    out << md.str();
    out.close();

    // Machine-readable line for the harness, independent of the file.
    std::printf("PYSLIC3R_%s: %s\n", label.c_str(), pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    BOOST_LOG_TRIVIAL(info) << "pyslic3r M0: result written to " << path
                            << " — " << (pass ? "PASS" : "FAIL");
}

// Terminal for EVERY path (pass or fail): finalize the interpreter, write the
// result, and exit. Called on the main thread. Never returns.
void finalize_and_exit()
{
    // The clean-shutdown gate is specifically about the *interpreter*:
    // "interpreter finalized, no hang on exit." Finalize it explicitly here,
    // on the main thread — the thing the spike de-risks — independently of
    // BambuStudio's full GUI teardown (which, run headless, hits a pre-existing
    // wxWebView::RunScript segfault in the first-run wizard's webview teardown,
    // unrelated to pyslic3r; see M0-RESULT "What fought back").
    s_results.py_version = Py_GetVersion();
    bool finalize_ok = true;
    try {
        const auto t0 = std::chrono::steady_clock::now();
        pyslic3r::host_shutdown();
        const auto t1 = std::chrono::steady_clock::now();
        s_results.finalize_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        finalize_ok = !pyslic3r::host_initialized();
    } catch (const std::exception &e) {
        finalize_ok = false;
        check(false, std::string("interpreter finalize threw: ") + e.what());
    }
    check(finalize_ok, "interpreter finalized cleanly (explicit host_shutdown, no hang/crash)");

    write_result_and_quit();

    // Exit without dragging the process through BambuStudio's flaky headless
    // GUI/webview teardown. Flush first.
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(s_results.failed ? 1 : 0);
}

void finalize_on_main()
{
    if (s_worker.joinable())
        s_worker.join();    // worker has already signalled; this is immediate
    finalize_and_exit();
}

void run_path_b_worker()
{
    const int  iters    = std::atoi(env_or("PYSLIC3R_M0_ITERS", "1000").c_str());
    s_results.path_b_iters_target = iters;
    const auto expected_version = s_results.version;
    const auto expected_count   = std::to_string(s_results.object_count);

    bool ok = true;
    std::string what_failed;
    const auto t0 = std::chrono::steady_clock::now();
    int i = 0;
    try {
        for (; i < iters; ++i) {
            const std::string v = py_eval_str("pyslic3r.app.version");
            const std::string c = py_eval_str("pyslic3r.app.active_document.object_count");
            if (v != expected_version || c != expected_count) {
                ok = false;
                what_failed = "iteration " + std::to_string(i) + " returned (" + v + ", " + c + ")";
                break;
            }
        }
    } catch (const std::exception &e) {
        ok = false;
        what_failed = "exception at iteration " + std::to_string(i) + ": " + e.what();
    }
    const auto t1 = std::chrono::steady_clock::now();

    s_results.path_b_total_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s_results.path_b_iters_done = i;
    check(ok && i == iters, "path (b): background thread, " + std::to_string(iters) +
          " marshalled round-trips, values stable" +
          (ok ? "" : " [" + what_failed + "]"));

    // Fire-and-forget: hand the finale to the main loop, then let this
    // thread exit. finalize_on_main joins us before touching results.
    wxTheApp->CallAfter([]() { finalize_on_main(); });
}

// Split a ';'-separated path list.
std::vector<std::string> split_paths(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ';') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Load the fixtures through Plater exactly as the GUI's "Import model" does —
// LoadModel geometry import, no LoadConfig, Silence set — so no config-load
// modal fires under a headless display. This is the UI-parity load path;
// launching with the file on argv instead routes through the interactive
// project-open path (load_project) whose modals wedge offscreen.
void load_fixtures_on_main()
{
    const std::string spec = env_or("PYSLIC3R_M0_FIXTURES", "");
    if (spec.empty()) {
        check(false, "setup: PYSLIC3R_M0_FIXTURES not set");
        return;
    }
    std::vector<std::string> paths = split_paths(spec);
    auto *plater = Slic3r::GUI::wxGetApp().plater();
    if (plater == nullptr) {
        check(false, "setup: no Plater to load fixtures into");
        return;
    }
    try {
        const auto strategy = Slic3r::LoadStrategy::LoadModel |
                              Slic3r::LoadStrategy::AddDefaultInstances |
                              Slic3r::LoadStrategy::Silence;
        plater->load_files(paths, strategy, /*ask_multi=*/false);
        check(true, "setup: imported " + std::to_string(paths.size()) + " fixture(s)");
    } catch (const std::exception &e) {
        check(false, std::string("setup: fixture import threw: ") + e.what());
    }
}

void run_selftest_on_main()
{
    BOOST_LOG_TRIVIAL(info) << "pyslic3r M0 self-test starting";
    const long expected_count = std::atol(env_or("PYSLIC3R_M0_EXPECT_OBJECTS", "2").c_str());

    // Dismiss any stray headless modal (see ModalDismissTimer) for the whole run.
    start_modal_dismisser();

    load_fixtures_on_main();
    if (s_results.failed) { finalize_and_exit(); }

    // ---- path (a): straight main-thread use -------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    try {
        py::gil_scoped_acquire gil;
        py::object ns = py::module_::import("__main__").attr("__dict__");
        py::exec("import pyslic3r", ns);
        check(true, "path (a): import pyslic3r");

        s_results.version = py::eval("pyslic3r.app.version", ns).cast<std::string>();
        check(!s_results.version.empty(),
              "path (a): app.version non-empty (\"" + s_results.version + "\")");

        py::object doc = py::eval("pyslic3r.app.active_document", ns);
        check(!doc.is_none(), "path (a): active_document present");

        s_results.object_count = doc.is_none()
            ? -1 : py::eval("pyslic3r.app.active_document.object_count", ns).cast<long>();
        check(s_results.object_count == expected_count,
              "path (a): object_count == " + std::to_string(expected_count) +
              " (got " + std::to_string(s_results.object_count) + ")");
    } catch (const std::exception &e) {
        check(false, std::string("path (a): exception: ") + e.what());
    }
    const auto t1 = std::chrono::steady_clock::now();
    s_results.path_a_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- M1: read-only object model, verified from a Python script --------
    // The surface under test is Python-visible, so the assertions are a Python
    // file (PYSLIC3R_M1_SCRIPT) that walks app→document→model→objects→volumes,
    // plates, and config. Raising there fails the milestone.
    if (std::getenv("PYSLIC3R_M1_TEST") != nullptr && !s_results.failed) {
        const std::string script = env_or("PYSLIC3R_M1_SCRIPT", "");
        try {
            py::gil_scoped_acquire gil;
            py::object ns = py::module_::import("__main__").attr("__dict__");
            if (script.empty())
                throw std::runtime_error("PYSLIC3R_M1_SCRIPT not set");
            py::eval_file(script, ns);
            check(true, "M1: read-only object model asserts passed (" + script + ")");
        } catch (const std::exception &e) {
            check(false, std::string("M1: object model asserts failed: ") + e.what());
        }
    }

    // ---- M2: mutation, verified via read-back -----------------------------
    if (std::getenv("PYSLIC3R_M2_TEST") != nullptr && !s_results.failed) {
        const std::string script = env_or("PYSLIC3R_M2_SCRIPT", "");
        try {
            py::gil_scoped_acquire gil;
            py::object ns = py::module_::import("__main__").attr("__dict__");
            if (script.empty())
                throw std::runtime_error("PYSLIC3R_M2_SCRIPT not set");
            py::eval_file(script, ns);
            check(true, "M2: mutation asserts passed (" + script + ")");
        } catch (const std::exception &e) {
            check(false, std::string("M2: mutation asserts failed: ") + e.what());
        }
    }

    // ---- M3: slicing, verified via read-back ------------------------------
    if (std::getenv("PYSLIC3R_M3_TEST") != nullptr && !s_results.failed) {
        const std::string script = env_or("PYSLIC3R_M3_SCRIPT", "");
        try {
            py::gil_scoped_acquire gil;
            py::object ns = py::module_::import("__main__").attr("__dict__");
            if (script.empty())
                throw std::runtime_error("PYSLIC3R_M3_SCRIPT not set");
            py::eval_file(script, ns);
            check(true, "M3: slicing asserts passed (" + script + ")");
        } catch (const std::exception &e) {
            check(false, std::string("M3: slicing asserts failed: ") + e.what());
        }
    }

    // ---- M4: cloud device plane (read-only) -------------------------------
    if (std::getenv("PYSLIC3R_M4_TEST") != nullptr && !s_results.failed) {
        const std::string script = env_or("PYSLIC3R_M4_SCRIPT", "");
        try {
            py::gil_scoped_acquire gil;
            py::object ns = py::module_::import("__main__").attr("__dict__");
            if (script.empty())
                throw std::runtime_error("PYSLIC3R_M4_SCRIPT not set");
            py::eval_file(script, ns);
            check(true, "M4: device plane asserts passed (" + script + ")");
        } catch (const std::exception &e) {
            check(false, std::string("M4: device plane asserts failed: ") + e.what());
        }
    }

    // ---- SEND: slice + camera_url + DRY-RUN send (no dispatch) -------------
    if (std::getenv("PYSLIC3R_SEND_TEST") != nullptr && !s_results.failed) {
        const std::string script = env_or("PYSLIC3R_SEND_SCRIPT", "");
        try {
            py::gil_scoped_acquire gil;
            py::object ns = py::module_::import("__main__").attr("__dict__");
            if (script.empty())
                throw std::runtime_error("PYSLIC3R_SEND_SCRIPT not set");
            py::eval_file(script, ns);
            check(true, "SEND: camera_url + dry-run send asserts passed (" + script + ")");
        } catch (const std::exception &e) {
            check(false, std::string("SEND: asserts failed: ") + e.what());
        }
    }

    if (s_results.failed) {
        // Path (b) would only hang on a broken foundation; report honestly
        // and exit cleanly (do not leave the app running).
        finalize_and_exit();
    }

    // ---- path (b): background thread through the marshalling primitive ----
    s_worker = std::thread(run_path_b_worker);
}

} // anonymous namespace

void maybe_start_m0_selftest()
{
    const char *flag = std::getenv("PYSLIC3R_M0_TEST");
    if (flag == nullptr || std::string(flag) != "1")
        return;
    // Defer one event-loop turn so post_init (and anything it queued, e.g.
    // config-wizard checks) settles before the test reads the document.
    wxTheApp->CallAfter([]() { run_selftest_on_main(); });
}

} // namespace pyslic3r
