#include "PyHost.hpp"
#include "PyBindings.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/embed.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#include <wx/app.h>
#include <wx/thread.h>
#include <wx/timer.h>
#include <wx/dialog.h>
#include <wx/toplevel.h>

#include "libslic3r/libslic3r.h"
#include "libslic3r/Model.hpp"
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/Plater.hpp"

namespace py = pybind11;

namespace pyslic3r {

namespace {

// Auto-dismiss stray headless modals (arrange/slice progress dialogs) during a
// script run; without this the script path blocks headless (the M0 harness has
// its own dismisser).
class ScriptModalDismisser : public wxTimer {
public:
    void Notify() override {
        for (wxWindow *w : wxTopLevelWindows)
            if (auto *dlg = dynamic_cast<wxDialog *>(w))
                if (dlg->IsModal()) dlg->EndModal(wxID_OK);
    }
};
ScriptModalDismisser *s_script_dismisser = nullptr;
void start_script_dismisser() {
    if (s_script_dismisser == nullptr) s_script_dismisser = new ScriptModalDismisser();
    if (!s_script_dismisser->IsRunning()) s_script_dismisser->Start(200);
}

bool   s_initialized = false;
double s_init_ms     = 0.0;

// While the app is idle the main thread does not hold the GIL; it is parked
// here released so worker-marshalled Python (which always executes on the
// main thread) can acquire it around each call.
std::unique_ptr<py::gil_scoped_release> s_parked_gil;

void ensure_main_thread(const char *what)
{
    if (!wxThread::IsMain())
        throw std::runtime_error(std::string(what) +
            " must be accessed on the wx main thread; use the marshalling primitive");
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Embedded module. Thin composition point: the read-only object-model surface
// lives in PyObjectModel.cpp and is attached via register_object_model(). This
// TU is referenced by GUI_App, so the static module initializer can't be
// dropped by the linker.
// ---------------------------------------------------------------------------

static void bridge_emit(const std::string &line);

PYBIND11_EMBEDDED_MODULE(pyslic3r, m)
{
    m.doc() = "pyslic3r — embedded object model + cloud device plane over the running app";
    register_object_model(m);
#ifndef PYSLIC3R_NO_DEVICE
    register_device(m);
    m.def("_emit_raw", [](const std::string &s) { bridge_emit(s); },
          "enqueue a JSON-RPC notification line to the active bridge connection");
#endif
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool host_initialized() { return s_initialized; }
double interpreter_init_ms() { return s_init_ms; }

void host_init()
{
    ensure_main_thread("pyslic3r::host_init");
    if (s_initialized)
        return;

    const auto t0 = std::chrono::steady_clock::now();
    py::initialize_interpreter();
    {
        // Import once so the module's own init cost lands in s_init_ms and
        // later importers hit sys.modules.
        py::module_::import("pyslic3r");
    }
    const auto t1 = std::chrono::steady_clock::now();
    s_init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Park the GIL released; every Python call from here on acquires it
    // explicitly for the duration of the call only.
    s_parked_gil = std::make_unique<py::gil_scoped_release>();
    s_initialized = true;

    maybe_start_m0_selftest();
    maybe_start_bridge();
    maybe_run_user_script();
}

void host_shutdown()
{
    if (!s_initialized)
        return;
    ensure_main_thread("pyslic3r::host_shutdown");

    s_parked_gil.reset();       // reacquire the GIL for finalization
    py::finalize_interpreter();
    s_initialized = false;
}

// ---------------------------------------------------------------------------
// Marshalling primitive
// ---------------------------------------------------------------------------

void run_on_main_blocking(std::function<void()> fn)
{
    if (wxThread::IsMain()) {
        fn();
        return;
    }

    auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
    std::future<void> done = task->get_future();

    // wxEvtHandler::CallAfter is documented thread-safe: it queues an event
    // to the main loop. The calling thread then blocks on the future — it
    // holds neither the GIL nor any wx object while waiting, so the main
    // thread is free to run the closure.
    wxTheApp->CallAfter([task]() { (*task)(); });

    done.get();     // rethrows any exception from fn on the calling thread
}

std::string py_eval_str(const std::string &expr)
{
    std::string out;
    run_on_main_blocking([&]() {
        py::gil_scoped_acquire gil;     // GIL held only around the Python call
        py::object ns  = py::module_::import("__main__").attr("__dict__");
        py::object res = py::eval(expr, ns);
        out = py::str(res).cast<std::string>();
    });
    return out;
}

// ---------------------------------------------------------------------------
// Agent bridge (M5) — loopback TCP + JSON-RPC 2.0 over the façade.
// ---------------------------------------------------------------------------

static const char *BRIDGE_MINIMAL_PY = R"PY(
import json as _json, pyslic3r as _ps
def _handle_request(s):
    try: req = _json.loads(s)
    except Exception: return _json.dumps({"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":"parse error"}})
    rid = req.get("id")
    try:
        if req.get("method") == "ping":
            r = {"app": _ps.app.name, "version": _ps.app.version, "ok": True, "minimal": True}
        else:
            raise RuntimeError("dispatcher not loaded; set PYSLIC3R_DISPATCH_FILE")
        return _json.dumps({"jsonrpc":"2.0","id":rid,"result":r})
    except Exception as e:
        return _json.dumps({"jsonrpc":"2.0","id":rid,"error":{"code":-32000,"message":str(e)}})
_ps._handle_request = _handle_request
)PY";

#ifndef _WIN32
// Outbound channel: all writes (responses + async notifications) serialize through
// one writer thread draining a queue, so the app can PUSH notifications to the
// connected client (bridge_emit), not just answer requests. Single connection.
static std::mutex               g_out_mtx;
static std::condition_variable  g_out_cv;
static std::deque<std::string>  g_outbox;
static int                      g_conn_fd = -1;
static bool                     g_conn_open = false;

static void outbox_push(std::string line)
{
    { std::lock_guard<std::mutex> lk(g_out_mtx); g_outbox.push_back(std::move(line)); }
    g_out_cv.notify_one();
}

static void bridge_emit(const std::string &line)   // Python/main-thread callable
{
    if (g_conn_open) outbox_push(line);
}

static void bridge_writer_loop()
{
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(g_out_mtx);
            g_out_cv.wait(lk, [] { return !g_outbox.empty() || !g_conn_open; });
            if (g_outbox.empty() && !g_conn_open) return;
            line = std::move(g_outbox.front());
            g_outbox.pop_front();
        }
        line.push_back('\n');
        if (g_conn_fd >= 0) {
            ssize_t off = 0, len = (ssize_t) line.size();
            while (off < len) {
                ssize_t w = ::send(g_conn_fd, line.data() + off, (size_t)(len - off), MSG_NOSIGNAL);
                if (w <= 0) break;
                off += w;
            }
        }
    }
}

static void bridge_serve(int port)
{
    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { std::perror("pyslic3r bridge: socket"); return; }
    int opt = 1; ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); addr.sin_port = htons((uint16_t) port);
    if (::bind(srv, (sockaddr *) &addr, sizeof(addr)) < 0) { std::perror("pyslic3r bridge: bind"); ::close(srv); return; }
    ::listen(srv, 4);
    std::fprintf(stderr, "pyslic3r bridge: listening on 127.0.0.1:%d\n", port); std::fflush(stderr);
    for (;;) {
        int c = ::accept(srv, nullptr, nullptr);
        if (c < 0) continue;
        { std::lock_guard<std::mutex> lk(g_out_mtx); g_outbox.clear(); }
        g_conn_fd = c; g_conn_open = true;
        std::thread writer(bridge_writer_loop);
        std::string buf; char tmp[8192];
        for (;;) {
            ssize_t n = ::recv(c, tmp, sizeof(tmp), 0);
            if (n <= 0) break;
            buf.append(tmp, (size_t) n);
            size_t pos;
            while ((pos = buf.find('\n')) != std::string::npos) {
                std::string line = buf.substr(0, pos); buf.erase(0, pos + 1);
                if (line.empty()) continue;
                std::string resp;
                try {
                    run_on_main_blocking([&]() {
                        py::gil_scoped_acquire gil;
                        py::object h = py::module_::import("pyslic3r").attr("_handle_request");
                        resp = h(line).cast<std::string>();
                    });
                } catch (const std::exception &e) {
                    resp = std::string("{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32001,\"message\":\"bridge: ")
                         + e.what() + "\"}}";
                }
                outbox_push(std::move(resp));   // response rides the writer thread too
            }
        }
        g_conn_open = false;
        g_out_cv.notify_all();
        writer.join();
        g_conn_fd = -1;
        ::close(c);
    }
}
#else
static void bridge_emit(const std::string &) {}   // Winsock port deferred
#endif

void maybe_start_bridge()
{
    const char *ps = std::getenv("PYSLIC3R_BRIDGE_PORT");
    if (ps == nullptr || *ps == '\0') return;
    int port = std::atoi(ps);
    if (port <= 0) return;
#ifdef _WIN32
    std::fprintf(stderr, "pyslic3r bridge: not yet ported to Windows (Winsock)\n");
#else
    {
        py::gil_scoped_acquire gil;
        py::object nsdict = py::module_::import("pyslic3r").attr("__dict__");
        const char *df = std::getenv("PYSLIC3R_DISPATCH_FILE");
        try {
            if (df != nullptr && *df != '\0') py::eval_file(df, nsdict);
            else py::exec(BRIDGE_MINIMAL_PY, nsdict);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "pyslic3r bridge: dispatcher load failed: %s\n", e.what());
            py::exec(BRIDGE_MINIMAL_PY, nsdict);
        }
    }
    std::thread(bridge_serve, port).detach();
#endif
}

// ---------------------------------------------------------------------------
// Script runner — the general "run a .py against the live app" entry point.
// ---------------------------------------------------------------------------

void maybe_run_user_script()
{
    const char *path = std::getenv("PYSLIC3R_SCRIPT");
    if (path == nullptr || *path == '\0')
        return;

    const char *exit_env = std::getenv("PYSLIC3R_SCRIPT_EXIT");
    const bool  exit_after = (exit_env != nullptr && std::strcmp(exit_env, "1") == 0);
    const std::string script = path;

    // Defer one event-loop turn so post_init has settled and the document
    // exists, then run the script ON the main thread — pyslic3r's bindings
    // touch Plater/Model there. (A script that blocks, e.g. slice().wait(),
    // pumps the loop itself; see SliceJob.wait.)
    wxTheApp->CallAfter([script, exit_after]() {
        int rc = 0;
        start_script_dismisser();  // clear headless arrange/slice modals
        {
            py::gil_scoped_acquire gil;   // tight scope: released before any shutdown
            try {
                py::object ns = py::module_::import("__main__").attr("__dict__");
                py::eval_file(script, ns);
            } catch (const std::exception &e) {
                rc = 1;
                std::fprintf(stderr, "pyslic3r: script error in %s: %s\n",
                             script.c_str(), e.what());
                std::fflush(stderr);
            }
        }
        std::printf("PYSLIC3R_SCRIPT: %s (%s)\n", rc == 0 ? "OK" : "ERROR", script.c_str());
        std::fflush(stdout);

        if (exit_after) {
            // Batch mode: finalize the interpreter and exit without dragging
            // the process through BambuStudio's flaky headless GUI teardown
            // (same rationale as the self-test). GIL scope above has closed,
            // so host_shutdown's parked-GIL reacquire is clean.
            host_shutdown();
            std::fflush(stdout);
            std::fflush(stderr);
            std::_Exit(rc);
        }
        // Otherwise leave the app running — the script has done its thing
        // against the live session.
    });
}

} // namespace pyslic3r
