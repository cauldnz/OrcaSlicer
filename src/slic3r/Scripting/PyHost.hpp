#ifndef slic3r_Scripting_PyHost_hpp_
#define slic3r_Scripting_PyHost_hpp_

// pyslic3r embedded Python runtime — in-process host.
//
// M0 scope: interpreter lifecycle + the cross-thread marshalling primitive.
// Two orthogonal locks are in play and must never be held while blocking on
// the other: the wx main thread (all Plater/Model/GUI objects) and the GIL
// (all Python execution). The rules encoded here:
//   - Python only ever runs on the wx main thread.
//   - The main thread parks the GIL released; every Python call re-acquires
//     it for exactly the duration of the call.
//   - Any other thread reaches Python (and the GUI object model) exclusively
//     through run_on_main_blocking().

#include <functional>
#include <string>

namespace pyslic3r {

// Lifecycle. Main thread only. Both idempotent.
void host_init();      // start embedded CPython, register the pyslic3r module
void host_shutdown();  // finalize the interpreter (reacquires the parked GIL first)
bool host_initialized();

// Milliseconds spent inside Py_Initialize + module import at host_init time.
double interpreter_init_ms();

// The marshalling primitive — callable from ANY thread.
// Posts `fn` onto the wx main loop, blocks the calling thread until it has
// run, and rethrows on the caller any exception `fn` threw. Called on the
// main thread it just runs `fn` inline (no deadlock by self-wait).
void run_on_main_blocking(std::function<void()> fn);

// Convenience built on the primitive: evaluate a Python expression on the
// main thread — GIL acquired only around the evaluation — and return
// str(result). Safe from any thread.
std::string py_eval_str(const std::string &expr);

// M0 self-test: no-op unless env PYSLIC3R_M0_TEST=1. Called by host_init.
void maybe_start_m0_selftest();

// Script runner (SPEC §4 "scripts run once"): if env PYSLIC3R_SCRIPT names a
// .py file, run it once on the wx main thread after the app is up, driving the
// live document through pyslic3r. Leaves the app running by default; set
// PYSLIC3R_SCRIPT_EXIT=1 for batch use (finalize + exit after the script,
// exit code reflects success). No-op if PYSLIC3R_SCRIPT is unset. Called by
// host_init.
// Agent bridge (M5): if env PYSLIC3R_BRIDGE_PORT is set, start a loopback
// TCP JSON-RPC listener that relays to pyslic3r._handle_request on the wx
// main thread. Dispatcher from PYSLIC3R_DISPATCH_FILE (else a minimal ping).
// No-op if the env is unset. Called by host_init. Loopback-only.
void maybe_start_bridge();

void maybe_run_user_script();

} // namespace pyslic3r

#endif // slic3r_Scripting_PyHost_hpp_
