// Minimal stand-ins for the parts of Rack's application layer that Module::toJson()/fromJson() and friends touch, so the *real* Rack
// engine::Module / ParamQuantity / jansson code can run in a headless unit test. Only used by tests/test_module.cpp.
#include <rack.hpp>
#define PRIVATE
#include <midiloopback.hpp>
#include <patch.hpp>
#undef PRIVATE
#include <chrono>

namespace rack {

thread_local Context* g_ctx = nullptr;
Context* contextGet() { return g_ctx; }
void contextSet(Context* c) { g_ctx = c; }

namespace engine {
float Engine::getParamValue(Module* m, int id) { return m->params[id].value; }
void Engine::setParamValue(Module* m, int id, float v) { m->params[id].value = v; }
float Engine::getParamSmoothValue(Module* m, int id) { return m->params[id].value; }
void Engine::setParamSmoothValue(Module* m, int id, float v) { m->params[id].value = v; }
Engine::~Engine() {}
} // namespace engine

namespace history { State::~State() {} }
namespace midiloopback { Context::~Context() {} }
namespace patch { Manager::~Manager() {} }
namespace window { Window::~Window() {} }
namespace plugin { Model* g_testModel = nullptr; Model* modelFromJson(json_t*) { return g_testModel; } Plugin::~Plugin() {} }
namespace asset { std::string system(std::string f) { return f; } }
namespace settings { bool cpuMeter = false; std::string language; }
namespace system {
std::string join(const std::string& a, const std::string& b) { return b.empty() ? a : a + "/" + b; }
std::string getStem(const std::string& p) { return p; }
std::string getExtension(const std::string& p) { return p; }
double getTime() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double getUnixTime() { return 0.0; }
bool createDirectories(const std::string&) { return true; }
std::vector<std::string> getEntries(const std::string&, int) { return std::vector<std::string>(); }
} // namespace system

// The tests never need a real application context, only an `APP->engine` pointer for Module::fromJson().
void headlessInstallContext() {
	static char ctxBuf[sizeof(Context)];
	static char engBuf[sizeof(engine::Engine)];
	std::memset(ctxBuf, 0, sizeof ctxBuf);
	std::memset(engBuf, 0, sizeof engBuf);
	Context* c = reinterpret_cast<Context*>(ctxBuf);
	c->engine = reinterpret_cast<engine::Engine*>(engBuf);
	contextSet(c);
}

} // namespace rack
