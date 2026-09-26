"""TouchDesigner link: the data bridge (Myrmex -> TD), the heartbeat (TD -> Myrmex), takes, and the TD network
builder run against a stand-in of TouchDesigner's Python API (the real one only exists inside TD)."""
import importlib.util
import math
import os
import socket
import sys
import types

import numpy as np
import pytest

from myrmex.bus.osc import OscMessage, decode
from myrmex.realtime.protocol import CameraState
from myrmex.realtime.session import LiveConfig, LiveSession
from myrmex.realtime.touch import CHANNELS, FX, PRESETS, TouchBridge, project

HERE = os.path.dirname(os.path.abspath(__file__))
TD_SCRIPT = os.path.join(HERE, "..", "touchdesigner", "myrmex_td.py")


class Sink:
    sent = 0

    def send_raw(self, b):
        pass

    def close(self):
        pass


def _udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.setblocking(False)
    return s, s.getsockname()[1]


def _drain(s):
    out = []
    while True:
        try:
            out.append(s.recv(65536))
        except BlockingIOError:
            return out


# ------------------------------------------------------------------------------------------ the bridge
def test_channels_are_unique_and_the_packet_is_what_td_reads():
    assert len(CHANNELS) == len(set(CHANNELS)) and len(CHANNELS) < 120
    b = TouchBridge({"preset": "Dream"})
    v = b.update(1 / 60)
    msgs = decode(b.packet(v))
    assert [m.address for m in msgs] == ["/myrmex/" + c for c in CHANNELS]      # strip 1 segment -> the names
    assert all(len(m.args) == 1 and isinstance(m.args[0], float) for m in msgs)
    d = dict(zip(CHANNELS, v))
    assert d["c_bloom"] == pytest.approx(PRESETS["Dream"]["bloom"]) and d["c_preset"] >= 0
    assert len(b.packet(v)) < 4096                                              # one datagram, far below the limit


def test_screen_projection_matches_the_blender_camera():
    cam = CameraState(np.array([0.0, -5.0, 2.0]), np.array([0.0, 0.0, 2.0]), 50.0, 5.0, 4.0, 1, "free")
    sx, sy, z, vis = project([0.0, 0.0, 2.0], cam)
    assert (sx, sy, vis) == (pytest.approx(0.5), pytest.approx(0.5), 1.0) and z == pytest.approx(5.0)
    rx = project([1.0, 0.0, 2.0], cam)[0]
    uy = project([0.0, 0.0, 3.0], cam)[1]
    assert rx > 0.5 and uy > 0.5                                               # right is right, up is up
    assert rx - 0.5 == pytest.approx(0.5 * 1.0 / (5.0 * 18.0 / 50.0))           # 36 mm sensor
    assert project([0.0, -8.0, 2.0], cam)[3] == 0.0                             # behind the lens


def test_events_kicks_and_the_rack_become_effect_drives():
    b = TouchBridge()
    from myrmex.creature.control import CreatureControlInput
    st = types.SimpleNamespace(beat=3.02, bpm=128.0, playing=True, beats_per_bar=4.0)
    body = types.SimpleNamespace(com=np.zeros(3), glow=0.4, arousal=0.6, surface=0.3, instability=0.2, heading=0.0,
                                 pos=np.zeros((3, 3)), morphology="CROWN", behavior="CRUISE")
    b.update(1 / 60, st=st, inp=CreatureControlInput(energy=0.8, transient=0.9, playing=True), state=body,
             events=["SHED"], controls={"td_glitch": 0.95})
    d = dict(zip(CHANNELS, b.values))
    assert d["kick"] > 0.8 and d["impact"] == pytest.approx(1.0) and d["event"] == 1
    assert d["fx_glitch"] > 0.8 and d["fx_flash"] > 0.3 and d["fx_shock"] < 0.1
    assert d["c_glitch"] == pytest.approx(0.95)                                 # a MIDI knob beats the slider
    assert d["regime"] >= 0 and d["intent"] >= 0 and d["bpm"] == 128.0
    for _ in range(90):
        b.update(1 / 60, st=st, state=body)
    d = dict(zip(CHANNELS, b.values))
    assert d["impact"] < 0.02 and d["fx_shock"] == 1.0


def test_session_streams_to_td_and_hears_it_back(tmp_path):
    rx, port = _udp()
    rt, tport = _udp()
    ses = LiveSession(LiveConfig(backend="ferro", clock="internal", out=[], record=str(tmp_path),
                                 touch={"enabled": True, "port": port, "text_port": tport, "preset": "Scanner"}),
                      start_inputs=False, sink=Sink(), now=0.0)
    now = 0.0
    pk, tx = [], []
    for i in range(240):
        now += 1 / 120
        if i == 60:                                                            # TD says hello
            ses.inputs.q.put(ses.inputs.osc._to_event(OscMessage("/myrmex/td/alive", [59.5]), 0.0)
                             if ses.inputs.osc else _alive_event())
        ses.step(now)
        pk += _drain(rx)
        tx += _drain(rt)
    assert 100 <= len(pk) <= 130                                               # 60 bundles per second
    d = {m.address.rsplit("/", 1)[-1]: m.args[0] for m in decode(pk[-1])}
    assert d["c_edges"] == pytest.approx(PRESETS["Scanner"]["edges"]) and 0.0 <= d["sx"] <= 1.0
    texts = {m.address: m.args[0] for p in tx for m in decode(p)}
    assert texts["/myrmex/text/organism"] == "ferro" and texts["/myrmex/text/preset"] == "Scanner"
    st = ses.status()["td"]
    assert st["enabled"] and st["connected"] and st["fps"] == pytest.approx(59.5)
    from myrmex.creature.take import CreatureTake
    take = CreatureTake(ses.save_take())
    assert take.resampled("td", 30).shape[1] == len(CHANNELS)                   # takes keep what TD saw
    assert [str(n) for n in take.d["td_names"]] == list(CHANNELS)


def _alive_event():
    from myrmex.bus.transport import LiveEvent
    return LiveEvent("osc", 0.0, {"address": "/myrmex/td/alive", "value": 59.5, "args": [59.5]})


def test_configure_live_changes_the_rack_and_recording():
    b = TouchBridge()
    b.configure(preset="Clean")
    assert b.cfg["fx"]["trails"] == 0.0
    b.configure(fx={"trails": 0.7, "nonsense": 3})
    assert b.cfg["fx"]["trails"] == 0.7 and "nonsense" not in b.cfg["fx"]
    b.configure(rec=True)
    assert b.rec_file.endswith(".mov")
    b.update(1 / 60)
    assert b.texts["recfile"] == b.rec_file and dict(zip(CHANNELS, b.values))["rec"] == 1.0
    b.configure(rec=False)
    assert b.rec_file == ""


# ------------------------------------------------------------------------------------------ the TD builder
class _Par:
    def __init__(self, owner, name):
        self.owner, self.name, self.val = owner, name, 0
        self.expr, self.mode, self.default = None, None, None
        self.menuNames, self.menuLabels = [], []

    def eval(self):
        return self.val

    def pulse(self, *a, **k):
        self.owner.pulses.append(self.name)


class _Pars:
    def __init__(self, owner):
        object.__setattr__(self, "_o", owner)
        object.__setattr__(self, "_p", {})

    def __getattr__(self, name):
        p = self._p.get(name)
        if p is None:
            if name[:1].isupper() and name not in self._o.custom:
                raise AttributeError(name)                                   # custom pars must exist
            p = self._p[name] = _Par(self._o, name)
            if name[:1].islower():
                _USED.setdefault(self._o.type, set()).add(name)
        return p

    def __setattr__(self, name, value):
        self.__getattr__(name).val = value


class _Page:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def _add(self, name, **_):
        assert name[:1].isupper() and name[1:] == name[1:].lower(), name        # TD's rule for custom names
        self.owner.custom.add(name)
        p = self.owner.par.__getattr__(name)
        self.owner.customPars.append(p)
        return [p]

    appendFloat = appendInt = appendToggle = appendStr = appendFolder = appendMenu = appendPulse = _add


class _Cell:
    def __init__(self, v):
        self.val = v


class _OP:
    def __init__(self, typ, name, parent=None):
        self.type, self.name, self.parent_ = typ, name, parent
        self.children, self.inputs, self.custom, self.customPars, self.pulses = {}, [], set(), [], []
        self.par = _Pars(self)
        self.nodeX = self.nodeY = 0
        self.text, self.bypass, self.viewer, self.isOpen = "", False, False, False
        self.seq = types.SimpleNamespace(vec=types.SimpleNamespace(numBlocks=1))
        self.rows, self.sent, self.chans = [], [], {}
        self._module = None

    @property
    def path(self):
        return (self.parent_.path.rstrip("/") + "/" + self.name) if self.parent_ else "/" + self.name

    def create(self, typ, name):
        o = _OP(typ, name, self)
        self.children[name] = o
        return o

    def op(self, name):
        if name == "..":
            return self.parent_
        return self.children.get(name)

    def setInputs(self, lst):
        assert all(isinstance(x, _OP) for x in lst)
        self.inputs = list(lst)

    def destroy(self):
        del self.parent_.children[self.name]

    def appendCustomPage(self, name):
        return _Page(self, name)

    # table DAT
    def clear(self):
        self.rows = []

    def appendRow(self, r):
        self.rows.append(list(r))

    def row(self, key):
        return next((r for r in self.rows if r[0] == key), None)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            r = self.row(key[0])
            return _Cell(r[key[1]]) if r is not None else None
        v = self.chans.get(key)                                              # a CHOP channel
        return None if v is None else types.SimpleNamespace(eval=lambda v=v: v)

    def __setitem__(self, key, value):
        self.row(key[0])[key[1]] = value

    def sendOSC(self, address, args):
        self.sent.append((address, list(args)))

    @property
    def module(self):
        if self._module is None:
            comp = self.parent_
            g = {"op": comp.op, "parent": lambda: comp, "absTime": _CLOCK, "me": self, "__name__": self.name}
            exec(compile(self.text, self.name, "exec"), g)
            self._module = types.SimpleNamespace(**{k: v for k, v in g.items() if not k.startswith("__")})
        return self._module


_USED: dict = {}
_CLOCK = types.SimpleNamespace(seconds=0.0)
# Parameter names of the TD operators the builder uses (TD 2025 documentation) - a typo is caught here.
KNOWN = {
    "oscinCHOP": {"port", "stripsegments", "active"}, "oscinDAT": {"port", "callbacks", "maxlines", "active"},
    "oscoutDAT": {"address", "port", "active"}, "syphonspoutinTOP": {"sendername"},
    "syphonspoutoutTOP": {"sendername", "active"},
    "noiseTOP": {"outputresolution", "resolutionw", "resolutionh", "period", "amp", "offset", "mono", "tz"},
    "switchTOP": {"index"}, "transformTOP": {"extend", "tx", "ty"},
    "glslTOP": {"pixeldat", "outputresolution", "resolutionw", "resolutionh", "format"} |
               {"vec%d%s" % (i, s) for i in range(8) for s in ("name", "valuex", "valuey", "valuez", "valuew")},
    "blurTOP": {"size", "outputresolution", "preshrink"}, "feedbackTOP": {"top", "format", "resetpulse"},
    "textTOP": {"outputresolution", "resolutionw", "resolutionh", "text", "fontsizex", "alignx", "aligny",
                "position1", "position2", "fontcolorr", "fontcolorg", "fontcolorb", "bgalpha", "fontalpha"},
    "moviefileoutTOP": {"type", "record", "file", "videocodec"},
    "windowCOMP": {"winop", "borders", "size", "justifyh", "justifyv", "monitor", "alwaysontop", "winopen",
                   "winclose"},
    "executeDAT": {"framestart", "active"},
    "parameterexecuteDAT": {"op", "pars", "custom", "builtin", "onpulse", "valuechange", "active"},
    "baseCOMP": {"nodeview", "opviewer"},
}


def _fake_td():
    td = types.ModuleType("td")
    for t in KNOWN:
        setattr(td, t, t)
    for t in ("textDAT", "tableDAT", "nullTOP", "overTOP"):
        setattr(td, t, t)
    root = _OP("root", "", None)
    project1 = root.create("containerCOMP", "project1")
    td.op = lambda path: project1 if path == "/project1" else None
    td.project = types.SimpleNamespace(cookRate=30, save=lambda p: None)
    td.absTime = _CLOCK
    td.ParMode = types.SimpleNamespace(EXPRESSION="EXPRESSION", CONSTANT="CONSTANT")
    return td, project1


@pytest.fixture()
def td_build(monkeypatch):
    td, project1 = _fake_td()
    monkeypatch.setitem(sys.modules, "td", td)
    spec = importlib.util.spec_from_file_location("myrmex_td_under_test", TD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                                # (no build on import)
    assert project1.op("myrmex") is None
    _USED.clear()
    comp, warnings = mod.build(project1, save=False)
    return mod, comp, warnings


def test_td_builder_makes_the_whole_network(td_build):
    mod, comp, warnings = td_build
    assert warnings == []
    need = {"data_in", "text", "text_in", "to_myrmex", "syphon_in", "test", "src", "shake", "grade", "bright",
            "blur_a", "blur_b", "fb", "trails", "trail_out", "post", "hud", "hud_over", "OUT", "syphon_out", "rec",
            "window", "fx_logic", "tick", "pars_exec", "README", "text_callbacks"}
    assert need <= set(comp.children)
    c = comp.children
    assert c["data_in"].par.port.val == 7000 and c["data_in"].par.stripsegments.val == 1
    assert c["post"].inputs == [c["trail_out"], c["grade"], c["blur_a"], c["blur_b"]]
    assert c["fb"].par.top.val == "trail_out" and c["trail_out"].inputs == [c["trails"]]
    assert c["window"].par.winop.val == "OUT" and c["to_myrmex"].par.port.val == 9100
    assert c["post"].seq.vec.numBlocks == 7 and c["post"].par.vec6name.val == "uShock"
    for t, used in _USED.items():                                               # no unknown parameter names
        assert used <= KNOWN.get(t, set()), (t, used - KNOWN.get(t, set()))
    assert {"Follow", "Source", "Preset", "Record", "Window", "Send"} <= comp.custom
    assert {k.capitalize() for k in FX} <= comp.custom
    assert tuple(mod.FX) == FX and mod.PRESETS == PRESETS                        # same rack as the app


def test_td_logic_drives_the_shaders_from_myrmex_data(td_build):
    mod, comp, _ = td_build
    c = comp.children
    b = TouchBridge({"preset": "Glitch Storm"})
    body = types.SimpleNamespace(com=np.zeros(3), glow=0.5, arousal=0.7, surface=0.3, instability=0.1, heading=0.0,
                                 pos=np.zeros((3, 3)), morphology="URCHIN", behavior="DISPLAY")
    logic = c["fx_logic"].module
    for i in range(30):
        _CLOCK.seconds = 1.0 + i / 60
        v = b.update(1 / 60, st=types.SimpleNamespace(beat=i / 30, bpm=120.0, playing=True, beats_per_bar=4.0),
                     state=body, events=["BLOW"] if i == 10 else ())
        c["data_in"].chans = dict(zip(CHANNELS, map(float, v)))
        logic.update()
    post = c["post"].par
    assert post.vec1valuex.val > 0.0                                            # chromatic aberration on
    assert post.vec4valuez.val == pytest.approx(PRESETS["Glitch Storm"]["edges"])  # the app's rack, followed
    assert all(math.isfinite(getattr(post, "vec%dvalue%s" % (i, s)).val) for i in range(7) for s in "xyzw")
    assert c["to_myrmex"].sent and c["to_myrmex"].sent[0][0] == "/myrmex/td/alive"
    c["text_callbacks"].module.onReceiveOSC(None, 0, "", b"", 0, "/myrmex/text/regime", ["URCHIN"], None)
    assert c["text"].row("regime")[1] == "URCHIN"
    pe = c["pars_exec"].module
    pe.onPulse(types.SimpleNamespace(name="Resettrails"))
    assert "resetpulse" in c["fb"].pulses
    comp.par.Event.val = "SHED"
    pe.onPulse(types.SimpleNamespace(name="Send"))
    assert c["to_myrmex"].sent[-1] == ("/myrmex/trigger", ["creature:shed"])
    pe.onValueChange(types.SimpleNamespace(name="Preset", eval=lambda: "Clean"), None)
    assert comp.par.Trails.val == 0.0                                           # a preset in TD itself


def test_td_logic_runs_alone_without_myrmex(td_build):
    _, comp, _ = td_build
    logic = comp.children["fx_logic"].module
    for i in range(10):
        _CLOCK.seconds = 100.0 + i / 60
        logic.update()                                                          # the demo pulse, no errors
    assert comp.children["grade"].par.vec0valuey.val > 0.0


def test_td_shaders_compile():
    mgl = pytest.importorskip("moderngl")
    spec = importlib.util.spec_from_file_location("myrmex_td_under_test", TD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        ctx = mgl.create_standalone_context(backend="egl")
    except Exception:
        pytest.skip("no headless OpenGL here")
    prelude = ("#version 450 core\nuniform sampler2D sTD2DInputs[4];\nin vec3 vUV;\n"
               "struct TDTexInfo { vec4 res; vec4 depth; };\nuniform TDTexInfo uTDOutputInfo;\n"
               "vec4 TDOutputSwizzle(vec4 c) { return c; }\n")
    vert = "#version 450 core\nin vec2 p; out vec3 vUV;\nvoid main() { vUV = vec3(p * 0.5 + 0.5, 0.0); " \
           "gl_Position = vec4(p, 0.0, 1.0); }\n"
    for name, code in mod.SHADERS.items():
        ctx.program(vertex_shader=vert, fragment_shader=prelude + code)
