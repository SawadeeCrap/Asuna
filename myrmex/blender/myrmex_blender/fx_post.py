"""The picture effects on the GPU (Blender's gpu module) - only what the organism itself is, on black.

    scene camera --draw_view3d--> src (1920x1080 or 1080x1920, colour managed, EEVEE "Rendered")
    src --BRIGHT--> 1/2 --BLUR--> 1/4 --BLUR--> 1/8          glow of the organism's highlights, two sizes
    src + echo(prev) --TRAIL--> echo                           motion echo: what the organism is leaves a
                                                               fading trace where it was (no drift, no hue)
    src + glow + echo --POST--> out                            exposure / contrast / saturation that keep
                                                               black black; pure black stays pure black
    out --> the 3D view in camera view (a monitor, letterboxed), Syphon, take renders (fx_render)

Nothing moves, tears, tints or dirties the frame: the background stays black, the effects stay on the
organism.  Every pass samples with a hand-written bilinear filter (texelFetch), so the result is the same
on every Blender (4.2 ... 5.x) and on Metal and OpenGL.
"""
from __future__ import annotations

import time

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

UBO = """
struct FxParams {
  vec4 frame;
  vec4 center;
  vec4 motion;
  vec4 drive;
  vec4 bloom;
  vec4 trail;
  vec4 look1;
  vec4 look2;
  vec4 look3;
  vec4 grade;
  vec4 shock;
  vec4 size;
};
"""
# frame   x time (s)  y aspect w/h  z frame counter  w -
# bloom   x threshold  y knee  z glow amount  w -
# trail   x echo decay per frame (0 = off)
# look3   y echo mix
# grade   x exposure (stops)  y contrast (a curve through 0 and 1: 1 = neutral)  z saturation
# size    x w  y h  z 1/w  w 1/h
# (center, motion, drive, look1, look2, shock: kept in the block, unused)
N_PARAMS = 48

VERT = """
void main() {
  uv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}
"""
VERT_RECT = """
void main() {
  uv = texco;
  gl_Position = vec4(pos, 0.0, 1.0);
}
"""
COMMON = ""


def _sampler_fn(fn: str, sampler: str) -> str:
    """A bilinear read of one sampler (texelFetch: no dependence on the texture's filter state)."""
    return f"""
vec4 {fn}(vec2 q) {{
  ivec2 sz = textureSize({sampler}, 0);
  vec2 p = q * vec2(sz) - 0.5;
  vec2 f = fract(p);
  ivec2 i = ivec2(floor(p));
  ivec2 lo = ivec2(0);
  ivec2 hi = sz - ivec2(1);
  vec4 a = texelFetch({sampler}, clamp(i, lo, hi), 0);
  vec4 b = texelFetch({sampler}, clamp(i + ivec2(1, 0), lo, hi), 0);
  vec4 c = texelFetch({sampler}, clamp(i + ivec2(0, 1), lo, hi), 0);
  vec4 d = texelFetch({sampler}, clamp(i + ivec2(1, 1), lo, hi), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}}
"""


BRIGHT = """
void main() {
  vec3 c = s_src(uv).rgb;
  float m = max(c.r, max(c.g, c.b));
  float k = smoothstep(P.bloom.x - P.bloom.y, P.bloom.x + P.bloom.y, m);
  FragColor = vec4(c * k, 1.0);
}
"""
BLUR = """
void main() {
  vec2 px = pp.xy / vec2(textureSize(tin, 0));
  vec3 acc = s_tin(uv).rgb * 0.2270270;
  acc += (s_tin(uv + px * 1.3846154 * pp.z).rgb + s_tin(uv - px * 1.3846154 * pp.z).rgb) * 0.3162162;
  acc += (s_tin(uv + px * 3.2307692 * pp.z).rgb + s_tin(uv - px * 3.2307692 * pp.z).rgb) * 0.0702703;
  FragColor = vec4(acc, 1.0);
}
"""
TRAIL = """
void main() {
  vec3 cur = s_src(uv).rgb;
  float l = dot(cur, vec3(0.2126, 0.7152, 0.0722));
  vec3 feed = cur * smoothstep(0.02, 0.3, l);
  vec3 prev = s_prev(uv).rgb * P.trail.x;
  FragColor = vec4(max(feed, prev), 1.0);
}
"""
POST = """
void main() {
  vec3 dry = s_src(uv).rgb;
  vec3 echo = s_tr(uv).rgb;
  vec3 glow = s_b1(uv).rgb * 0.6 + s_b2(uv).rgb * 0.8;
  vec3 col = dry + max(echo - dry, vec3(0.0)) * P.look3.y + glow * P.bloom.z;
  col *= exp2(P.grade.x);
  col = pow(max(col, vec3(0.0)), vec3(P.grade.y));
  float lg = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = max(mix(vec3(lg), col, P.grade.z), vec3(0.0));
  col = max(col - vec3(0.004), vec3(0.0)) * (1.0 / 0.996);
  FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""
SHOW = """
void main() {
  FragColor = vec4(s_img(uv).rgb, 1.0);
}
"""
SOLID = """
void main() {
  FragColor = pp;
}
"""

# name: (fragment, samplers, vertex with a rect, per-pass vec4 "pp", the FxParams block)
SPECS = {
    "bright": (BRIGHT, ("src",), False, False, True),
    "blur": (BLUR, ("tin",), False, True, False),
    "trail": (TRAIL, ("src", "prev"), False, False, True),
    "post": (POST, ("src", "b1", "b2", "tr"), False, False, True),
    "show": (SHOW, ("img",), True, False, False),
    "solid": (SOLID, (), True, True, False),
}
_SHADERS: dict = {}


def _make(name: str):
    frag, samplers, rect, pp, ubo = SPECS[name]
    info = gpu.types.GPUShaderCreateInfo()
    if ubo:
        info.typedef_source(UBO)
        info.uniform_buf(0, "FxParams", "P")
    if pp:
        info.push_constant("VEC4", "pp")
    for i, s in enumerate(samplers):
        info.sampler(i, "FLOAT_2D", s)
    info.vertex_in(0, "VEC2", "pos")
    if rect:
        info.vertex_in(1, "VEC2", "texco")
    iface = gpu.types.GPUStageInterfaceInfo("myrmex_fx_" + name)
    iface.smooth("VEC2", "uv")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "FragColor")
    info.vertex_source(VERT_RECT if rect else VERT)
    info.fragment_source(COMMON + "".join(_sampler_fn("s_" + s, s) for s in samplers) + frag)
    sh = gpu.shader.create_from_info(info)
    del iface, info
    return sh


def shaders() -> dict:
    if not _SHADERS:
        for name in SPECS:
            _SHADERS[name] = _make(name)
    return _SHADERS


QUAD = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
QUAD_IDX = ((0, 1, 2), (2, 1, 3))


class Pipeline:
    """The passes at one output size.  ``run(params)`` -> the finished picture (RGBA8 texture)."""

    def __init__(self, w: int, h: int):
        self.w, self.h = int(w), int(h)
        sh = shaders()
        self.src = gpu.types.GPUOffScreen(self.w, self.h)
        f16 = {"format": "RGBA16F"}
        self.half = gpu.types.GPUOffScreen(max(1, self.w // 2), max(1, self.h // 2), **f16)
        self.qa = gpu.types.GPUOffScreen(max(1, self.w // 4), max(1, self.h // 4), **f16)
        self.qb = gpu.types.GPUOffScreen(max(1, self.w // 4), max(1, self.h // 4), **f16)
        self.ea = gpu.types.GPUOffScreen(max(1, self.w // 8), max(1, self.h // 8), **f16)
        self.eb = gpu.types.GPUOffScreen(max(1, self.w // 8), max(1, self.h // 8), **f16)
        self.trail = [gpu.types.GPUOffScreen(self.w, self.h, **f16) for _ in range(2)]
        self.out = gpu.types.GPUOffScreen(self.w, self.h)
        self.ti = 0
        self.fresh = True
        self.params = np.zeros(N_PARAMS, np.float32)
        self.ubo = gpu.types.GPUUniformBuf(gpu.types.Buffer("FLOAT", N_PARAMS, self.params.tolist()))
        self.quad = batch_for_shader(sh["post"], "TRIS", {"pos": QUAD}, indices=QUAD_IDX)
        self.frames = 0

    def free(self) -> None:
        for off in (self.src, self.half, self.qa, self.qb, self.ea, self.eb, self.out, *self.trail):
            try:
                off.free()
            except Exception:
                pass

    def _pass(self, off, name: str, tex: dict, pp=None) -> None:
        sh = shaders()[name]
        with off.bind():
            gpu.state.blend_set("NONE")
            gpu.state.depth_test_set("NONE")
            sh.bind()
            if SPECS[name][4]:
                sh.uniform_block("P", self.ubo)
            if pp is not None:
                sh.uniform_float("pp", pp)
            for k, t in tex.items():
                sh.uniform_sampler(k, t)
            self.quad.draw(sh)

    def run(self, params, src_tex=None):
        """``params``: N_PARAMS floats (fx.post_params).  ``src_tex``: another source than ``self.src``."""
        p = np.asarray(params, np.float32)
        self.params[:len(p)] = p[:N_PARAMS]
        self.ubo.update(gpu.types.Buffer("FLOAT", N_PARAMS, self.params.tolist()))
        src = src_tex if src_tex is not None else self.src.texture_color
        self._pass(self.half, "bright", {"src": src})
        self._pass(self.qa, "blur", {"tin": self.half.texture_color}, (1.0, 0.0, 1.0, 0.0))
        self._pass(self.qb, "blur", {"tin": self.qa.texture_color}, (0.0, 1.0, 1.0, 0.0))
        self._pass(self.ea, "blur", {"tin": self.qb.texture_color}, (1.0, 0.0, 1.6, 0.0))
        self._pass(self.eb, "blur", {"tin": self.ea.texture_color}, (0.0, 1.0, 1.6, 0.0))
        trail_tex = src
        if self.params[20] > 0.0:                                 # trail decay: the feedback is on
            prev, cur = self.trail[self.ti], self.trail[1 - self.ti]
            if self.fresh:
                with prev.bind():
                    gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 1.0))
                self.fresh = False
            self._pass(cur, "trail", {"src": src, "prev": prev.texture_color})
            self.ti = 1 - self.ti
            trail_tex = cur.texture_color
        else:
            self.fresh = True
        self._pass(self.out, "post", {"src": src, "b1": self.qb.texture_color, "b2": self.eb.texture_color,
                                      "tr": trail_tex})
        self.frames += 1
        return self.out.texture_color

    def reset(self) -> None:
        self.fresh = True

    def read(self) -> np.ndarray:
        """The finished picture as (h, w, 4) uint8."""
        buf = self.out.texture_color.read()
        try:
            a = np.frombuffer(buf, np.uint8)
        except (TypeError, ValueError):
            a = np.array(buf.to_list(), np.uint8)
        return a.reshape(self.h, self.w, 4)


def draw_rect(tex, x0: float, y0: float, x1: float, y1: float) -> None:
    """Draw a texture into NDC rect (x0, y0)-(x1, y1) of whatever is bound (a 3D view region)."""
    sh = shaders()["show"]
    b = batch_for_shader(sh, "TRIS", {"pos": ((x0, y0), (x1, y0), (x0, y1), (x1, y1)),
                                     "texco": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))}, indices=QUAD_IDX)
    sh.bind()
    sh.uniform_sampler("img", tex)
    b.draw(sh)


def draw_solid(color, x0=-1.0, y0=-1.0, x1=1.0, y1=1.0) -> None:
    sh = shaders()["solid"]
    b = batch_for_shader(sh, "TRIS", {"pos": ((x0, y0), (x1, y0), (x0, y1), (x1, y1)),
                                     "texco": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))}, indices=QUAD_IDX)
    sh.bind()
    sh.uniform_float("pp", tuple(float(c) for c in color))
    b.draw(sh)


# ---------------------------------------------------------------------- the monitor in the 3D view
# How the live picture is made without choking Blender:
# * the offscreen picture is drawn with the 3D view's own settings - nothing is switched while Blender
#   draws (changing the shading inside a draw frees the very region being drawn);
# * draw_view3d renders *all* of EEVEE's viewport samples on every call (16 by default = 16 renders per
#   frame): while the monitor (or Syphon) runs the viewport samples are 1 - a live picture changes every
#   frame anyway.  The scene's own value is kept and comes back when they stop (a saved look keeps it);
# * EEVEE compiles materials for an offscreen picture synchronously (the UI would stop): the monitor waits
#   until the view (in "Rendered") has compiled them; a warm-up object carries the afterimage and ribbon
#   materials from the start;
# * it paces itself: never more often than it can afford; if Blender cannot keep up it lowers the size,
#   then pauses (the organism, its afterimages and ribbons stay in the normal view).
_M: dict = {"handler": None, "pipe": None, "done": -1, "last": 0.0, "last_new": 0.0, "error": "", "note": "",
            "fps": 0.0, "n": 0, "t0": 0.0, "size": (0, 0), "busy": 0.0, "period": 0.0, "interval": 1.0 / 60.0,
            "warm_until": 0.0, "slow": 0, "paused": False, "scale": 1.0}
WARM_S = 6.0                    # the 3D view compiles the materials meanwhile (in the background)
SLOW_PERIOD = 0.15              # live pictures further apart than this: Blender is struggling
_LIMIT: set = set()
TAA_KEY = "myrmex_fx_taa"
EEVEE = ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")


def limit_samples(owner: str, on: bool, scene=None) -> None:
    """EEVEE's viewport samples at 1 while someone draws offscreen pictures every frame (monitor, Syphon)."""
    if on:
        _LIMIT.add(owner)
    else:
        _LIMIT.discard(owner)
    sc = scene or bpy.context.scene
    if sc is None:
        return
    try:
        ee = sc.eevee
        if _LIMIT and TAA_KEY not in sc:
            sc[TAA_KEY] = int(ee.taa_samples)
            ee.taa_samples = 1
        elif not _LIMIT and TAA_KEY in sc:
            ee.taa_samples = int(sc[TAA_KEY])
            del sc[TAA_KEY]
    except (AttributeError, TypeError, ValueError, KeyError):
        pass


def reapply(scene=None) -> None:
    """A new file / look was opened while the monitor or Syphon runs: its scene gets the 1-sample limit,
    and the monitor waits for its materials again."""
    if _LIMIT:
        limit_samples(next(iter(_LIMIT)), True, scene)
    if _M["handler"] is not None:
        _M.update(warm_until=time.perf_counter() + WARM_S, paused=False, slow=0, note="", period=0.0)
        try:
            from . import fx
            fx.warmup(scene)
        except Exception as e:
            print("Myrmex FX warm-up:", e)


def enabled() -> bool:
    return _M["handler"] is not None


def enable() -> None:
    if _M["handler"] is None:
        _M["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL")
    if not bpy.app.timers.is_registered(_pump):
        bpy.app.timers.register(_pump, first_interval=0.1, persistent=True)
    _M.update(warm_until=time.perf_counter() + WARM_S, paused=False, slow=0, note="", scale=1.0, busy=0.0,
              period=0.0, interval=1.0 / 60.0)
    limit_samples("monitor", True)
    try:
        shaders()                                     # compiled now, not in the middle of a draw
    except Exception as e:
        print("Myrmex FX shaders (compiled at the first picture instead):", e)
    try:
        from . import fx
        fx.warmup(bpy.context.scene)
    except Exception as e:
        print("Myrmex FX warm-up:", e)
    _tag()


def disable() -> None:
    if _M["handler"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_M["handler"], "WINDOW")
        except ValueError:
            pass
        _M["handler"] = None
    if bpy.app.timers.is_registered(_pump):
        bpy.app.timers.unregister(_pump)
    pipe = _M.get("pipe")
    if pipe is not None:
        pipe.free()
    _M["pipe"] = None
    limit_samples("monitor", False)
    _tag()


def _tag() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _pump():
    """Keep the monitor drawing for a moment after the last frame (the trails still fade), then rest."""
    if _M["handler"] is None:
        return None
    now = time.perf_counter()
    if now < _M["warm_until"] + 1.0 or (now - _M["last"] > 1.0 / 30.0 and now - _M["last_new"] < 2.5):
        _tag()
    return 1.0 / 30.0


def output_size(scene) -> tuple[int, int]:
    from . import fx
    r = scene.render
    s = float(fx.S["cfg"].get("preview", 1.0)) * float(_M.get("scale", 1.0)) if not bpy.app.background else 1.0
    s = min(1.0, max(0.25, s))
    return max(16, int(r.resolution_x * r.resolution_percentage / 100 * s)), \
        max(16, int(r.resolution_y * r.resolution_percentage / 100 * s))


def _compiling() -> bool:
    try:
        return bool(bpy.app.is_job_running("SHADER_COMPILATION"))
    except (TypeError, ValueError, AttributeError):
        return False


def _why_not(scene, space) -> str:
    """Why the monitor cannot draw this view right now ("" = it can)."""
    engine = scene.render.engine
    if engine not in EEVEE and engine != "BLENDER_WORKBENCH":
        return "Myrmex FX: the live picture needs EEVEE (Render engine)"
    if engine in EEVEE and space.shading.type not in ("RENDERED", "MATERIAL"):
        return "Myrmex FX: switch this view to Rendered (Z) to see the effects"
    now = time.perf_counter()
    if now < _M["warm_until"] or (now < _M["warm_until"] + 20.0 and _compiling()):
        return "Myrmex FX: preparing shaders…"
    if _M["paused"]:
        return _M["note"] or "Myrmex FX paused"
    return ""


def _draw() -> None:
    from . import fx, syphon_out
    if not fx.active():
        return
    ctx = bpy.context
    scene, space, region = ctx.scene, ctx.space_data, ctx.region
    if scene is None or space is None or space.type != "VIEW_3D" or region is None or scene.camera is None:
        return
    in_cam = space.region_3d is not None and space.region_3d.view_perspective == "CAMERA"
    try:
        why = _why_not(scene, space)
        if why:
            if in_cam:
                _text(why)
            return
        now = time.perf_counter()
        new = fx.S["frame"] != _M["done"]
        due = (new or now - _M["last"] > 0.25) and now - _M["last"] >= _M["interval"]
        if due and (in_cam or syphon_out.wants_fx()):
            live = new and now - _M["last_new"] < 0.5         # frames keep coming: the pace means something
            period = now - _M["last"]
            if new:
                _M["last_new"] = now
            t0 = time.perf_counter()
            _produce(ctx, scene, space, region)
            _adapt(time.perf_counter() - t0, period if live else None)
            _M["done"] = fx.S["frame"]
            _M["last"] = now
        pipe = _M.get("pipe")
        if in_cam and pipe is not None:
            _show(region, pipe)
        _M["error"] = ""
    except Exception as e:                                    # never break the viewport
        _M["error"] = f"{type(e).__name__}: {e}"


def _adapt(dt: float, period: float | None = None) -> None:
    """Pace: at most as often as a picture takes (x1.3).  If Blender cannot keep up (the pictures of a
    running performance come further apart than SLOW_PERIOD, or one takes over 0.2 s), lower the size,
    then pause."""
    _M["busy"] = 0.7 * _M["busy"] + 0.3 * dt if _M["busy"] else dt
    _M["interval"] = min(1.0 / 12.0, max(1.0 / 60.0, _M["busy"] * 1.3))
    if period is not None:
        _M["period"] = 0.8 * _M["period"] + 0.2 * period if _M["period"] else period
    slow = _M["busy"] > 0.2 or (period is not None and _M["period"] > SLOW_PERIOD)
    _M["slow"] = _M["slow"] + 1 if slow else max(0, _M["slow"] - 2)
    if _M["slow"] < 20:
        return
    _M["slow"] = 0
    if _M["scale"] > 0.5:
        _M["scale"] = max(0.5, _M["scale"] * 0.7)
        _M["note"] = f"Myrmex FX: live picture lowered to {int(_M['scale'] * 100)}% (Blender was too slow)"
    else:
        _M["paused"] = True
        _M["note"] = (f"Myrmex FX paused: Blender manages {1.0 / max(_M['period'], 1e-3):.0f} pictures a second - "
                      "set Live picture lower, then turn Myrmex FX off and on")


def _produce(ctx, scene, space, region) -> None:
    from . import fx, syphon_out
    w, h = output_size(scene)
    pipe = _M.get("pipe")
    if pipe is None or (pipe.w, pipe.h) != (w, h):
        if pipe is not None:
            pipe.free()
        pipe = _M["pipe"] = Pipeline(w, h)
    cam = scene.camera
    depsgraph = ctx.evaluated_depsgraph_get()
    view = cam.matrix_world.inverted()
    proj = cam.calc_matrix_camera(depsgraph, x=w, y=h)
    pipe.src.draw_view3d(scene, ctx.view_layer, space, region, view, proj, do_color_management=True)
    tex = pipe.run(fx.post_params(scene, w, h, cam))
    syphon_out.publish_texture(tex, w, h)
    _M["n"] += 1
    now = time.perf_counter()
    if now - _M["t0"] >= 1.0:
        _M["fps"] = _M["n"] / (now - _M["t0"]) if _M["t0"] else 0.0
        _M["t0"], _M["n"] = now, 0
    _M["size"] = (w, h)


def _text(msg: str, y: int = 10) -> None:
    try:
        import blf
        gpu.state.blend_set("ALPHA")
        blf.size(0, 11)
        blf.color(0, 1.0, 1.0, 1.0, 0.7)
        blf.position(0, 12, y, 0)
        blf.draw(0, msg)
    except Exception:
        pass


def _show(region, pipe: Pipeline) -> None:
    rw, rh = max(1, region.width), max(1, region.height)
    s = min(rw / pipe.w, rh / pipe.h)
    dw, dh = pipe.w * s, pipe.h * s
    x0, y0 = (rw - dw) * 0.5, (rh - dh) * 0.5
    gpu.state.blend_set("NONE")
    draw_solid((0.0, 0.0, 0.0, 1.0))
    draw_rect(pipe.out.texture_color, x0 / rw * 2 - 1, y0 / rh * 2 - 1, (x0 + dw) / rw * 2 - 1, (y0 + dh) / rh * 2 - 1)
    fps = f" · {_M['fps']:.0f} fps" if _M["fps"] else ""
    _text(f"Myrmex FX · {pipe.w}×{pipe.h}{fps}" + (f" · {_M['error']}" if _M["error"] else ""))
    if _M["note"]:
        _text(_M["note"], 26)


def status() -> dict:
    return {"monitor": enabled(), "size": list(_M["size"]), "fps": round(_M["fps"], 1), "error": _M["error"],
            "note": _M["note"], "paused": _M["paused"], "ms": round(1000 * _M["busy"], 1)}


__all__ = ["Pipeline", "shaders", "enable", "disable", "enabled", "status", "output_size", "draw_rect",
           "limit_samples", "N_PARAMS", "TAA_KEY"]
