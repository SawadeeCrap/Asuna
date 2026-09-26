"""The picture effects on the GPU (Blender's gpu module) - what TouchDesigner did, inside Blender.

    scene camera --draw_view3d--> src (1920x1080 or 1080x1920, colour managed, EEVEE "Rendered")
    src --BRIGHT--> 1/2 --BLUR--> 1/4 --BLUR--> 1/8          bloom, two sizes
    src + trail(prev) --TRAIL--> trail                          light trails: what glows leaves a fading,
                                                                drifting, hue-shifting streak (feedback)
    src + bloom + trail --POST--> out                           shake, shockwave, glitch, chromatic aberration,
                                                                speed lines, impact frame, flash, grade,
                                                                scanlines, vignette, grain
    out --> the 3D view in camera view (a monitor, letterboxed), Syphon, take renders (fx_render)

Live, the 3D view that looks through the camera shows ``out``.  The offscreen picture is always drawn in
"Rendered" mode, so the view itself can stay in Solid (cheap): Blender then renders the scene once, at the
output size, instead of twice.  Every pass samples with a hand-written bilinear filter (texelFetch), so
the result is the same on every Blender (4.2 ... 5.x) and on Metal and OpenGL.
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
# center  xy the organism on screen (0..1, y up)  z its size on screen (fraction of the height)  w visible
# motion  xy its screen velocity (uv / s)  z speed 0..1  w -
# drive   x kick  y impact  z morph  w cut
# bloom   x threshold  y knee  z amount  w pop (brightness on the kick)
# trail   x decay (0 = off)  y zoom  z rotate (rad)  w hue drift (turns / frame)
# look1   x chromatic aberration (px at 1080)  y glitch  z shockwave strength  w shockwave radius
# look2   x impact frame  y speed lines  z scanlines  w grain
# look3   x vignette  y trails mix  z flash  w shake
# grade   x exposure (stops)  y contrast  z saturation  w hue (turns)
# shock   xy where the hit was  z -  w strobe
# size    x w  y h  z 1/w  w 1/h
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
COMMON = """
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
vec3 hue_rot(vec3 c, float turns) {
  float a = turns * 6.28318531;
  float cs = cos(a);
  float sn = sin(a);
  vec3 k = vec3(0.57735027);
  return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}
"""


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
  float m = max(cur.r, max(cur.g, cur.b));
  vec3 feed = cur * smoothstep(P.bloom.x - 0.3, P.bloom.x + 0.05, m);
  vec2 asp = vec2(P.frame.y, 1.0);
  vec2 p = (uv - P.center.xy) * asp;
  float cs = cos(P.trail.z);
  float sn = sin(P.trail.z);
  p = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y) / (1.0 + P.trail.y);
  vec3 prev = s_prev(p / asp + P.center.xy).rgb;
  prev = max(hue_rot(prev, P.trail.w), vec3(0.0)) * P.trail.x;
  FragColor = vec4(max(feed, prev), 1.0);
}
"""
POST = """
vec3 base(vec2 q) {
  vec3 dry = s_src(q).rgb;
  vec3 tr = s_tr(q).rgb;
  vec3 bl = s_b1(q).rgb * 0.7 + s_b2(q).rgb;
  return dry + max(tr - dry, vec3(0.0)) * P.look3.y + bl * P.bloom.z;
}

void main() {
  vec2 asp = vec2(P.frame.y, 1.0);
  float t = P.frame.x;
  vec2 q = uv;
  q += (vec2(hash12(vec2(t * 61.0, 1.7)), hash12(vec2(t * 47.0, 9.1))) - 0.5) * P.look3.w;
  vec2 d = (q - P.shock.xy) * asp;
  float dist = length(d);
  float ring = exp(-pow((dist - P.look1.w) * 16.0, 2.0)) * P.look1.z;
  q -= d / max(dist, 0.0001) / asp * ring * 0.035;
  float gsh = 0.0;
  if (P.look1.y > 0.001) {
    float rows = 6.0 + 40.0 * hash12(vec2(floor(t * 12.0), 3.1));
    float by = floor(q.y * rows);
    float seed = floor(t * 15.0);
    if (hash12(vec2(by, seed)) < P.look1.y * 0.6) {
      gsh = (hash12(vec2(seed, by)) - 0.5) * P.look1.y * 0.25;
      q.x += gsh;
    }
  }
  vec2 radial = (q - P.center.xy) * asp;
  float rl = length(radial);
  vec2 dir = rl > 0.00001 ? radial / rl : vec2(1.0, 0.0);
  vec2 off = dir / asp * (P.look1.x / 1080.0) * (0.35 + 1.3 * min(rl, 1.0));
  vec3 col = vec3(base(q + off + vec2(gsh * 0.3, 0.0)).r, base(q).g, base(q - off - vec2(gsh * 0.3, 0.0)).b);
  if (P.look2.y > 0.001) {
    vec2 pc = (uv - P.center.xy) * asp;
    float r = length(pc);
    float fa = (atan(pc.y, pc.x) / 6.28318531 + 0.5) * 160.0;
    float cell = floor(fa);
    float tick = floor(t * 18.0);
    float h1 = hash12(vec2(cell, tick));
    float h2 = hash12(vec2(cell * 1.37, tick + 7.0));
    float w = abs(fract(fa) - 0.5) * 2.0;
    float thick = 0.12 + 0.3 * h2;
    float line = 1.0 - smoothstep(thick * 0.4, thick, w);
    float r0 = 0.1 + P.center.z * 0.9 + 0.25 * h2;
    float reach = smoothstep(r0, r0 + 0.12, r);
    float ml = length(P.motion.xy);
    float behind = ml > 0.0001 ? 0.3 + 0.7 * max(0.0, dot(-P.motion.xy / ml, pc / max(r, 0.0001))) : 1.0;
    col += vec3(1.0) * line * reach * step(0.5, h1) * behind * P.look2.y;
  }
  if (P.look2.x > 0.001) {
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float v = smoothstep(0.22, 0.5, l);
    v = mod(P.frame.z, 2.0) < 1.0 ? 1.0 - v : v;
    vec3 inkpaper = mix(vec3(0.03, 0.0, 0.01), vec3(1.0, 0.97, 0.93), v);
    col = mix(col, inkpaper, P.look2.x);
  }
  col = col * (1.0 + P.bloom.w + P.shock.w) + vec3(P.look3.z);
  col *= exp2(P.grade.x);
  col = (col - 0.5) * P.grade.y + 0.5;
  float lg = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = mix(vec3(lg), col, P.grade.z);
  col = hue_rot(col, P.grade.w);
  col *= 1.0 - P.look2.z * 0.3 * (0.5 + 0.5 * sin(uv.y * P.size.y * 3.14159265));
  vec2 v2 = (uv - 0.5) * asp;
  col *= 1.0 - P.look3.x * dot(v2, v2) / (0.25 * (asp.x * asp.x + 1.0)) * 0.85;
  col += (hash12(uv * P.size.xy + fract(t * 7.0) * 311.0) - 0.5) * P.look2.w * 0.12;
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
_M: dict = {"handler": None, "pipe": None, "done": -1, "last": 0.0, "last_new": 0.0, "error": "", "fps": 0.0,
            "n": 0, "t0": 0.0, "size": (0, 0)}


def enabled() -> bool:
    return _M["handler"] is not None


def enable() -> None:
    if _M["handler"] is None:
        _M["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL")
    if not bpy.app.timers.is_registered(_pump):
        bpy.app.timers.register(_pump, first_interval=0.1, persistent=True)
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
    if now - _M["last"] > 1.0 / 30.0 and now - _M["last_new"] < 2.5:
        _tag()
    return 1.0 / 30.0


def output_size(scene) -> tuple[int, int]:
    from . import fx
    r = scene.render
    s = float(fx.S["cfg"].get("preview", 1.0)) if not bpy.app.background else 1.0
    s = min(1.0, max(0.25, s))
    return max(16, int(r.resolution_x * r.resolution_percentage / 100 * s)), \
        max(16, int(r.resolution_y * r.resolution_percentage / 100 * s))


def _draw() -> None:
    from . import fx, syphon_out
    if not fx.active():
        return
    ctx = bpy.context
    scene, space, region = ctx.scene, ctx.space_data, ctx.region
    if scene is None or space is None or space.type != "VIEW_3D" or region is None or scene.camera is None:
        return
    in_cam = space.region_3d is not None and space.region_3d.view_perspective == "CAMERA"
    now = time.perf_counter()
    fresh = fx.S["frame"] != _M["done"] or now - _M["last"] > 1.0 / 30.0
    try:
        if fresh and (in_cam or syphon_out.wants_fx()):
            if fx.S["frame"] != _M["done"]:
                _M["last_new"] = now
            _produce(ctx, scene, space, region)
            _M["done"] = fx.S["frame"]
            _M["last"] = now
        pipe = _M.get("pipe")
        if in_cam and pipe is not None:
            _show(region, pipe)
        _M["error"] = ""
    except Exception as e:                                    # never break the viewport
        _M["error"] = f"{type(e).__name__}: {e}"


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
    shading = space.shading
    old_type, old_ov = shading.type, space.overlay.show_overlays
    engine_ok = scene.render.engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH")
    try:
        if engine_ok and old_type != "RENDERED":         # the picture is always the "Rendered" one
            shading.type = "RENDERED"
        if old_ov:
            space.overlay.show_overlays = False
        pipe.src.draw_view3d(scene, ctx.view_layer, space, region, view, proj, do_color_management=True)
    finally:
        if shading.type != old_type:
            shading.type = old_type
        if space.overlay.show_overlays != old_ov:
            space.overlay.show_overlays = old_ov
    tex = pipe.run(fx.post_params(scene, w, h, cam))
    syphon_out.publish_texture(tex, w, h)
    _M["n"] += 1
    now = time.perf_counter()
    if now - _M["t0"] >= 1.0:
        _M["fps"] = _M["n"] / (now - _M["t0"]) if _M["t0"] else 0.0
        _M["t0"], _M["n"] = now, 0
    _M["size"] = (w, h)


def _show(region, pipe: Pipeline) -> None:
    rw, rh = max(1, region.width), max(1, region.height)
    s = min(rw / pipe.w, rh / pipe.h)
    dw, dh = pipe.w * s, pipe.h * s
    x0, y0 = (rw - dw) * 0.5, (rh - dh) * 0.5
    gpu.state.blend_set("NONE")
    draw_solid((0.0, 0.0, 0.0, 1.0))
    draw_rect(pipe.out.texture_color, x0 / rw * 2 - 1, y0 / rh * 2 - 1, (x0 + dw) / rw * 2 - 1, (y0 + dh) / rh * 2 - 1)
    gpu.state.blend_set("ALPHA")
    try:
        import blf
        blf.size(0, 11)
        blf.color(0, 1.0, 1.0, 1.0, 0.55)
        blf.position(0, 12, 10, 0)
        fps = f" · {_M['fps']:.0f} fps" if _M["fps"] else ""
        blf.draw(0, f"Myrmex FX · {pipe.w}×{pipe.h}{fps}" + (f" · {_M['error']}" if _M["error"] else ""))
    except Exception:
        pass


def status() -> dict:
    return {"monitor": enabled(), "size": list(_M["size"]), "fps": round(_M["fps"], 1), "error": _M["error"]}


__all__ = ["Pipeline", "shaders", "enable", "disable", "enabled", "status", "output_size", "draw_rect",
           "N_PARAMS"]
