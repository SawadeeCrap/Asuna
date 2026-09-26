"""Myrmex FX for TouchDesigner - builds the whole network in one go (made for TD 2025.3x, Non-Commercial is fine).

In TouchDesigner open the Textport (Dialogs > Textport and DATs, or Alt+T), paste the line the Myrmex app
copied for you and press Enter:

    exec(open('/Users/you/Myrmex/touchdesigner/myrmex_td.py').read())

It builds /project1/myrmex and saves the project as ~/Myrmex/touchdesigner/Myrmex_FX.toe; from then on the
Myrmex app opens that file directly.  Running it again rebuilds the network and keeps your settings.

    Myrmex ──OSC 7000 (data) / 7001 (text)──> [data_in] ──> fx_logic (Python, every frame)
    Blender ──Syphon "Myrmex"──> [syphon_in] ─> grade ─┬─> trails ⟲ feedback ─┐
                                                       ├─> bright ─> blur ×2 ──┼─> post ─> HUD ─> OUT
                                                       └──────────── dry ──────┘            ├─> window (2nd display)
    TD ──OSC 9100──> Myrmex (heartbeat, events)                                              ├─> Syphon "Myrmex FX"
                                                                                             └─> ProRes recording
Everything the music and the organism do arrives as channels (see myrmex/realtime/touch.py); the effect
amounts are the Myrmex app's FX rack (Follow on) or this component's own parameters (Follow off).
"""
import os

VERSION = 1
PARENT = "/project1"
NAME = "myrmex"
DATA_PORT, TEXT_PORT, MYRMEX_PORT = 7000, 7001, 9100
WIDTH, HEIGHT = 1280, 720                     # Non-Commercial TouchDesigner: at most 1280 x 1280
SAVE_AS = os.path.expanduser("~/Myrmex/touchdesigner/Myrmex_FX.toe")

FX = ("bloom", "trails", "chroma", "glitch", "warp", "shock", "kaleido", "edges", "grain", "vignette", "hud",
      "react", "exposure", "contrast", "saturation", "hue", "mix")
FX_DEFAULTS = {"bloom": 0.6, "trails": 0.45, "chroma": 0.3, "glitch": 0.25, "warp": 0.15, "shock": 0.5,
               "kaleido": 0.0, "edges": 0.0, "grain": 0.25, "vignette": 0.4, "hud": 0.0, "react": 0.8,
               "exposure": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.0, "mix": 1.0}
PRESETS = {
    "Clean": {"bloom": 0.45, "trails": 0.0, "chroma": 0.1, "glitch": 0.0, "warp": 0.0, "shock": 0.2, "kaleido": 0.0,
              "edges": 0.0, "grain": 0.15, "vignette": 0.35, "hud": 0.0, "saturation": 0.5, "hue": 0.0},
    "Neon Trails": {"bloom": 0.85, "trails": 0.75, "chroma": 0.3, "glitch": 0.1, "warp": 0.1, "shock": 0.5,
                    "kaleido": 0.0, "edges": 0.0, "grain": 0.2, "vignette": 0.4, "hud": 0.0, "saturation": 0.6,
                    "hue": 0.3},
    "Glitch Storm": {"bloom": 0.5, "trails": 0.3, "chroma": 0.65, "glitch": 0.9, "warp": 0.3, "shock": 0.7,
                     "kaleido": 0.0, "edges": 0.1, "grain": 0.45, "vignette": 0.45, "hud": 0.3, "saturation": 0.45,
                     "hue": 0.1},
    "Dream": {"bloom": 0.95, "trails": 0.9, "chroma": 0.2, "glitch": 0.0, "warp": 0.45, "shock": 0.3,
              "kaleido": 0.0, "edges": 0.0, "grain": 0.2, "vignette": 0.5, "hud": 0.0, "saturation": 0.7, "hue": 0.5},
    "Kaleido": {"bloom": 0.65, "trails": 0.55, "chroma": 0.3, "glitch": 0.1, "warp": 0.1, "shock": 0.5,
                "kaleido": 0.6, "edges": 0.0, "grain": 0.2, "vignette": 0.45, "hud": 0.0, "saturation": 0.6,
                "hue": 0.25},
    "Scanner": {"bloom": 0.5, "trails": 0.35, "chroma": 0.25, "glitch": 0.2, "warp": 0.05, "shock": 0.6,
                "kaleido": 0.0, "edges": 0.85, "grain": 0.35, "vignette": 0.5, "hud": 1.0, "saturation": 0.25,
                "hue": 0.0},
    "Liquid": {"bloom": 0.7, "trails": 0.65, "chroma": 0.25, "glitch": 0.05, "warp": 0.7, "shock": 0.6,
               "kaleido": 0.0, "edges": 0.0, "grain": 0.2, "vignette": 0.4, "hud": 0.0, "saturation": 0.55,
               "hue": 0.15},
}
EVENTS = ("MORPHOLOGY_SHIFT", "IMPULSE", "OBSTACLE", "COLLAPSE", "RECONSTRUCTION", "LASH", "COIL", "UNFURL", "CLAP",
          "FURL", "BLOOM", "SPROUT", "SHED", "PULSE", "SPLIT", "SURGE", "CALM", "BLOW", "ANNEAL", "OVERLOAD", "DASH",
          "SCATTER", "GATHER", "SLASH", "RECONFIGURE", "POUNCE", "STRIKE", "OSSIFY", "QUILLS", "SCAN", "GLITCH")

# ---------------------------------------------------------------------------------------------- shaders
# TouchDesigner GLSL TOP pixel shaders (TD adds #version, the input samplers sTD2DInputs[], vUV,
# uTDOutputInfo and TDOutputSwizzle).  Every uniform is a vec4 set by fx_logic each frame.
HUE = """
vec3 hueRot(vec3 c, float h) {
    const mat3 toYIQ = mat3(0.299, 0.596, 0.211, 0.587, -0.274, -0.523, 0.114, -0.322, 0.312);
    const mat3 toRGB = mat3(1.0, 1.0, 1.0, 0.956, -0.272, -1.106, 0.621, -0.647, 1.703);
    vec3 yiq = toYIQ * c;
    float a = h * 6.28318531;
    float cs = cos(a);
    float sn = sin(a);
    yiq.yz = vec2(cs * yiq.y - sn * yiq.z, sn * yiq.y + cs * yiq.z);
    return toRGB * yiq;
}
"""

GRADE = """// Myrmex: exposure, contrast, saturation, hue, flip, beat strobe
uniform vec4 uGrade;    // x exposure (stops)  y contrast  z saturation  w hue (turns)
uniform vec4 uMisc;     // x flip (0/1)  y -  z -  w strobe
layout(location = 0) out vec4 fragColor;
""" + HUE + """
void main() {
    vec2 uv = vUV.st;
    if (uMisc.x > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    vec3 col = texture(sTD2DInputs[0], uv).rgb * exp2(uGrade.x);
    col = (col - 0.5) * uGrade.y + 0.5;
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(l), col, uGrade.z);
    col = hueRot(col, uGrade.w);
    col *= 1.0 + 0.6 * uMisc.w;
    fragColor = TDOutputSwizzle(vec4(max(col, vec3(0.0)), 1.0));
}
"""

BRIGHT = """// Myrmex: what glows (bloom prefilter, soft knee)
uniform vec4 uBright;   // x threshold  y knee  z gain
layout(location = 0) out vec4 fragColor;
void main() {
    vec3 c = texture(sTD2DInputs[0], vUV.st).rgb;
    float m = max(c.r, max(c.g, c.b));
    float k = smoothstep(uBright.x - uBright.y, uBright.x + uBright.y, m);
    fragColor = TDOutputSwizzle(vec4(c * k * uBright.z, 1.0));
}
"""

TRAILS = """// Myrmex: echoes of the motion - the last frames, fading, drifting round the organism
uniform vec4 uTrail;    // x decay (0 = off)  y zoom  z rotate (rad)  w hue drift (turns)
uniform vec4 uCenter;   // xy the organism on screen  z aspect (w / h)  w -
layout(location = 0) out vec4 fragColor;
""" + HUE + """
void main() {
    vec2 uv = vUV.st;
    vec3 cur = texture(sTD2DInputs[0], uv).rgb;
    vec2 asp = vec2(uCenter.z, 1.0);
    vec2 p = (uv - uCenter.xy) * asp;
    float cs = cos(uTrail.z);
    float sn = sin(uTrail.z);
    p = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y) / (1.0 + uTrail.y);
    vec3 prev = texture(sTD2DInputs[1], p / asp + uCenter.xy).rgb;
    prev = max(hueRot(prev, uTrail.w), vec3(0.0)) * uTrail.x;
    fragColor = TDOutputSwizzle(vec4(max(cur, prev), 1.0));
}
"""

POST = """// Myrmex: bloom, chromatic aberration, shockwave, glitch, warp, kaleidoscope, edges, HUD, grain, vignette
uniform vec4 uMixB;     // x trails mix  y bloom  z flash  w pop (brightness on the kick)
uniform vec4 uChroma;   // x aberration (px)  y radial 0..1  z shockwave radius  w shockwave strength
uniform vec4 uGlitch;   // x amount  y block height (px)  z seed  w scanlines
uniform vec4 uWarp;     // x amount  y scale  z time  w kaleidoscope segments (0 = off)
uniform vec4 uLook;     // x grain  y vignette  z edges  w HUD
uniform vec4 uCenter;   // xy the organism on screen  z aspect  w time
uniform vec4 uShock;    // xy where the hit was  z the organism's size on screen  w visible (0/1)
layout(location = 0) out vec4 fragColor;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

vec3 base(vec2 uv) {
    vec3 trail = texture(sTD2DInputs[0], uv).rgb;
    vec3 dry = texture(sTD2DInputs[1], uv).rgb;
    vec3 bloom = texture(sTD2DInputs[2], uv).rgb * 0.6 + texture(sTD2DInputs[3], uv).rgb * 0.9;
    return mix(dry, trail, uMixB.x) + bloom * uMixB.y;
}

float luma(vec2 uv) {
    return dot(texture(sTD2DInputs[1], uv).rgb, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    vec2 res = uTDOutputInfo.res.zw;
    vec2 asp = vec2(uCenter.z, 1.0);
    vec2 uv = vUV.st;
    if (uWarp.w >= 2.0) {                                   // kaleidoscope round the organism
        vec2 p = (uv - uCenter.xy) * asp;
        float r = length(p);
        float seg = 6.28318531 / uWarp.w;
        float a = mod(atan(p.y, p.x), seg);
        a = abs(a - 0.5 * seg);
        uv = vec2(cos(a), sin(a)) * r / asp + uCenter.xy;
    }
    if (uWarp.x > 0.0001) {                                 // a liquid warp
        vec2 q = uv * uWarp.y;
        uv += (vec2(vnoise(q + uWarp.z), vnoise(q - uWarp.z + 17.3)) - 0.5) * uWarp.x * 0.06;
    }
    vec2 d = (uv - uShock.xy) * asp;                        // a shockwave from where the hit was
    float dist = length(d);
    float ring = exp(-pow((dist - uChroma.z) * 14.0, 2.0)) * uChroma.w;
    uv -= d / max(dist, 0.0001) / asp * ring * 0.03;
    if (uGlitch.x > 0.001) {                                // glitch: torn rows
        float rows = max(uGlitch.y, 2.0);
        float by = floor(uv.y * res.y / rows);
        float seed = floor(uGlitch.z * 15.0);
        if (hash12(vec2(by, seed)) < uGlitch.x * 0.5) {
            uv.x += (hash12(vec2(seed, by)) - 0.5) * uGlitch.x * 0.2;
        }
    }
    vec2 radial = (uv - uCenter.xy) * asp + vec2(0.00001);
    vec2 dir = mix(vec2(1.0, 0.0), normalize(radial), uChroma.y);
    vec2 off = dir / asp * uChroma.x / res.y;
    vec3 col = vec3(base(uv + off).r, base(uv).g, base(uv - off).b);
    if (uLook.z > 0.001) {                                  // neon outline (Sobel)
        vec2 px = 1.0 / res;
        float tl = luma(uv + px * vec2(-1.0, 1.0));
        float tc = luma(uv + px * vec2(0.0, 1.0));
        float tr = luma(uv + px * vec2(1.0, 1.0));
        float ml = luma(uv + px * vec2(-1.0, 0.0));
        float mr = luma(uv + px * vec2(1.0, 0.0));
        float bl = luma(uv + px * vec2(-1.0, -1.0));
        float bc = luma(uv + px * vec2(0.0, -1.0));
        float br = luma(uv + px * vec2(1.0, -1.0));
        float gx = tr + 2.0 * mr + br - tl - 2.0 * ml - bl;
        float gy = tl + 2.0 * tc + tr - bl - 2.0 * bc - br;
        float edge = smoothstep(0.05, 0.6, length(vec2(gx, gy)));
        col = mix(col, col * 0.35, uLook.z * 0.6) + vec3(0.55, 0.85, 1.0) * edge * uLook.z * 1.6;
    }
    col *= 1.0 - uGlitch.w * 0.25 * (0.5 + 0.5 * sin(vUV.t * res.y * 3.14159265));
    col = col * (1.0 + uMixB.w) + vec3(uMixB.z);
    if (uLook.w > 0.001 && uShock.w > 0.5) {                // HUD: brackets round the organism
        vec2 q = abs((vUV.st - uCenter.xy) * asp);
        float s = clamp(uShock.z * 0.5, 0.05, 0.45);
        float box = max(q.x, q.y);
        float line = smoothstep(2.0 / res.y, 0.0, abs(box - s)) * step(0.62 * s, min(q.x, q.y));
        float dotc = smoothstep(4.0 / res.y, 0.0, length(q));
        col += vec3(0.6, 0.95, 1.0) * (line + dotc) * uLook.w;
    }
    vec2 v = (vUV.st - 0.5) * asp;
    col *= 1.0 - uLook.y * dot(v, v) * 0.9;
    col += (hash12(vUV.st * res + fract(uCenter.w * 7.0) * 311.0) - 0.5) * uLook.x * 0.1;
    fragColor = TDOutputSwizzle(vec4(max(col, vec3(0.0)), 1.0));
}
"""

SHADERS = {"shader_grade": GRADE, "shader_bright": BRIGHT, "shader_trails": TRAILS, "shader_post": POST}

# ---------------------------------------------------------------------------------------------- per-frame logic
LOGIC = '''"""Myrmex FX logic: reads the channels from Myrmex every frame and drives the shaders (edit freely)."""
import math
import os
import time

FX = __FX__
FX_DEFAULTS = __FX_DEFAULTS__
PRESETS = __PRESETS__
S = {"last": 0.0, "beat": None, "beat_t": -99.0, "cx": 0.5, "cy": 0.5, "fps": 60.0, "hb": 0.0, "sender": 0.0,
     "hud": 0.0, "seed": 0.0, "was_rec": False}


def ch(name, default=0.0):
    c = op("data_in")
    try:
        x = c[name]
        return float(x.eval()) if x is not None else default
    except Exception:
        return default


def text(key, default=""):
    t = op("text")
    try:
        cell = t[key, 1]
        return str(cell.val) if cell is not None else default
    except Exception:
        return default


def live():
    return absTime.seconds - S["beat_t"] < 1.0


def fx(name):
    comp = parent()
    if comp.par.Follow.eval() and live() and op("data_in")["c_" + name] is not None:
        return ch("c_" + name, FX_DEFAULTS[name])
    return float(getattr(comp.par, name.capitalize()).eval())


def apply_preset(name):
    comp = parent()
    for k, v in PRESETS.get(name, {}).items():
        try:
            setattr(comp.par, k.capitalize(), v)
        except Exception:
            pass


def _set(o, name, v):
    try:
        setattr(o.par, name, v)
    except Exception:
        pass


def _uni(o, i, x, y=0.0, z=0.0, w=0.0):
    if o is None:
        return
    p = o.par
    try:
        setattr(p, "vec%dvaluex" % i, x)
        setattr(p, "vec%dvaluey" % i, y)
        setattr(p, "vec%dvaluez" % i, z)
        setattr(p, "vec%dvaluew" % i, w)
    except Exception:
        pass


def update():
    comp = parent()
    now = absTime.seconds
    dt = min(0.1, max(0.001, now - S["last"]))
    S["last"] = now
    S["fps"] += (1.0 / dt - S["fps"]) * 0.05
    beat = ch("beat", -1.0)
    if beat != S["beat"]:
        S["beat"], S["beat_t"] = beat, now
    on = live()
    W, H = int(comp.par.Width.eval()), int(comp.par.Height.eval())
    aspect = W / max(H, 1)
    for n in ("grade", "trails", "post", "hud", "test"):
        o = op(n)
        try:
            if o is not None and (o.par.resolutionw.eval() != W or o.par.resolutionh.eval() != H):
                _set(o, "resolutionw", W)
                _set(o, "resolutionh", H)
        except Exception:
            pass
    # what the music and the organism are doing (a slow demo pulse when Myrmex is silent)
    if on:
        energy, kick, impact = ch("energy"), ch("kick"), ch("impact")
        fxb, fxt, fxg, fxc = ch("fx_bloom", 0.4), ch("fx_trails", 0.4), ch("fx_glitch"), ch("fx_chroma", 0.2)
        fxw, flash, shake, shock = ch("fx_warp"), ch("fx_flash"), ch("fx_shake"), ch("fx_shock", 1.0)
        hue, strobe = ch("fx_hue"), ch("fx_strobe")
        sx, sy, vis, ssize = ch("sx", 0.5), ch("sy", 0.5), ch("visible"), ch("ssize", 0.3)
        shx, shy = ch("shock_x", 0.5), ch("shock_y", 0.5)
    else:
        ph = (now * 2.0) % 1.0
        energy, kick, impact = 0.5, math.exp(-ph * 8.0), 0.0
        fxb, fxt, fxg, fxc = 0.5 + 0.2 * kick, 0.5, 0.0, 0.2 + 0.4 * kick
        fxw, flash, shake, shock = 0.1, 0.0, 0.0, 1.0
        hue, strobe = (now / 16.0) % 1.0, 0.0
        sx, sy, vis, ssize, shx, shy = 0.5, 0.5, 1.0, 0.3, 0.5, 0.5
    if vis < 0.5:
        sx, sy = 0.5, 0.5
    k = min(1.0, dt * 6.0)
    S["cx"] += (sx - S["cx"]) * k
    S["cy"] += (sy - S["cy"]) * k
    cx, cy = S["cx"], S["cy"]
    react = fx("react")

    def m(amount, drive):                                   # an amount, moved by the music / the organism
        return amount * ((1.0 - react) + react * drive)
    bloom, trails, chroma, glitch = fx("bloom"), fx("trails"), fx("chroma"), fx("glitch")
    warp, shk, kal, edges = fx("warp"), fx("shock"), fx("kaleido"), fx("edges")
    grain, vig, hud, mixa = fx("grain"), fx("vignette"), fx("hud"), fx("mix")
    expo, contr, sat, hue_amt = fx("exposure"), fx("contrast"), fx("saturation"), fx("hue")
    src = op("src")
    if src is not None:
        _set(src, "index", 1 if str(comp.par.Source.eval()) == "test" else 0)
    _uni(op("grade"), 0, (expo - 0.5) * 3.0, 0.6 + 0.8 * contr, 2.0 * sat, hue_amt * hue)
    _uni(op("grade"), 1, 1.0 if comp.par.Flip.eval() else 0.0, 0.0, 0.0, strobe * react * 0.5)
    _uni(op("bright"), 0, 0.62 - 0.3 * bloom, 0.22, 1.0 + react * kick)
    _set(op("blur_a"), "size", 4.0 + 10.0 * bloom)
    _set(op("blur_b"), "size", 12.0 + 36.0 * bloom)
    decay = min(0.975, trails * (0.8 + 0.17 * min(1.0, fxt))) if trails > 0.01 else 0.0
    _uni(op("trails"), 0, decay, (0.002 + 0.012 * trails * energy) if trails > 0.01 else 0.0,
         0.015 * math.sin(now * 0.3) * trails, 0.004 * hue_amt)
    _uni(op("trails"), 1, cx, cy, aspect)
    S["seed"] = now if fxg * glitch > 0.05 else S["seed"]
    post = op("post")
    _uni(post, 0, mixa if trails > 0.01 else 0.0, m(1.2 * bloom, fxb), flash * react * 0.6, 0.12 * kick * react)
    _uni(post, 1, m(6.0 * chroma, fxc), 0.8, 1.25 * shock, shk * (1.0 - shock) * react)
    _uni(post, 2, glitch * fxg * react, 6.0 + 30.0 * ((now * 3.7) % 1.0), S["seed"], 0.5 * glitch * fxg)
    _uni(post, 3, m(warp, min(1.0, fxw + 0.3)), 3.0, now * 0.3, 0.0 if kal < 0.05 else float(round(2.0 + 10.0 * kal)))
    _uni(post, 4, grain, vig, edges, hud)
    _uni(post, 5, cx, cy, aspect, now)
    _uni(post, 6, shx, shy, ssize, 1.0 if vis > 0.5 else 0.0)
    shaker = op("shake")
    if shaker is not None:                                   # a camera shake on the kick
        a = 0.004 * shake * react
        _set(shaker, "tx", a * math.sin(now * 53.0))
        _set(shaker, "ty", a * math.cos(now * 47.0))
    # HUD text
    ov = op("hud_over")
    if ov is not None:
        ov.bypass = hud < 0.01
    if hud >= 0.01 and now - S["hud"] > 0.2:
        S["hud"] = now
        t = op("hud")
        if t is not None:
            org = text("organism", "myrmex").upper()
            reg = text("regime", "").upper()
            line1 = org + ("  /  " + reg if reg else "")
            line2 = "%s   %.0f BPM   E %s" % (text("intent", "").upper(), ch("bpm", 120.0),
                                             "|" * int(round(8 * energy)) + "." * (8 - int(round(8 * energy))))
            _set(t, "text", line1 + "\\n" + line2)
            _set(t, "fontalpha", hud)
    # recording (ProRes .mov)
    rec = (ch("rec") > 0.5) if (comp.par.Follow.eval() and on) else bool(comp.par.Record.eval())
    mfo = op("rec")
    if mfo is not None and rec != S["was_rec"]:
        S["was_rec"] = rec
        if rec:
            path = text("recfile", "") or os.path.join(os.path.expanduser(str(comp.par.Recdir.eval())),
                                                        time.strftime("Myrmex_TD_%Y%m%d_%H%M%S.mov"))
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            except OSError:
                pass
            _set(mfo, "file", path)
        _set(mfo, "record", rec)
    # the output window (fullscreen on another display)
    want = (ch("window") > 0.5) if (comp.par.Follow.eval() and on) else bool(comp.par.Window.eval())
    win = op("window")
    if win is not None:
        try:
            if want and not win.isOpen:
                _set(win, "monitor", int(ch("monitor", 1.0) if on else comp.par.Monitor.eval()))
                win.par.winopen.pulse()
            elif not want and win.isOpen:
                win.par.winclose.pulse()
        except Exception:
            pass
    so = op("syphon_out")
    if so is not None:
        _set(so, "active", bool(comp.par.Syphonout.eval()))
        _set(so, "sendername", str(comp.par.Outname.eval()))
    # hello to Myrmex (the app shows "TouchDesigner connected")
    if now - S["hb"] > 0.5:
        S["hb"] = now
        try:
            op("to_myrmex").sendOSC("/myrmex/td/alive", [float(S["fps"])])
        except Exception:
            pass
    # find Blender's Syphon picture even if Blender started later
    if now - S["sender"] > 2.0:
        S["sender"] = now
        find_sender()


def find_sender():
    comp = parent()
    si = op("syphon_in")
    if si is None:
        return
    want = str(comp.par.Sender.eval()) or "Myrmex"
    p = si.par.sendername
    try:
        names = [str(n) for n in p.menuNames]
    except Exception:
        names = []
    match = next((n for n in names if n == want), None) or \\
        next((n for n in names if want.lower() in n.lower() and "fx" not in n.lower()), None)
    if match and str(p.eval()) != match:
        p.val = match
    elif not match and not str(p.eval()):
        p.val = want


def send_event(name):
    try:
        op("to_myrmex").sendOSC("/myrmex/trigger", ["creature:" + str(name).lower()])
    except Exception:
        pass
'''

PAREXEC = '''def onPulse(par):
    m = op("fx_logic").module
    if par.name == "Send":
        m.send_event(parent().par.Event.eval())
    elif par.name == "Morph":
        m.send_event("MORPHOLOGY_SHIFT")
    elif par.name == "Cut":
        op("to_myrmex").sendOSC("/myrmex/trigger", ["camera"])
    elif par.name == "Resettrails":
        op("fb").par.resetpulse.pulse()
    elif par.name == "Findblender":
        m.find_sender()
    return


def onValueChange(par, prev):
    if par.name == "Preset":
        op("fx_logic").module.apply_preset(par.eval())
    return
'''

TICK = '''def onFrameStart(frame):
    op("fx_logic").module.update()
    return
'''

TEXT_CALLBACKS = '''def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    if address.startswith("/myrmex/text/") and args:
        key = address.split("/")[-1]
        t = op("text")
        if t.row(key) is None:
            t.appendRow([key, str(args[0])])
        else:
            t[key, 1] = str(args[0])
    return
'''

README = """Myrmex FX (built by myrmex_td.py v%d)

Myrmex (the app) sends the music, the organism and the camera to data_in (OSC %d) and texts to text_in (%d).
Blender sends its picture over Syphon ("Myrmex") to syphon_in.
fx_logic turns it all into the shader uniforms every frame - edit it freely.
Parameters of this component (select it, press P): Myrmex (link, source, output, recording),
FX (amounts - used when Follow is off; with Follow on the Myrmex app's FX rack drives them), Events.
OUT is the final picture: the window (Window toggle), Syphon "Myrmex FX" (Syphon out), ProRes recording.
"""


# ---------------------------------------------------------------------------------------------- the build
def build(root=None, save=True):
    import td
    warnings = []
    parent = root or td.op(PARENT)
    if parent is None:
        raise RuntimeError(PARENT + " not found: open a new project first")
    old = parent.op(NAME)
    kept = {}
    if old is not None:                                          # keep what you had set
        for p in getattr(old, "customPars", []):
            try:
                kept[p.name] = p.eval()
            except Exception:
                pass
        old.destroy()
    comp = parent.create(_t(td, "baseCOMP"), NAME)
    comp.nodeX, comp.nodeY = 0, 0

    def mk(kind, name, x, y):
        o = comp.create(_t(td, kind), name)
        o.nodeX, o.nodeY = x * 180, -y * 140
        return o

    def setp(o, **pars):
        for k, v in pars.items():
            try:
                setattr(o.par, k, v)
            except Exception as e:
                warnings.append("%s.%s: %s" % (o.name, k, e))

    def text_dat(name, text, x, y):
        d = mk("textDAT", name, x, y)
        d.text = text
        return d

    # --- data in
    data_in = mk("oscinCHOP", "data_in", 0, 0)
    setp(data_in, port=DATA_PORT, stripsegments=1, active=True)
    text_tab = mk("tableDAT", "text", 1, 0)
    text_tab.clear()
    text_tab.appendRow(["key", "value"])
    for key in ("organism", "regime", "intent", "shot", "preset", "recfile", "take"):
        text_tab.appendRow([key, ""])
    text_dat("text_callbacks", TEXT_CALLBACKS, 2, 1)
    text_in = mk("oscinDAT", "text_in", 2, 0)
    setp(text_in, port=TEXT_PORT, callbacks="text_callbacks", maxlines=20, active=True)
    to_myrmex = mk("oscoutDAT", "to_myrmex", 3, 0)
    setp(to_myrmex, address="127.0.0.1", port=MYRMEX_PORT, active=True)
    # --- sources
    syphon_in = mk("syphonspoutinTOP", "syphon_in", 0, 2)
    setp(syphon_in, sendername="Myrmex")
    test = mk("noiseTOP", "test", 0, 3)
    setp(test, outputresolution="custom", resolutionw=WIDTH, resolutionh=HEIGHT, period=1.6, amp=0.6, offset=0.25,
         mono=False)
    try:
        test.par.tz.expr = "absTime.seconds * 0.25"
        test.par.tz.mode = td.ParMode.EXPRESSION
    except Exception as e:
        warnings.append("test.tz: %s" % e)
    src = mk("switchTOP", "src", 1, 2)
    src.setInputs([syphon_in, test])
    shake = mk("transformTOP", "shake", 2, 2)
    shake.setInputs([src])
    setp(shake, extend="mirror")
    # --- shaders
    for i, (name, code) in enumerate(SHADERS.items()):
        text_dat(name, code, 1 + i, 5)

    def glsl(name, pixel, inputs, uniforms, x, y, fmt=None):
        g = mk("glslTOP", name, x, y)
        g.setInputs(inputs)
        setp(g, pixeldat=pixel, outputresolution="custom", resolutionw=WIDTH, resolutionh=HEIGHT)
        if fmt:
            setp(g, format=fmt)
        try:
            g.seq.vec.numBlocks = len(uniforms)
        except Exception as e:
            warnings.append("%s vectors: %s" % (name, e))
        for i, u in enumerate(uniforms):
            setp(g, **{"vec%dname" % i: u})
        return g

    grade = glsl("grade", "shader_grade", [shake], ["uGrade", "uMisc"], 3, 2)
    bright = glsl("bright", "shader_bright", [grade], ["uBright"], 4, 3)
    blur_a = mk("blurTOP", "blur_a", 5, 3)
    blur_a.setInputs([bright])
    setp(blur_a, size=10, outputresolution="half", preshrink=1)
    blur_b = mk("blurTOP", "blur_b", 6, 3)
    blur_b.setInputs([blur_a])
    setp(blur_b, size=30, outputresolution="half", preshrink=1)
    fb = mk("feedbackTOP", "fb", 4, 1)
    fb.setInputs([grade])
    trails = glsl("trails", "shader_trails", [grade, fb], ["uTrail", "uCenter"], 5, 1, fmt="rgba16float")
    trail_out = mk("nullTOP", "trail_out", 6, 1)
    trail_out.setInputs([trails])
    setp(fb, top="trail_out", format="rgba16float")
    post = glsl("post", "shader_post", [trail_out, grade, blur_a, blur_b],
                ["uMixB", "uChroma", "uGlitch", "uWarp", "uLook", "uCenter", "uShock"], 7, 2)
    hud = mk("textTOP", "hud", 7, 4)
    setp(hud, outputresolution="custom", resolutionw=WIDTH, resolutionh=HEIGHT, text="MYRMEX", fontsizex=20,
         alignx="left", aligny="bottom", position1=28, position2=24, fontcolorr=0.7, fontcolorg=0.95,
         fontcolorb=1.0, bgalpha=0.0)
    hud_over = mk("overTOP", "hud_over", 8, 2)
    hud_over.setInputs([hud, post])
    hud_over.bypass = True
    out = mk("nullTOP", "OUT", 9, 2)
    out.setInputs([hud_over])
    try:
        out.viewer = True
    except Exception:
        pass
    # --- outputs
    so = mk("syphonspoutoutTOP", "syphon_out", 10, 1)
    so.setInputs([out])
    setp(so, sendername="Myrmex FX", active=False)
    rec = mk("moviefileoutTOP", "rec", 10, 2)
    rec.setInputs([out])
    setp(rec, type="movie", record=False)
    try:                                                        # ProRes on a Mac (H.264 needs a paid licence)
        names = [str(n) for n in rec.par.videocodec.menuNames]
        pick = next((n for n in names if "prores" in n.lower()), None) or \
            next((n for n in names if "jpeg" in n.lower() or "mjp" in n.lower()), None)
        if pick:
            rec.par.videocodec = pick
    except Exception as e:
        warnings.append("rec.videocodec: %s" % e)
    win = mk("windowCOMP", "window", 10, 3)
    setp(win, winop="OUT", borders=False, size="fill", justifyh="center", justifyv="center", monitor=1,
         alwaysontop=True)
    # --- logic
    text_dat("fx_logic", LOGIC.replace("__FX_DEFAULTS__", repr(FX_DEFAULTS)).replace("__FX__", repr(FX))
             .replace("__PRESETS__", repr(PRESETS)), 4, -1)
    tick = mk("executeDAT", "tick", 5, -1)
    tick.text = TICK
    setp(tick, framestart=True, active=True)
    pe = mk("parameterexecuteDAT", "pars_exec", 6, -1)
    pe.text = PAREXEC
    setp(pe, op="..", pars="*", custom=True, builtin=False, onpulse=True, valuechange=True, active=True)
    text_dat("README", README % (VERSION, DATA_PORT, TEXT_PORT), 0, -1)
    # --- parameters of the component
    pg = comp.appendCustomPage("Myrmex")
    _tog(pg, "Follow", "Follow the Myrmex app", True)
    _menu(pg, "Source", "Picture", ["syphon", "test"], ["Blender (Syphon)", "Test pattern"], "syphon")
    _str(pg, "Sender", "Syphon from Blender", "Myrmex")
    _tog(pg, "Flip", "Flip the picture", False)
    _int(pg, "Width", "Width", WIDTH, 320, 1280)
    _int(pg, "Height", "Height", HEIGHT, 180, 1280)
    _tog(pg, "Window", "Output window", False)
    _int(pg, "Monitor", "Window on display", 1, 0, 4)
    _tog(pg, "Syphonout", "Syphon out", False)
    _str(pg, "Outname", "Syphon out name", "Myrmex FX")
    _tog(pg, "Record", "Record (ProRes .mov)", False)
    _folder(pg, "Recdir", "Recordings folder", "~/Myrmex/td_recordings")
    _pulse(pg, "Resettrails", "Clear trails")
    _pulse(pg, "Findblender", "Find Blender picture")
    fxp = comp.appendCustomPage("FX")
    _menu(fxp, "Preset", "Preset", list(PRESETS), list(PRESETS), "Neon Trails")
    for k in FX:
        _float(fxp, k.capitalize(), k.capitalize(), FX_DEFAULTS[k])
    ev = comp.appendCustomPage("Events")
    _menu(ev, "Event", "Event", list(EVENTS), [e.replace("_", " ").title() for e in EVENTS], "IMPULSE")
    _pulse(ev, "Send", "Send to the organism")
    _pulse(ev, "Morph", "Change form")
    _pulse(ev, "Cut", "Camera cut")
    for name, val in kept.items():                              # your settings, back
        try:
            setattr(comp.par, name, val)
        except Exception:
            pass
    try:
        comp.par.nodeview = "opviewer"
        comp.par.opviewer = "OUT"
        comp.viewer = True
    except Exception:
        pass
    try:
        td.project.cookRate = 60
    except Exception:
        pass
    saved = ""
    if save:
        try:
            os.makedirs(os.path.dirname(SAVE_AS), exist_ok=True)
            td.project.save(SAVE_AS)
            saved = SAVE_AS
        except Exception as e:
            warnings.append("save: %s" % e)
    print("Myrmex FX v%d built in %s%s" % (VERSION, comp.path, ("; saved " + saved) if saved else ""))
    for w in warnings:
        print("  (note) " + w)
    return comp, warnings


def _t(td, name):
    t = getattr(td, name, None)
    if t is None:
        raise RuntimeError("this TouchDesigner has no " + name)
    return t


def _first(group):
    try:
        return group[0]
    except TypeError:
        return group


def _float(page, name, label, default, lo=0.0, hi=1.0):
    p = _first(page.appendFloat(name, label=label))
    for k, v in (("normMin", lo), ("normMax", hi), ("min", lo), ("max", hi), ("clampMin", True), ("clampMax", True),
                 ("default", default)):
        try:
            setattr(p, k, v)
        except Exception:
            pass
    p.val = default
    return p


def _int(page, name, label, default, lo, hi):
    p = _first(page.appendInt(name, label=label))
    for k, v in (("normMin", lo), ("normMax", hi), ("min", lo), ("max", hi), ("clampMin", True), ("default", default)):
        try:
            setattr(p, k, v)
        except Exception:
            pass
    p.val = default
    return p


def _tog(page, name, label, default):
    p = _first(page.appendToggle(name, label=label))
    try:
        p.default = default
    except Exception:
        pass
    p.val = default
    return p


def _str(page, name, label, default):
    p = _first(page.appendStr(name, label=label))
    try:
        p.default = default
    except Exception:
        pass
    p.val = default
    return p


def _folder(page, name, label, default):
    p = _first(page.appendFolder(name, label=label))
    p.val = default
    return p


def _menu(page, name, label, names, labels, default):
    p = _first(page.appendMenu(name, label=label))
    p.menuNames = list(names)
    p.menuLabels = list(labels)
    try:
        p.default = default
    except Exception:
        pass
    p.val = default
    return p


def _pulse(page, name, label):
    return _first(page.appendPulse(name, label=label))


try:                                                             # inside TouchDesigner: build now
    import td as _td_module                                      # noqa: F401
    if __name__ != "myrmex_td_under_test":
        build()
except ImportError:                                              # outside TouchDesigner (tests)
    pass
