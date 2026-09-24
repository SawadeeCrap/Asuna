"""Cinematic output: shot cameras, travelling studio lighting, look-dev materials, render presets."""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from myrmex.camera.cinematographer import CameraTrack

from . import compat


# ============================================================================ fast keyframing
def _key_channels(owner: bpy.types.ID, action_name: str, channels: list[tuple[str, int, np.ndarray]],
                  frames: np.ndarray, interpolation: int = 1) -> None:
    act = compat.new_action_for(owner, action_name)
    for path, idx, vals in channels:
        fc = compat.ensure_fcurve(act, owner, path, idx, "Myrmex")
        n = len(frames)
        fc.keyframe_points.add(n)
        co = np.empty(2 * n)
        co[0::2] = frames
        co[1::2] = vals
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [interpolation] * n)
        fc.update()


def _look_eulers(pos: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    out = np.zeros((len(pos), 3))
    prev = None
    for i in range(len(pos)):
        d = Vector(tgt[i]) - Vector(pos[i])
        q = d.to_track_quat("-Z", "Y")
        e = q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")
        out[i] = (e.x, e.y, e.z)
        prev = e
    return out


# ============================================================================ cameras
def apply_camera_track(track: CameraTrack, frame_start: int = 1, prefix: str = "Shot",
                       sensor: float = 36.0, collection: bpy.types.Collection | None = None) -> list[bpy.types.Object]:
    sc = bpy.context.scene
    coll = collection or bpy.data.collections.get("MyrmexCameras")
    if coll is None:
        coll = bpy.data.collections.new("MyrmexCameras")
        sc.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(sc.timeline_markers):
        if m.name.startswith(prefix):
            sc.timeline_markers.remove(m)
    cams = []
    T = len(track.positions)
    for i, sh in enumerate(track.shots):
        a = max(0, sh.start - 1)
        b = min(T, sh.end + 1)
        if b - a < 2:
            continue
        cd = bpy.data.cameras.new(f"{prefix}_{i + 1:02d}_{sh.kind}")
        cd.sensor_width = sensor
        cd.dof.use_dof = True
        cam = bpy.data.objects.new(cd.name, cd)
        coll.objects.link(cam)
        cam.rotation_mode = "XYZ"
        frames = np.arange(a, b, dtype=float) + frame_start
        eul = _look_eulers(track.positions[a:b], track.targets[a:b])
        _key_channels(cam, cam.name + "_act",
                      [("location", 0, track.positions[a:b, 0]), ("location", 1, track.positions[a:b, 1]),
                       ("location", 2, track.positions[a:b, 2]), ("rotation_euler", 0, eul[:, 0]),
                       ("rotation_euler", 1, eul[:, 1]), ("rotation_euler", 2, eul[:, 2])], frames)
        _key_channels(cd, cd.name + "_act",
                      [("lens", 0, track.lens[a:b]), ("dof.focus_distance", 0, track.focus[a:b]),
                       ("dof.aperture_fstop", 0, track.fstop[a:b])], frames)
        mk = sc.timeline_markers.new(f"{prefix}_{i + 1:02d}", frame=int(sh.start + frame_start))
        mk.camera = cam
        cams.append(cam)
    if cams:
        sc.camera = cams[0]
    return cams


# ============================================================================ materials
MATERIALS = {
    "black_chrome": {"Base Color": (0.018, 0.018, 0.02, 1), "Metallic": 1.0, "Roughness": 0.12,
                     "Coat Weight": 0.6, "Coat Roughness": 0.03, "Specular IOR Level": 0.5},
    "chrome": {"Base Color": (0.92, 0.92, 0.93, 1), "Metallic": 1.0, "Roughness": 0.05},
    "liquid_metal": {"Base Color": (0.55, 0.57, 0.6, 1), "Metallic": 1.0, "Roughness": 0.07, "bump": 0.15},
    "gunmetal": {"Base Color": (0.2, 0.21, 0.23, 1), "Metallic": 1.0, "Roughness": 0.28, "Anisotropic": 0.4},
    "ceramic": {"Base Color": (0.9, 0.9, 0.88, 1), "Metallic": 0.0, "Roughness": 0.25, "Coat Weight": 0.8,
                "Coat Roughness": 0.08},
    "clay": {"Base Color": (0.72, 0.72, 0.74, 1), "Metallic": 0.0, "Roughness": 0.6},
    "iridescent": {"Base Color": (0.1, 0.1, 0.12, 1), "Metallic": 1.0, "Roughness": 0.1,
                   "Thin Film Thickness": 420.0, "Thin Film IOR": 1.45},
}


def make_material(preset: str, name: str | None = None) -> bpy.types.Material:
    spec = MATERIALS[preset]
    mat = bpy.data.materials.new(name or f"Myrmex_{preset}")
    nt = compat.material_node_tree(mat)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    for k, v in spec.items():
        if k == "bump":
            continue
        if k in bsdf.inputs:
            try:
                bsdf.inputs[k].default_value = v
            except (TypeError, ValueError):
                pass
    if spec.get("bump"):
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 18.0
        noise.inputs["Detail"].default_value = 3.0
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = float(spec["bump"])
        bump.inputs["Distance"].default_value = 0.002
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def apply_material(obj: bpy.types.Object, preset: str, keep_textures: bool = False) -> bpy.types.Material:
    if keep_textures and obj.data.materials and any(m and m.node_tree and any(n.type == "TEX_IMAGE"
                                                                            for n in m.node_tree.nodes)
                                                   for m in obj.data.materials):
        return obj.data.materials[0]
    mat = make_material(preset)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    return mat


# ============================================================================ surface
def smooth_surface(obj: bpy.types.Object, iterations: int = 6, strength: float = 0.5) -> None:
    """Remove generator noise from the rest mesh (Laplacian smooth, volume preserving), in place.

    Marching-cubes surfaces from image-to-3D models carry millimetre ripples that read as
    crumpled foil under mirror-like materials.  Smoothing the *rest* mesh keeps topology,
    vertex groups and weights untouched, so it can be done at any point before rendering.
    """
    if iterations <= 0:
        return
    saved = [(m, m.show_viewport) for m in obj.modifiers]
    for m, _ in saved:
        m.show_viewport = False
    mod = obj.modifiers.new("MyrmexSmooth", "LAPLACIANSMOOTH")
    mod.iterations = iterations
    mod.lambda_factor = strength
    mod.lambda_border = 0.0
    mod.use_volume_preserve = True
    mod.use_normalized = True
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    n = len(obj.data.vertices)
    co = np.empty(3 * n)
    ev.data.vertices.foreach_get("co", co)
    obj.modifiers.remove(mod)
    for m, vis in saved:
        m.show_viewport = vis
    obj.data.vertices.foreach_set("co", co)
    obj.data.update()


# ============================================================================ studio
def _emission_material(name: str, strength: float, color=(1, 1, 1, 1)) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    nt = compat.material_node_tree(mat)
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    return mat


def floor_material(color=(0.62, 0.62, 0.64), panel: float = 2.0) -> bpy.types.Material:
    """Polished concrete with panel seams.

    A travelling camera needs something on the floor to read the forward motion against:
    the seams and the low-contrast mottling slide through the frame with every step.
    """
    mat = bpy.data.materials.new("MyrmexFloorMat")
    nt = compat.material_node_tree(mat)
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeNewGeometry")        # world-space position: the floor can travel
    base = (*color, 1.0)
    mottle = nt.nodes.new("ShaderNodeTexNoise")
    mottle.inputs["Scale"].default_value = 1.6
    mottle.inputs["Detail"].default_value = 6.0
    mottle.inputs["Roughness"].default_value = 0.62
    nt.links.new(tc.outputs["Position"], mottle.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = tuple(c * 0.9 for c in color) + (1.0,)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = tuple(min(1.0, c * 1.06) for c in color) + (1.0,)
    nt.links.new(mottle.outputs["Fac"], ramp.inputs["Fac"])
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Brick Width"].default_value = panel
    brick.inputs["Row Height"].default_value = panel
    brick.inputs["Mortar Size"].default_value = 0.012
    brick.inputs["Mortar Smooth"].default_value = 0.4
    brick.inputs["Color1"].default_value = base
    brick.inputs["Color2"].default_value = base
    nt.links.new(tc.outputs["Position"], brick.inputs["Vector"])
    seam = nt.nodes.new("ShaderNodeMix")
    seam.data_type = "RGBA"
    seam.inputs["B"].default_value = tuple(c * 0.55 for c in color) + (1.0,)
    nt.links.new(brick.outputs["Fac"], seam.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], seam.inputs["A"])
    nt.links.new(seam.outputs["Result"], b.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.22
    rough.inputs["To Max"].default_value = 0.42
    nt.links.new(mottle.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = 0.35
    return mat


def setup_studio(subject_path: np.ndarray, headings: np.ndarray, frame_start: int = 1,
                 floor_color=(0.5, 0.5, 0.52), world_color=(0.72, 0.72, 0.74), height: float = 1.7,
                 key_power: float = 900.0, panel: float = 2.0) -> bpy.types.Object:
    """Light-grey infinite studio; the light rig travels with the subject (keyframed empty)."""
    sc = bpy.context.scene
    coll = bpy.data.collections.get("MyrmexStudio")
    if coll is None:
        coll = bpy.data.collections.new("MyrmexStudio")
        sc.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    # Floor.
    me = bpy.data.meshes.new("MyrmexFloor")
    s = 400.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("MyrmexFloor", me)
    coll.objects.link(floor)
    me.materials.append(floor_material(floor_color, panel))
    # World.
    world = sc.world or bpy.data.worlds.new("MyrmexWorld")
    sc.world = world
    wnt = compat.world_node_tree(world)
    bg = next((n for n in wnt.nodes if n.type == "BACKGROUND"), None)
    if bg is not None:
        bg.inputs["Color"].default_value = (*world_color, 1)
        bg.inputs["Strength"].default_value = 0.8
    # Light rig following the subject.
    rig = bpy.data.objects.new("MyrmexLightRig", None)
    coll.objects.link(rig)
    k = height / 1.7

    def area(name, loc, rot, size, power, color=(1, 1, 1), shape="RECTANGLE", size_y=None):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = power
        ld.shape = shape
        ld.size = size
        if size_y is not None:
            ld.size_y = size_y
        ld.color = color
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        ob.parent = rig
        ob.location = Vector(loc) * k
        ob.rotation_euler = Euler(rot)
        return ob

    # Rig frame: +X = subject forward, +Y = subject left.
    area("Key", (2.2, 1.8, 3.6), (math.radians(40), 0, math.radians(125)), 3.0 * k, key_power)
    area("Fill", (2.8, -2.6, 1.8), (math.radians(70), 0, math.radians(45)), 4.0 * k, key_power * 0.28)
    area("Rim", (-2.6, 0.6, 2.6), (math.radians(55), 0, math.radians(-100)), 4.0 * k, key_power * 0.9,
         color=(0.95, 0.97, 1.0), size_y=0.6 * k)
    area("Top", (0.0, 0.0, 5.0), (0, 0, 0), 5.0 * k, key_power * 0.3)
    # Reflection cards (invisible to camera) for metals.
    for name, loc, rot in (("CardL", (0.8, 3.0, 1.6), (math.radians(90), 0, 0)),
                           ("CardR", (0.8, -3.0, 1.6), (math.radians(-90), 0, 0))):
        cm = bpy.data.meshes.new(name)
        w, h = 1.2 * k, 3.2 * k
        cm.from_pydata([(-w, -h / 2, 0), (w, -h / 2, 0), (w, h / 2, 0), (-w, h / 2, 0)], [], [(0, 1, 2, 3)])
        cm.materials.append(_emission_material(name + "Mat", 3.0))
        card = bpy.data.objects.new(name, cm)
        coll.objects.link(card)
        card.parent = rig
        card.location = Vector(loc) * k
        card.rotation_euler = Euler(rot)
        card.visible_camera = False
        card.visible_shadow = False
    frames = np.arange(len(subject_path), dtype=float) + frame_start
    yaw = np.unwrap(np.arctan2(headings[:, 1], headings[:, 0]))
    _key_channels(rig, "MyrmexLightRig_act",
                  [("location", 0, subject_path[:, 0]), ("location", 1, subject_path[:, 1]),
                   ("location", 2, np.zeros(len(subject_path))), ("rotation_euler", 2, yaw)], frames)
    return rig


# ============================================================================ render presets
def configure_render(preset: str = "eevee", resolution=(1920, 1080), fps: float = 30.0,
                     motion_blur: bool = True, samples: int | None = None, shadows: str = "all") -> None:
    """Render presets.  ``shadows="key"`` keeps only the key light's shadow (much cheaper in EEVEE)."""
    sc = bpy.context.scene
    r = sc.render
    studio = bpy.data.collections.get("MyrmexStudio")
    if studio is not None:
        for ob in studio.objects:
            if ob.type == "LIGHT":
                ob.data.use_shadow = shadows == "all" or ob.name == "Key"
            elif ob.name.startswith("Card"):
                # Workbench ignores ray visibility: the reflection cards would block the camera.
                ob.hide_render = preset == "workbench"
    r.resolution_x, r.resolution_y = resolution
    r.resolution_percentage = 100
    r.fps = int(round(fps))
    r.use_motion_blur = motion_blur
    if hasattr(r, "motion_blur_shutter"):
        r.motion_blur_shutter = 0.5
    compat.set_view_transform(sc, "AgX", "Medium High Contrast")
    if preset in ("eevee", "eevee_preview"):
        r.engine = compat.eevee_engine_id()
        ee = sc.eevee
        ee.taa_render_samples = samples or (16 if preset == "eevee_preview" else 64)
        final = preset == "eevee"
        for attr, val in (("use_raytracing", final), ("use_shadows", True), ("use_fast_gi", final),
                          ("shadow_resolution_scale", 1.0 if final else 0.5),
                          ("shadow_step_count", 6 if final else 2), ("shadow_ray_count", 1)):
            if hasattr(ee, attr):
                try:
                    setattr(ee, attr, val)
                except Exception:
                    pass
        if hasattr(ee, "ray_tracing_options"):
            try:
                ee.ray_tracing_options.resolution_scale = "2"
            except Exception:
                pass
    elif preset in ("cycles", "cycles_preview"):
        r.engine = "CYCLES"
        cy = sc.cycles
        cy.samples = samples or (48 if preset == "cycles_preview" else 256)
        cy.use_adaptive_sampling = True
        cy.adaptive_threshold = 0.02 if preset == "cycles_preview" else 0.01
        cy.use_denoising = True
        cy.max_bounces = 8
        cy.glossy_bounces = 6
        cy.caustics_reflective = False
        cy.caustics_refractive = False
        cy.sample_clamp_indirect = 8.0
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dev in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
                try:
                    prefs.compute_device_type = dev
                    prefs.get_devices()
                    if any(d.type == dev for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = True
                        cy.device = "GPU"
                        break
                except TypeError:
                    continue
        except Exception:
            pass
    elif preset == "workbench":
        r.engine = "BLENDER_WORKBENCH"
        shd = sc.display.shading
        shd.light = "STUDIO"
        shd.color_type = "MATERIAL"
        shd.show_shadows = False
        shd.show_cavity = False
