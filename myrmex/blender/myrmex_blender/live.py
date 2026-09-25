"""Blender live link: poses streamed by ``myrmex live`` drive the armature in real time.

Start it from the sidebar (View3D > Myrmex > Live) or from Python::

    from myrmex_blender import live
    link = live.LiveLink(bpy.data.objects["Rig"])
    link.start()                      # polls UDP 9101 from a bpy.app.timers callback

The engine may run in another process (``myrmex live --rig rig.json``) or inside
Blender (``live.start_embedded_engine(...)``) - both talk over the same socket.

Every packet carries rest-world bone deltas; :class:`BasisSolver` converts them
to ``matrix_basis`` with the armature's own rest matrices (vectorised, ~0.1 ms).
The live camera, the travelling light rig and the floor follow the subject, so
the walk can go on for hours.
"""
from __future__ import annotations

import socket
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

from myrmex.realtime.protocol import Reassembler, decode_names, decode_pose

LIVE_CAMERA = "MyrmexLiveCam"
_ACTIVE: dict = {"link": None}


@bpy.app.handlers.persistent
def _data_reloaded(*_args):
    """Undo / redo / opening a file re-creates Blender's data: the running link re-finds its objects."""
    link = _ACTIVE.get("link")
    if link is not None:
        link.refresh()


_HANDLERS = ("undo_post", "redo_post", "load_post")


class BasisSolver:
    """Rest-world deltas (packet order) -> pose-bone ``matrix_basis`` (armature order)."""

    def __init__(self, arm_obj: bpy.types.Object, names: list[str]):
        bones = arm_obj.data.bones
        self.names = [n for n in names if n in bones]
        if not self.names:
            raise ValueError("none of the streamed bones exists in the armature")
        self.src = np.array([names.index(n) for n in self.names])
        self.Mobj = np.array(arm_obj.matrix_world)
        self.Mobj_inv = np.linalg.inv(self.Mobj)
        R = np.array([np.array(bones[n].matrix_local) for n in self.names])
        self.R = R
        self.Rinv = np.linalg.inv(R)
        pos = {n: j for j, n in enumerate(self.names)}
        self.parent = np.full(len(self.names), -1)
        Rp = np.tile(np.eye(4), (len(self.names), 1, 1))
        self.has_parent = np.zeros(len(self.names), bool)
        for j, n in enumerate(self.names):
            b = bones[n]
            if b.parent is not None:
                self.has_parent[j] = True
                Rp[j] = np.array(b.parent.matrix_local)
                self.parent[j] = pos.get(b.parent.name, -1)
        self.A = self.Rinv @ Rp                                      # Rest^-1 · Rest_parent
        self.MR = self.Mobj @ R                                      # Mobj · Rest
        # Parents that are not streamed stay at rest: P_parent = Rest_parent.
        self.Pp_rest_inv = np.linalg.inv(Rp)

    def solve(self, deltas: np.ndarray) -> np.ndarray:
        D = deltas[self.src]
        P = self.Mobj_inv @ D @ self.MR                            # posed, armature space
        basis = np.empty_like(P)
        root = ~self.has_parent
        basis[root] = self.Rinv[root] @ P[root]
        ch = self.has_parent
        Pp_inv = self.Pp_rest_inv.copy()
        driven = self.parent >= 0
        if driven.any():
            Pp_inv[driven] = np.linalg.inv(P[self.parent[driven]])
        basis[ch] = self.A[ch] @ Pp_inv[ch] @ P[ch]
        return basis


def _tag_redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class LiveLink:
    def __init__(self, arm_obj: bpy.types.Object, port: int = 9101, host: str = "127.0.0.1",
                 camera: bool = True, lights: bool = True, floor: bool = True, fast_viewport: bool = True):
        self.arm = arm_obj
        self.arm_name = arm_obj.name if arm_obj is not None else None
        self.fast_viewport = fast_viewport
        self.creature_view = None
        self._restore: list[tuple] = []
        self.port, self.host = port, host
        self.use_camera, self.use_lights, self.use_floor = camera, lights, floor
        self.sock: socket.socket | None = None
        self.names: list[str] | None = None
        self.rig: int | None = None
        self.solver: BasisSolver | None = None
        self.last = None
        self.running = False
        self.stats = {"packets": 0, "applied": 0, "fps": 0.0, "age_ms": 0.0, "dropped": 0, "error": ""}
        self._fps_t, self._fps_n = time.perf_counter(), 0
        self._last_seq = None
        self.reasm = Reassembler()                 # big frames arrive in fragments

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind((self.host, self.port))
        self.sock.setblocking(False)
        for pb in (self.arm.pose.bones if self.arm is not None else ()):
            pb.rotation_mode = "QUATERNION"
        if self.arm is not None and self.arm.animation_data is not None:
            self.arm.animation_data.action = None          # baked actions would fight the stream
        if self.fast_viewport and self.arm is not None:
            # Corrective Smooth costs tens of ms per frame on 150k vertices; the Armature modifier
            # (preserve volume) alone deforms in a few ms.  Render keeps the smoothing.
            for ob in bpy.data.objects:
                if ob.type == "MESH" and ob.parent == self.arm:
                    for m in ob.modifiers:
                        if m.type == "CORRECTIVE_SMOOTH" and m.show_viewport:
                            self._restore.append((ob.name, m.name))
                            m.show_viewport = False
        self.running = True
        _ACTIVE["link"] = self
        for h in _HANDLERS:
            lst = getattr(bpy.app.handlers, h)
            if _data_reloaded not in lst:
                lst.append(_data_reloaded)
        if not bpy.app.timers.is_registered(self._timer):
            bpy.app.timers.register(self._timer, first_interval=0.0, persistent=True)

    def refresh(self) -> None:
        """Drop every cached Blender reference; they are looked up again by name on the next packet."""
        self.creature_view = None
        self.solver = None
        if self.arm_name is not None or self.arm is not None:
            s = getattr(bpy.context.scene, "myrmex_live", None)
            arm = bpy.data.objects.get(self.arm_name or "") or (s.armature if s is not None else None) or next(
                (o for o in bpy.data.objects if o.type == "ARMATURE" and o.get("myrmex_rig")), None)
            self.arm = arm
            if arm is not None:
                self.arm_name = arm.name
                for pb in arm.pose.bones:
                    pb.rotation_mode = "QUATERNION"
                if arm.animation_data is not None:
                    arm.animation_data.action = None
                if self.names:
                    try:
                        self.solver = BasisSolver(arm, self.names)
                    except ValueError:
                        self.solver = None
        self.stats["error"] = ""

    def stop(self) -> None:
        self.running = False
        if _ACTIVE.get("link") is self:
            _ACTIVE["link"] = None
            for h in _HANDLERS:
                lst = getattr(bpy.app.handlers, h)
                if _data_reloaded in lst:
                    lst.remove(_data_reloaded)
        if bpy.app.timers.is_registered(self._timer):
            bpy.app.timers.unregister(self._timer)
        for ob_name, m_name in self._restore:
            ob = bpy.data.objects.get(ob_name)
            if ob is not None and m_name in ob.modifiers:
                ob.modifiers[m_name].show_viewport = True
        self._restore = []
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _timer(self):
        if not self.running:
            return None
        try:
            if self.poll():
                _tag_redraw()
        except ReferenceError:                 # objects re-created (undo, file load): find them again
            self.refresh()
        except Exception as e:                 # never kill the timer: report and keep listening
            self.stats["error"] = f"{type(e).__name__}: {e}"
        return 1.0 / 240.0

    # ------------------------------------------------------------------ receive + apply
    def poll(self) -> bool:
        """Drain the socket, apply the newest pose.  Returns True if something changed."""
        if self.sock is None:
            return False
        newest = None
        creature = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            data = self.reasm.feed(data)
            if data is None:
                continue
            if data[:4] == b"MYRC":
                creature = data
                self.stats["packets"] += 1
                continue
            if data[:4] == b"MYRN":
                nm = decode_names(data)
                if nm is not None and self.arm is not None and (self.rig != nm[0] or self.solver is None):
                    self.rig, self.names = nm[0], nm[1]
                    try:
                        self.solver = BasisSolver(self.arm, self.names)
                        self.stats["error"] = ""
                    except ValueError as e:
                        self.solver = None
                        self.stats["error"] = str(e)
                continue
            fr = decode_pose(data)
            if fr is None:
                continue
            self.stats["packets"] += 1
            if newest is not None:
                self.stats["dropped"] += 1
            newest = fr
        if creature is not None:
            return self._apply_creature(creature)
        if newest is None or self.solver is None or newest.rig != self.rig:
            return False
        self.apply(newest)
        return True

    def apply(self, fr) -> None:
        basis = self.solver.solve(fr.deltas)
        pbs = self.arm.pose.bones
        for j, n in enumerate(self.solver.names):
            pbs[n].matrix_basis = Matrix(basis[j].tolist())
        if self.use_camera and fr.camera is not None:
            self._apply_camera(fr.camera)
        if self.use_lights or self.use_floor:
            self._follow(fr)
        self.last = fr
        self.stats["applied"] += 1
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self.stats["fps"] = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0

    def _apply_creature(self, data: bytes) -> bool:
        from myrmex.creature.protocol import decode_creature

        from .creature import CreatureView
        fr = decode_creature(data)
        if fr is None:
            return False
        if self.creature_view is None:
            self.creature_view = CreatureView(bpy.context.scene)
            _clear_take_animation()
        self.creature_view.apply(fr)
        if self.use_camera and fr.camera is not None:
            self._apply_camera(fr.camera)
        if self.use_lights or self.use_floor:
            class _F:                                      # the follow code only needs these
                subject_pos, heading = fr.com, fr.heading
                lift = max(0.0, float(fr.com[2]) - 1.2) if fr.links is not None else 0.0   # flying: lights rise too
            self._follow(_F)
        self.last = fr
        self.stats["applied"] += 1
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self.stats["fps"] = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0
        return True

    def _apply_camera(self, c) -> None:
        sc = bpy.context.scene
        cam = bpy.data.objects.get(LIVE_CAMERA)
        if cam is None:
            cd = bpy.data.cameras.new(LIVE_CAMERA)
            cd.sensor_width = 36.0
            cd.dof.use_dof = True
            cam = bpy.data.objects.new(LIVE_CAMERA, cd)
            sc.collection.objects.link(cam)
        if sc.camera is None or sc.camera.name != LIVE_CAMERA:
            sc.camera = cam
        pos, tgt = Vector(c.position), Vector(c.target)
        d = tgt - pos
        if d.length > 1e-6:
            cam.matrix_world = Matrix.Translation(pos) @ d.to_track_quat("-Z", "Y").to_matrix().to_4x4()
        cam.data.lens = float(c.lens)
        cam.data.dof.focus_distance = float(c.focus)
        cam.data.dof.aperture_fstop = float(c.fstop)

    def _follow(self, fr) -> None:
        p = fr.subject_pos
        if self.use_lights:
            rig = bpy.data.objects.get("MyrmexLightRig")
            if rig is not None:
                if rig.animation_data is not None and rig.animation_data.action is not None:
                    rig.animation_data.action = None      # offline keys would fight the live follow
                rig.location = (float(p[0]), float(p[1]), float(getattr(fr, "lift", 0.0)))
                rig.rotation_euler = (0.0, 0.0, float(fr.heading))
        if self.use_floor:
            floor = bpy.data.objects.get("MyrmexFloor")
            if floor is not None:
                if floor.animation_data is not None and floor.animation_data.action is not None:
                    floor.animation_data.action = None
                # The floor shader works in world space, so the plane can simply travel along.
                floor.location = (float(p[0]), float(p[1]), 0.0)


def _clear_take_animation() -> None:
    """An imported take (keyframes, strut cache) would fight the live stream: switch it off."""
    from .creature import LATTICE, LURE, META, OBSTACLE, PLATES, SWARM
    mb = bpy.data.metaballs.get(META)
    for idb in [mb, mb.materials[0].node_tree if mb is not None and mb.materials and mb.materials[0] else None,
                bpy.data.objects.get(LURE)] + [o for o in bpy.data.objects if o.name.startswith(OBSTACLE)]:
        if idb is not None and idb.animation_data is not None and idb.animation_data.action is not None:
            idb.animation_data.action = None
    for name in (LATTICE, PLATES, SWARM):
        ob = bpy.data.objects.get(name)
        if ob is not None and "TakeCache" in ob.modifiers:
            ob.modifiers.remove(ob.modifiers["TakeCache"])


# ---------------------------------------------------------------------------- embedded engine
_EMBEDDED = {"session": None}


def start_embedded_engine(rig_json: str | None, port: int = 9101, osc_port: int = 9100, clock: str = "auto",
                          midi: list[str] | None = None, bpm: float = 120.0, style: str = "catwalk",
                          latency: float = 0.06, seed: int = 0, record: str | None = None, backend: str = "humanoid"):
    """Run the realtime engine inside Blender (a background thread) streaming to ``port``."""
    from myrmex.realtime.inputs import InputConfig
    from myrmex.realtime.session import LiveConfig, LiveSession
    stop_embedded_engine()
    cfg = LiveConfig(rig=rig_json, seed=seed, out=[f"127.0.0.1:{port}"], clock=clock, bpm=bpm, style=style,
                     latency=latency, record=record, backend=backend,
                     inputs=InputConfig(osc_port=osc_port, midi=midi or []))
    s = LiveSession(cfg)
    s.start()
    _EMBEDDED["session"] = s
    return s


def stop_embedded_engine() -> str | None:
    s = _EMBEDDED.get("session")
    if s is None:
        return None
    _EMBEDDED["session"] = None
    return s.stop()


def embedded_session():
    return _EMBEDDED.get("session")


def export_rig_json(arm_obj: bpy.types.Object) -> str | None:
    """Where the auto-rig step stored the rig description (custom property), if any."""
    return arm_obj.get("myrmex_rig_json")


__all__ = ["LiveLink", "BasisSolver", "start_embedded_engine", "stop_embedded_engine", "embedded_session"]
