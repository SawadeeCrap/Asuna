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

from myrmex.realtime.protocol import decode_names, decode_pose

LIVE_CAMERA = "MyrmexLiveCam"


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
                 camera: bool = True, lights: bool = True, floor: bool = True):
        self.arm = arm_obj
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

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind((self.host, self.port))
        self.sock.setblocking(False)
        for pb in self.arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
        if self.arm.animation_data is not None:
            self.arm.animation_data.action = None          # baked actions would fight the stream
        self.running = True
        if not bpy.app.timers.is_registered(self._timer):
            bpy.app.timers.register(self._timer, first_interval=0.0, persistent=True)

    def stop(self) -> None:
        self.running = False
        if bpy.app.timers.is_registered(self._timer):
            bpy.app.timers.unregister(self._timer)
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _timer(self):
        if not self.running:
            return None
        try:
            if self.poll():
                _tag_redraw()
        except Exception as e:                 # never kill the timer: report and keep listening
            self.stats["error"] = f"{type(e).__name__}: {e}"
        return 1.0 / 240.0

    # ------------------------------------------------------------------ receive + apply
    def poll(self) -> bool:
        """Drain the socket, apply the newest pose.  Returns True if something changed."""
        if self.sock is None:
            return False
        newest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            if data[:4] == b"MYRN":
                nm = decode_names(data)
                if nm is not None and (self.rig != nm[0] or self.solver is None):
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
                rig.location = (float(p[0]), float(p[1]), 0.0)
                rig.rotation_euler = (0.0, 0.0, float(fr.heading))
        if self.use_floor:
            floor = bpy.data.objects.get("MyrmexFloor")
            if floor is not None:
                # The floor shader works in world space, so the plane can simply travel along.
                floor.location = (float(p[0]), float(p[1]), 0.0)


# ---------------------------------------------------------------------------- embedded engine
_EMBEDDED = {"session": None}


def start_embedded_engine(rig_json: str, port: int = 9101, osc_port: int = 9100, clock: str = "auto",
                          midi: list[str] | None = None, bpm: float = 120.0, style: str = "catwalk",
                          latency: float = 0.06, seed: int = 0, record: str | None = None):
    """Run the realtime engine inside Blender (a background thread) streaming to ``port``."""
    from myrmex.realtime.inputs import InputConfig
    from myrmex.realtime.session import LiveConfig, LiveSession
    stop_embedded_engine()
    cfg = LiveConfig(rig=rig_json, seed=seed, out=[f"127.0.0.1:{port}"], clock=clock, bpm=bpm, style=style,
                     latency=latency, record=record, inputs=InputConfig(osc_port=osc_port, midi=midi or []))
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
