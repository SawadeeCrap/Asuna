"""Open a recorded take in Blender, ready to render (used by the Myrmex app).

    Blender [character.blend] --python blender/scripts/open_take.py -- --take TAKE.npz [--audio SONG.wav]
    Blender -b [character.blend] --python blender/scripts/open_take.py -- --take TAKE.npz --render \\
            [--size 1080x1920] [--quality eevee_preview|eevee|cycles] [--out video.mp4]

Creature takes (Black Nanomaterial / Mimetic Polyalloy) need no .blend: the scene is built.
Humanoid takes need the character's .blend (the app passes it).
"""
import argparse
import os
import sys

import bpy

HERE = os.path.dirname(os.path.realpath(__file__))
for p in (os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "src")):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--audio", default="")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--quality", default="eevee")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--keep-settings", action="store_true", help="render with the look's own settings")
    a = ap.parse_args(argv)
    if not hasattr(bpy.types.Scene, "myrmex_live"):
        import myrmex_blender
        myrmex_blender.register()
    from myrmex_blender import creature_take, ui
    s = bpy.context.scene.myrmex_live
    s.take_audio = a.audio
    s.render_size = a.size if a.size in {i.identifier for i in s.bl_rna.properties["render_size"].enum_items} else "1920x1080"
    s.render_quality = a.quality if a.quality != "look" else s.render_quality
    s.keep_settings = a.keep_settings or a.quality == "look"
    msg = ui.import_any_take(bpy.context, a.take, a.audio or None, not a.no_camera)
    print("Myrmex:", msg, flush=True)
    w, h = (int(x) for x in a.size.split("x"))
    out = a.out or os.path.splitext(os.path.abspath(os.path.expanduser(a.take)))[0] + f"_{w}x{h}.mp4"
    if a.render:
        creature_take.configure_video_output(out, (w, h), s.render_quality, keep=s.keep_settings)
        sc = bpy.context.scene
        total = sc.frame_end - sc.frame_start + 1

        def progress(scene, *_):
            print(f"Myrmex: frame {scene.frame_current - scene.frame_start + 1}/{total}", flush=True)
        bpy.app.handlers.render_post.append(progress)
        bpy.ops.render.render(animation=True)
        print("Myrmex: video written:", out, flush=True)
        return
    if not bpy.app.background:
        from myrmex_blender import control
        if control.enabled():              # the app can save / load looks in this Blender
            control.start()

        def look():
            for win in bpy.context.window_manager.windows:
                for area in win.screen.areas:
                    if area.type == "VIEW_3D":
                        for sp in area.spaces:
                            if sp.type == "VIEW_3D":
                                sp.shading.type = "MATERIAL"
                                sp.region_3d.view_perspective = "CAMERA"
            return None
        bpy.app.timers.register(look, first_interval=1.0)

        def syphon():                      # the take's picture for TouchDesigner (MYRMEX_SYPHON)
            from myrmex_blender import syphon_out
            st = syphon_out.from_env()
            if st is not None:
                print("Myrmex: Syphon", "ON:" if st.get("on") else "failed:", st.get("name") or st.get("error"),
                      flush=True)
            return None
        bpy.app.timers.register(syphon, first_interval=2.0)


main()
