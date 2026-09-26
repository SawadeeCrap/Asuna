#!/usr/bin/env python3
"""Build a self-contained Myrmex.app for Apple Silicon (runs on macOS or Linux).

    python3 tools/build_macos_app.py [--cache build/cache] [--out dist]

Result: ``dist/Myrmex.app`` and ``dist/Myrmex-macOS-arm64.zip``.  Nothing to
install on the Mac: the bundle carries its own Python 3.12 (python-build-standalone),
numpy, PySide6 (Qt, trimmed to what the window needs, arm64 only), Ableton Link
(aalink), MIDI (python-rtmidi + mido), audio input (sounddevice), the Myrmex engine,
the Blender add-on and scripts, the Ableton Remote Script and the ready character.

Layout::

    Myrmex.app/Contents/
        Info.plist
        MacOS/Myrmex            launcher (sets PYTHONHOME, runs ``python3 -m myrmex.app``)
        MacOS/python3           the interpreter, inside MacOS/ so the menu bar says "Myrmex"
        lib -> Resources/python/lib   (the interpreter finds libpython via @executable_path/../lib)
        Resources/python/       CPython 3.12 + site-packages
        Resources/app/          src/myrmex, blender/, ableton/, characters/, docs/
        Resources/Myrmex.icns

All Mach-O files keep their original (ad-hoc / linker) signatures: universal2 files are
thinned by copying out the arm64 slice, which carries its own signature.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PBS_TAG = "20260310"
PY_VER = "3.12.13"
PBS_URL = (f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_TAG}/"
           f"cpython-{PY_VER}+{PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz")
REQUIREMENTS = ["numpy", "PySide6-Essentials", "aalink", "python-rtmidi", "mido", "sounddevice"]
PLATFORMS = ["macosx_14_0_arm64", "macosx_13_0_arm64", "macosx_12_0_arm64", "macosx_11_0_arm64",
             "macosx_13_0_universal2", "macosx_11_0_universal2", "macosx_10_9_universal2",
             "macosx_10_6_universal2", "any"]
QT_MODULES = ("QtCore", "QtGui", "QtWidgets")
# Syphon for Blender's own Python (the picture for TouchDesigner): Blender 4.2-5.0 = 3.11, 5.1+ = 3.13.
BLENDER_PYTHONS = ("3.11", "3.13")
SYPHON_PURE = ("syphon-python==0.1.1", "pyopengl==3.1.10")          # Python + Syphon.framework, any CPython
PYOBJC = ("pyobjc-core==10.3.2", "pyobjc-framework-Cocoa==10.3.2", "pyobjc-framework-Metal==10.3.2")
VENDOR_PLATFORMS = ("macosx_11_0_arm64", "macosx_10_13_universal2", "macosx_10_9_universal2", "macosx_11_0_universal2",
                    "any")
QT_PLUGINS = ("platforms/libqcocoa.dylib", "styles/libqmacstyle.dylib", "imageformats/libqicns.dylib")
VERSION = "0.3.0"

CPU_ARM64 = 0x0100000C
LOAD_DYLIB_CMDS = (0xC, 0x80000018, 0x8000001F, 0x80000023)


# ---------------------------------------------------------------------------- Mach-O helpers
def thin_to_arm64(path: str) -> bool:
    """Replace a universal (fat) Mach-O by its arm64 slice.  Returns True if the file was fat."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            return False
        magic, n = struct.unpack(">II", head)
        if magic not in (0xCAFEBABE, 0xCAFEBABF) or not 0 < n < 16:   # (Java classes share CAFEBABE)
            return False
        wide = magic == 0xCAFEBABF
        size = 32 if wide else 20
        entries = f.read(n * size)
        for i in range(n):
            e = entries[i * size:(i + 1) * size]
            if wide:
                cpu, _sub, off, ln, _al, _res = struct.unpack(">iiQQII", e)
            else:
                cpu, _sub, off, ln, _al = struct.unpack(">iiIII", e)
            if cpu == CPU_ARM64:
                f.seek(off)
                data = f.read(ln)
                break
        else:
            raise ValueError(f"{path}: no arm64 slice")
    mode = os.stat(path).st_mode
    with open(path, "wb") as f:
        f.write(data)
    os.chmod(path, mode)
    return True


def dylib_deps(path: str) -> list[str]:
    with open(path, "rb") as f:
        data = f.read(65536)
    if len(data) < 32 or struct.unpack("<I", data[:4])[0] != 0xFEEDFACF:
        return []
    ncmds, sizeofcmds = struct.unpack("<II", data[16:24])
    if 32 + sizeofcmds > len(data):
        with open(path, "rb") as f:
            data = f.read(32 + sizeofcmds)
    off, out = 32, []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack("<II", data[off:off + 8])
        if cmd in LOAD_DYLIB_CMDS:
            name_off = struct.unpack("<I", data[off + 8:off + 12])[0]
            out.append(data[off + name_off:off + cmdsize].split(b"\0", 1)[0].decode())
        off += cmdsize
    return out


def is_macho(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            m = f.read(4)
    except OSError:
        return False
    return m in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf")


# ---------------------------------------------------------------------------- downloads
def fetch(url: str, dst: str) -> str:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("download", url)
    if shutil.which("curl"):
        subprocess.run(["curl", "-sSfL", "-o", dst, url], check=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, dst)
    return dst


def fetch_wheels(dst: str, pip: list[str]) -> list[str]:
    os.makedirs(dst, exist_ok=True)
    cmd = pip + ["download", "-q", "-d", dst, "--only-binary=:all:", "--python-version", "3.12",
                 "--implementation", "cp", "--abi", "cp312", "--abi", "abi3", "--abi", "none"]
    for p in PLATFORMS:
        cmd += ["--platform", p]
    subprocess.run(cmd + REQUIREMENTS, check=True)
    return sorted(os.path.join(dst, f) for f in os.listdir(dst) if f.endswith(".whl"))


# ---------------------------------------------------------------------------- assembly
def install_wheel(whl: str, site: str) -> None:
    with zipfile.ZipFile(whl) as z:
        for info in z.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            target = os.path.join(site, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                os.chmod(target, mode)


def prune_pyside(site: str) -> None:
    ps = os.path.join(site, "PySide6")
    qt = os.path.join(ps, "Qt")
    # Python-level modules and loose files.
    keep_top = {f"{m}.abi3.so" for m in QT_MODULES}
    for f in os.listdir(ps):
        p = os.path.join(ps, f)
        if f in ("Qt", "support", "__pycache__"):
            continue
        if os.path.isdir(p):
            shutil.rmtree(p)                       # tools .app bundles, typesystems, glue, include, scripts...
        elif f.endswith(".py") or f in keep_top or f.startswith("libpyside6.abi3") or f in ("py.typed",):
            continue
        else:
            os.remove(p)
    for d in ("qml", "translations", "metatypes", "libexec"):
        shutil.rmtree(os.path.join(qt, d), ignore_errors=True)
    # Plugins: only what a widgets window on macOS needs.
    plug = os.path.join(qt, "plugins")
    keep_plugins = {os.path.join(plug, p) for p in QT_PLUGINS}
    for root, _dirs, files in os.walk(plug):
        for f in files:
            p = os.path.join(root, f)
            if p not in keep_plugins:
                os.remove(p)
    for root, dirs, _files in os.walk(plug, topdown=False):
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except OSError:
                pass
    # Frameworks: the dependency closure of the kept binaries.
    libdir = os.path.join(qt, "lib")
    roots = [os.path.join(ps, f) for f in os.listdir(ps) if f.endswith((".so", ".dylib"))] + sorted(keep_plugins)
    needed, todo = set(), list(roots)
    while todo:
        b = todo.pop()
        for dep in dylib_deps(b):
            if ".framework/" in dep and dep.startswith("@rpath/"):
                fw = [p for p in dep.split("/") if p.endswith(".framework")][0]
                if fw not in needed:
                    needed.add(fw)
                    name = fw[:-len(".framework")]
                    todo.append(os.path.join(libdir, fw, "Versions", "A", name))
    for fw in os.listdir(libdir):
        if fw not in needed:
            shutil.rmtree(os.path.join(libdir, fw))
    print("Qt frameworks kept:", sorted(needed))


def check_closure(res: str) -> None:
    """Every @rpath framework referenced by a kept binary must exist; every binary must be arm64."""
    missing, archs = set(), set()
    for root, _dirs, files in os.walk(res):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p) or not is_macho(p):
                continue
            with open(p, "rb") as fh:
                head = fh.read(8)
            archs.add(struct.unpack("<I", head[4:8])[0] if head[:4] == b"\xcf\xfa\xed\xfe" else "fat")
            for dep in dylib_deps(p):
                if dep.startswith("@rpath/") and ".framework/" in dep:
                    fw = [x for x in dep.split("/") if x.endswith(".framework")][0]
                    libdir = os.path.join(res, "python", "lib", "python3.12", "site-packages", "PySide6", "Qt", "lib")
                    if not os.path.exists(os.path.join(libdir, fw)):
                        missing.add((os.path.relpath(p, res), fw))
    if missing:
        raise SystemExit(f"missing Qt frameworks: {sorted(missing)}")
    if archs != {CPU_ARM64}:
        raise SystemExit(f"non-arm64 binaries left: {archs}")
    print("closure ok: all Mach-O arm64, all Qt frameworks present")


def prune_python(py: str) -> None:
    lib = os.path.join(py, "lib")
    std = os.path.join(lib, "python3.12")
    for d in ("tkinter", "idlelib", "turtledemo", "test", "ensurepip", "lib2to3", "pydoc_data"):
        shutil.rmtree(os.path.join(std, d), ignore_errors=True)
    for f in os.listdir(lib):
        if f.startswith(("libtcl", "libtk", "tcl", "tk", "itcl", "thread", "sqlite3")) and f != "python3.12":
            p = os.path.join(lib, f)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    dyn = os.path.join(std, "lib-dynload")
    for f in os.listdir(dyn):
        if f.startswith("_tkinter"):
            os.remove(os.path.join(dyn, f))
    for d in ("share", "include"):
        shutil.rmtree(os.path.join(py, d), ignore_errors=True)
    site = os.path.join(std, "site-packages")
    for f in os.listdir(site) if os.path.isdir(site) else []:
        if f == "pip" or (f.startswith("pip-") and f.endswith(".dist-info")):
            shutil.rmtree(os.path.join(site, f), ignore_errors=True)       # no installs inside the app
    for d in ("f2py", "typing", "_pyinstaller", "testing", "doc"):
        shutil.rmtree(os.path.join(site, "numpy", d), ignore_errors=True)
    shutil.rmtree(os.path.join(site, "numpy", "_core", "include"), ignore_errors=True)
    for root, dirs, _files in os.walk(site):
        for d in list(dirs):
            if d in ("tests", "testing") and "numpy" in root:
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)
    for root, _dirs, files in os.walk(site):
        for f in files:
            if f.endswith(".pyi"):
                os.remove(os.path.join(root, f))


def copy_app_sources(dst: str) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.blend1")
    shutil.copytree(os.path.join(REPO, "src", "myrmex"), os.path.join(dst, "src", "myrmex"), ignore=ign)
    shutil.copytree(os.path.join(REPO, "blender", "myrmex_blender"), os.path.join(dst, "blender", "myrmex_blender"),
                    ignore=ign)
    shutil.copytree(os.path.join(REPO, "blender", "scripts"), os.path.join(dst, "blender", "scripts"), ignore=ign)
    shutil.copytree(os.path.join(REPO, "ableton", "remote_script"), os.path.join(dst, "ableton", "remote_script"),
                    ignore=ign)
    shutil.copytree(os.path.join(REPO, "characters"), os.path.join(dst, "characters"), ignore=ign)
    shutil.copytree(os.path.join(REPO, "touchdesigner"), os.path.join(dst, "touchdesigner"), ignore=ign)
    os.makedirs(os.path.join(dst, "docs"), exist_ok=True)
    for f in ("docs/REALTIME.md", "docs/TOUCHDESIGNER.md", "README.md"):
        shutil.copy2(os.path.join(REPO, f), os.path.join(dst, f))


def vendor_blender_syphon(cache: str, blender_dir: str, pip: list[str]) -> None:
    """syphon-python + pyobjc for Blender's Python (blender/vendor/common + cpXY): nothing to install on the Mac."""
    base = os.path.join(blender_dir, "vendor")
    shutil.rmtree(base, ignore_errors=True)

    def download(dst: str, pyver: str, reqs) -> list[str]:
        os.makedirs(dst, exist_ok=True)
        cmd = pip + ["download", "-q", "--no-deps", "-d", dst, "--only-binary=:all:", "--python-version", pyver,
                     "--implementation", "cp"]
        for p in VENDOR_PLATFORMS:
            cmd += ["--platform", p]
        subprocess.run(cmd + list(reqs), check=True)
        return sorted(os.path.join(dst, f) for f in os.listdir(dst) if f.endswith(".whl"))
    wheels = os.path.join(cache, "blender-wheels")
    for whl in download(os.path.join(wheels, "common"), "3.12", SYPHON_PURE):
        install_wheel(whl, os.path.join(base, "common"))
    for pyver in BLENDER_PYTHONS:
        tag = "cp" + pyver.replace(".", "")
        for whl in download(os.path.join(wheels, tag), pyver, PYOBJC):
            install_wheel(whl, os.path.join(base, tag))
    thinned = 0
    for root, _dirs, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p) and is_macho(p):
                thinned += thin_to_arm64(p)
    print("Blender Syphon:", ", ".join(sorted(os.listdir(base))), f"(thinned {thinned})")


def make_icon(path: str) -> bool:
    drawn = os.path.join(REPO, "src", "myrmex", "app", "assets", "Myrmex.icns")     # tools/make_icon.py
    if os.path.exists(drawn):
        shutil.copy2(drawn, path)
        return True
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    S = 1024
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((40, 40, S - 40, S - 40), radius=210, fill=(18, 18, 22, 255))
    for i in range(0, 420, 6):                               # chrome sheen
        a = int(90 * (1 - i / 420))
        d.ellipse((S / 2 - 420 + i, 120 + i // 2, S / 2 + 420 - i, 560 + i // 3), outline=(255, 255, 255, a))
    # A walking figure: head, torso, legs in stride, arm swing (white on black chrome).
    w = (240, 240, 245, 255)
    d.ellipse((470, 190, 570, 290), fill=w)
    d.line((520, 300, 505, 560), fill=w, width=56)
    d.line((505, 560, 400, 820), fill=w, width=50)
    d.line((505, 560, 640, 800), fill=w, width=50)
    d.line((515, 340, 420, 500), fill=w, width=38)
    d.line((520, 340, 630, 470), fill=w, width=38)
    d.line((300, 860, 740, 860), fill=(255, 90, 54, 255), width=18)     # the beat line
    try:
        im.save(path, sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
        return True
    except Exception as e:
        print("icon skipped:", e)
        return False


def precompile(dirs: list[str], python312: str | None) -> None:
    if not python312:
        print("no CPython 3.12 on this machine: skipping .pyc precompilation")
        return
    for d in dirs:
        subprocess.run([python312, "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash", d],
                       check=False)


def build(cache: str, out: str, pip: list[str], python312: str | None) -> str:
    app = os.path.join(out, "Myrmex.app")
    shutil.rmtree(app, ignore_errors=True)
    contents = os.path.join(app, "Contents")
    res = os.path.join(contents, "Resources")
    os.makedirs(os.path.join(contents, "MacOS"))
    os.makedirs(res)
    # Python
    tgz = fetch(PBS_URL, os.path.join(cache, os.path.basename(PBS_URL)))
    with tarfile.open(tgz) as t:
        t.extractall(res, filter="tar") if sys.version_info >= (3, 12) else t.extractall(res)
    py = os.path.join(res, "python")
    prune_python(py)
    site = os.path.join(py, "lib", "python3.12", "site-packages")
    os.makedirs(site, exist_ok=True)
    for whl in fetch_wheels(os.path.join(cache, "wheels"), pip):
        install_wheel(whl, site)
    # Thin first: the dependency scan below reads thin arm64 headers.
    thinned = 0
    for root, _dirs, files in os.walk(res):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p) and is_macho(p):
                thinned += thin_to_arm64(p)
    print("thinned universal binaries:", thinned)
    prune_pyside(site)
    prune_python(py)
    check_closure(res)
    # The interpreter lives in MacOS/ (menu bar name, Dock icon); libpython via Contents/lib.
    shutil.copy2(os.path.join(py, "bin", "python3.12"), os.path.join(contents, "MacOS", "python3"))
    os.symlink("Resources/python/lib", os.path.join(contents, "lib"))
    # The app
    copy_app_sources(os.path.join(res, "app"))
    vendor_blender_syphon(cache, os.path.join(res, "app", "blender"), pip)
    precompile([os.path.join(res, "app", "src"), site], python312)
    launcher = os.path.join(contents, "MacOS", "Myrmex")
    with open(launcher, "w") as f:
        f.write('#!/bin/bash\n'
                '# Myrmex.app launcher: the bundled Python runs the app; logs go to ~/Myrmex/myrmex.log\n'
                'CONTENTS="$(cd "$(dirname "$0")/.." && pwd -P)"\n'
                'export PYTHONHOME="$CONTENTS/Resources/python"\n'
                'export PYTHONPATH="$CONTENTS/Resources/app/src"\n'
                'export PYTHONNOUSERSITE=1\n'
                'export PYTHONDONTWRITEBYTECODE=1\n'
                'export MYRMEX_BUNDLE="$CONTENTS"\n'
                'mkdir -p "$HOME/Myrmex"\n'
                'exec "$CONTENTS/MacOS/python3" -m myrmex.app "$@" >>"$HOME/Myrmex/myrmex.log" 2>&1\n')
    os.chmod(launcher, 0o755)
    icon = make_icon(os.path.join(res, "Myrmex.icns"))
    info = {
        "CFBundleName": "Myrmex", "CFBundleDisplayName": "Myrmex", "CFBundleIdentifier": "app.myrmex.live",
        "CFBundleVersion": VERSION, "CFBundleShortVersionString": VERSION, "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "Myrmex", "LSMinimumSystemVersion": "14.0", "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.music",
        "NSMicrophoneUsageDescription": "Myrmex listens to the music input to animate the character.",
        "NSAppleEventsUsageDescription": "Myrmex opens your character in Blender.",
    }
    if icon:
        info["CFBundleIconFile"] = "Myrmex.icns"
    with open(os.path.join(contents, "Info.plist"), "wb") as f:
        plistlib.dump(info, f)
    # Zip (symlinks preserved) for download.
    zpath = os.path.join(out, "Myrmex-macOS-arm64.zip")
    if os.path.exists(zpath):
        os.remove(zpath)
    subprocess.run(["zip", "-qry9", os.path.basename(zpath), "Myrmex.app"], cwd=out, check=True)
    return zpath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(REPO, "build", "cache"))
    ap.add_argument("--out", default=os.path.join(REPO, "dist"))
    ap.add_argument("--pip", default=f"{sys.executable} -m pip", help="pip command used to download wheels")
    ap.add_argument("--python312", default=shutil.which("python3.12"), help="CPython 3.12 for .pyc precompilation")
    a = ap.parse_args()
    z = build(os.path.abspath(a.cache), os.path.abspath(a.out), a.pip.split(), a.python312)
    print("built:", z, f"{os.path.getsize(z) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
