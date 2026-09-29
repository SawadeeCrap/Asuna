# Installing Fusion Clone (VCV Rack 2, macOS Apple silicon; Linux and Windows are analogous)

*(Русская версия: `docs/INSTALL.ru.md`.)*

There is **no pre-built binary in the repository**: a Rack plugin is compiled machine code for one platform, and the sandbox this project was
written in is a Linux container that has neither your Mac nor Rack. You therefore either **build it once from source (about 5 minutes)** or
**download the package that the GitHub workflow builds** (`.github/workflows/fusion-clone.yml`, artifact `FusionClone-mac-arm64`; see B below).

## A. Build from source on the Mac

1. **Compiler.** Xcode command line tools (skip if `clang --version` works):
   ```sh
   xcode-select --install
   ```
2. **Rack SDK.** On https://vcvrack.com/downloads download the **Rack SDK 2.x for Mac (arm64)** — the same 2.x version as your Rack
   (Rack → Help → About) — and unzip it to `~/Rack-SDK`. (The SDK is only headers and makefiles; you do not need the Rack source.)
3. **Source, build, install.**
   ```sh
   git clone https://github.com/SawadeeCrap/Asuna.git
   cd Asuna && git checkout claude/jolly-einstein-e9h02j
   cd fusion-clone
   export RACK_DIR="$HOME/Rack-SDK"
   make -j"$(sysctl -n hw.ncpu)"
   make install
   ```
   `make install` copies the plugin into `~/Library/Application Support/Rack2/plugins-mac-arm64/FusionClone/`.
   `make dist` builds a distributable `dist/FusionClone-<version>-mac-arm64.vcvplugin` instead.
4. **Restart Rack.** Add the module: right-click on the rack → search **Fusion Clone** (brand *Asuna*).

## B. Download the package built by GitHub (no compiler needed)

Open the repository on GitHub → **Actions** → the latest *fusion-clone* run of branch `claude/jolly-einstein-e9h02j` → **Artifacts** →
`FusionClone-mac-arm64`. Unzip it and double-click the `.vcvplugin` (Rack installs it), or unzip the archive's content into
`~/Library/Application Support/Rack2/plugins-mac-arm64/`. If macOS refuses to load it because it was downloaded:
```sh
xattr -dr com.apple.quarantine "$HOME/Library/Application Support/Rack2/plugins-mac-arm64/FusionClone"
```
The same run also executes the DSP tests and the CPU benchmark on an Apple-silicon runner (summary tab).

## First use

* Patch the **OUT** of your Fusion VCO2 (through your audio interface, into a Rack *Audio* module input) to **AUDIO IN**, and **OUT L / OUT R**
  to your mixer. Nominal level is ±5 V; the DSP is level independent but keep the signal out of clipping.
* Turn **VOICES** (1 = only the original, 16 = the original + 15 clones). The display shows **LOCK xx.x Hz** and the LOCK light comes on a few
  periods after a note starts; until then you hear the original alone. The module is monophonic (polyphonic cables: channel 1).
* Everything else is described in `README.md` (panel) and `docs/MANIFEST.md`; the context menu holds the expert controls.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `make`: *plugin.mk / RACK_DIR not found* | `RACK_DIR` must point at the unzipped SDK folder (it contains `plugin.mk`). |
| Rack shows no module / log says the plugin was built for another version | Use the SDK whose 2.x version equals your Rack's. Log: `~/Library/Application Support/Rack2/log.txt`. |
| Link error mentioning `pffft_…` | The SDK does not export pffft: edit `FLAGS` in `Makefile` — `-DFC_FFT_RACK` (Rack's own FFT wrapper) or remove `-DFC_FFT_PFFFT` (built-in FFT, slower). |
| Suspected SIMD problem on a platform | Rebuild with the portable dot product: `FLAGS=-DFC_NO_SIMD make -j … && make install`. |
| macOS blocks the downloaded plugin | `xattr -dr com.apple.quarantine …` as above. |
| The display stays at *ACQUIRING* / *PASS-THRU* | The input must be a stable, monophonic, periodic oscillator (saw/tri/pulse/sine, optional sub); noise, chords and DC never lock by design. Check the input level (input meter in the display). |
| CPU too high | Lower QUALITY (ECO/BALANCED) or VOICES; see `docs/BENCHMARKS.md` (the figures there are from an x86 Linux machine; the benchmark job in the GitHub run gives an Apple-silicon data point). |

## Tell me what happens

This code was never run inside Rack (no Rack, no display and no audio hardware where it was written). The most useful feedback after your first
run: the Rack log (`log.txt`) if the plugin does not appear, a screenshot of the panel, the CPU meter at VOICES = 16 for each QUALITY, and how
it *sounds* against your real oscillator(s) — `docs/REFERENCE_PROTOCOL.md` describes measurements that settle the open questions about the
real Fusion VCO2.
