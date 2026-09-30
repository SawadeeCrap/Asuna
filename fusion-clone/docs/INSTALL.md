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

1. Open <https://github.com/SawadeeCrap/Asuna/actions/workflows/fusion-clone.yml> and click the newest run of the branch
   `claude/jolly-einstein-e9h02j` whose jobs **Plugin (mac-arm64)** and **Load in Rack (mac-arm64)** are green (the small check marks in the run's job list).
2. At the bottom of the run page, under **Artifacts**, download **FusionClone-mac-arm64** (a ~85 KB `.zip`; you must be logged in to GitHub) and
   unzip it. It contains `FusionClone-2.0.0-mac-arm64.vcvplugin`. GitHub keeps artifacts for 90 days; after that re-run the workflow (Actions →
   *fusion-clone* → *Run workflow*) or build from source (A).
3. Install it: double-click the `.vcvplugin` (Rack installs it), **or** copy it into
   `~/Library/Application Support/Rack2/plugins-mac-arm64/` and start Rack — Rack unpacks the package itself at start-up.
4. If macOS refuses to load it because it was downloaded from the internet:
   ```sh
   xattr -dr com.apple.quarantine "$HOME/Library/Application Support/Rack2/plugins-mac-arm64/FusionClone"
   ```

(The same run also has `FusionClone-lin-x64`, the equivalent package for Linux x86-64; Rack loads it the same way from `~/.local/share/Rack2/plugins-lin-x64/`.)

What the package is: an arm64 build against the official **Rack SDK 2.6.x** (it is ad-hoc code-signed, not notarised — like every plugin that is not
downloaded through the VCV Library), for **Rack 2** (tested only with Rack 2.6.6; other 2.x versions should work because the plugin ABI is stable within a major version, but that is untested). The same workflow run also
loads exactly this package into a real **VCV Rack Free 2.6.6 on an Apple-silicon runner**, patches a Fundamental VCO into it and lets it run for 20 s;
its log shows the plugin loaded, the module created, and the module locked onto the VCO's pitch (details: `tests/rack_smoke.sh`, the job *Load in Rack
(mac-arm64)*). The DSP test suite and the CPU benchmark run on an Apple M1 runner in the same workflow.

You can repeat the load test against **your own Rack** (quit Rack first: the test starts its own windowless instance; it uses a scratch user folder and does not touch your Rack settings; needs `brew install zstd`):
```sh
bash fusion-clone/tests/rack_smoke.sh ~/Downloads/FusionClone-2.0.0-mac-arm64.vcvplugin "/Applications/VCV Rack 2 Free.app/Contents/MacOS/Rack" 20
```
(for Rack Pro use `"/Applications/VCV Rack 2 Pro.app/Contents/MacOS/Rack"`). It ends with `RACK SMOKE TEST: PASS` or `FAIL` and prints Rack's log.

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
| Rack shows no module / log says the plugin was built for another version | Use the SDK whose 2.x version equals your Rack's (the CI package was built with SDK 2.6.x and tested with Rack 2.6.6). Log: `~/Library/Application Support/Rack2/log.txt` — look for `Loaded plugin FusionClone` and `Fusion Clone: module added`. |
| Link error mentioning `pffft_…` | The SDK does not export pffft: edit `FLAGS` in `Makefile` — `-DFC_FFT_RACK` (Rack's own FFT wrapper) or remove `-DFC_FFT_PFFFT` (built-in FFT, slower). |
| Suspected SIMD problem on a platform | Rebuild with the portable dot product: `FLAGS=-DFC_NO_SIMD make -j … && make install`. |
| macOS blocks the downloaded plugin | `xattr -dr com.apple.quarantine …` as above. |
| The display stays at *ACQUIRING* / *PASS-THRU* | The input must be a stable, monophonic, periodic oscillator (saw/tri/pulse/sine, optional sub); noise, chords and DC never lock by design. Check the input level (input meter in the display). |
| CPU too high | Lower QUALITY (ECO/BALANCED) or VOICES; see `docs/BENCHMARKS.md` (§1–§4 are from an x86 Linux machine, §6 from an Apple M1 virtual machine). |

## Tell me what happens

Nobody has *heard* this module and it has not run on your Mac yet: it was written without audio hardware, and what has been exercised is the DSP tests
(x86 and Apple M1), the build against the Rack SDK, and a real Rack Free 2.6.6 loading the package and running it for 20 s in CI (plus a Linux
screenshot of the panel in `docs/figures/rack/`). The most useful feedback after your first run: the Rack log (`log.txt`) if the plugin does not appear,
a screenshot of the panel, the CPU meter at VOICES = 16 for each QUALITY, and how it *sounds* against your real oscillator(s) —
`docs/REFERENCE_PROTOCOL.md` describes measurements that settle the open questions about the real Fusion VCO2.
