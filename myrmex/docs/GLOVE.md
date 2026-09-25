# Hand Glove → organisms

The Hand Glove app streams 11 continuous values (thumb … pinky, roll, pitch, yaw, x, y, z) as MIDI.
Mapped to parameters they only nudge the behaviour; the glove link moves the **body itself**.

## Connecting (parallel to your other programs)

* Keep sending the glove to your other programs as before.  macOS lets any number of apps read the
  same MIDI source: enable the glove's port (or the IAC bus it uses) on **Inputs** - a port whose
  name contains "glove" or "hand" is joined automatically.
* It links by itself: as soon as the glove streams, the 11 controls are taken in the glove's own order
  (14-bit CC pairs, 7-bit CC and pitch bend are understood).  If the order is wrong:
  **Glove** page → *Learn* on a row, then press **MAP** on the same row in Hand Glove within 1.5 s.
* OSC works too: `/glove/thumb` … `/glove/z` (value 0..1 or 0..16383) to the OSC port (9100).
* Linked controls no longer drive parameters (MIDI page) while a preset is on - they drive the body.
* *Calibrate neutral pose*: hold the hand relaxed, palm down, click - that pose is "straight".

## Presets

| preset | hand turn | fingers | position | gestures |
|---|---|---|---|---|
| **Puppet** | the body turns with the hand, 1:1, rigidly (no lag) | five sectors around the body = five fingers: extended → limbs / spikes, curled → claw closes; open hand spreads, fist compacts | up/down = height, left/right = steering, towards the screen = bigger and the camera comes closer | on |
| **Sculpt** | turns the form in place | each finger blends in one of five forms (per organism, e.g. Osseous: carapace, spine, scythe, claw/mandible, thorn) | depth = size | on |
| **Conductor** | twist = spin speed, tilt = material (up melts, down hardens / ossifies) | open fingers = energy | yaw steers, height lifts | on |
| **Camera** | orbit / height | – | depth = zoom | flick = cut |

Gestures (derived from motion, the glove sends no triggers): **flick** → strike / impulse · **fist** →
harden / ossify · **spread** → burst / quill volley · **push** → strike · **pinch** → split, pinch again →
merge.  Sliders: intensity, smoothing (One-Euro filter: smooth at rest, instant in motion), gesture
sensitivity; *invert fingers* if an open hand closes the organism.

Works with every organism: Black Nanomaterial (turn, fingers = its limbs and their curl, rear up /
crouch), Mimetic Polyalloy, Colony, Hive and the Osseous line (every body of a flock turns with the hand).
