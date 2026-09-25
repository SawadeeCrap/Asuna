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

## Presets (20 + Off)

| preset | hand turn | fingers | position | what you see |
|---|---|---|---|---|
| **Puppet** | the body turns with the hand, 1:1, rigidly (no lag) | five sectors around the body = five fingers: extended → limbs / spikes, curled → claw closes | up/down = height, left/right = steering, towards the screen = bigger + camera closer | the organism is your hand |
| **Sculpt** | turns the form in place | each finger blends in one of five forms (per organism; Cyber Hive: prism, halo, array, scythe, spine) | depth = size | a form you mould |
| **Conductor** | twist = spin speed, tilt = material (up melts, down hardens) | open = energy | yaw steers, height lifts | tempo and matter |
| **Camera** | orbit / height | – | depth = zoom | the organism is free, you are the camera |
| **Marionette** | tilt = tilt | five strings along the body (tail → head): curl a finger and its part is pulled up | height, steering | a puppet on strings |
| **Harp** | twist = wave speed | every finger plucks its own travelling wave; the faster the move, the stronger it rings | height, steering | ripples running through the body |
| **Heartbeat** | loose follow | open = depth of the beat-locked breathing, fist = hard contractions | depth = size | it pulses on the beat |
| **Elastic** | twist screws the body | open = soft, fist = stiff | forward/back = length, left/right = width, up/down = height | rubber in your hand |
| **Dust** | twist swirls the cloud | open = it falls apart into a cloud, fist = gathers back | height, steering | disintegration / assembly |
| **Stasis** | turns the frozen form like a sculpture | fist = time stops (motion freezes), open = it lives again | – | freeze-frame |
| **Storm** | where the hand points = where the wind comes from | open = wind strength | height, steering | streams like a flag / a comet |
| **Leash** | twist = circle direction and speed | open = wider circle | where the hand points (camera view) = where it flies | a kite on a line |
| **Pilot** | bank = turn, tilt = climb / dive | open = throttle | – | fly it like an aircraft |
| **Flywheel** | a quick turn throws spin into it - it keeps spinning | fist = brake | – | momentum |
| **Shepherd** | turns the flock's formation | number of extended fingers = number of bodies (v3, v4, v6–v8), open = spread | height, steering | you herd the flock |
| **Swarm** | twist = swirl | open = nanomachines fly out to your hand, fist = back (v4, v7, v8; others scatter) | where the cloud gathers | a cloud in your hand |
| **Neon** | loose follow | open = brighter light lines, quick finger taps flash them | height = energy | light (best on the Cyber Hive) |
| **Rhythm** | the body snaps to the hand's angle in 45° steps, only on the beat | open = pulse on the beat | – | robotic, quantized |
| **Echo** | Puppet, one beat later | | | a canon with your hand |
| **Mandala** | twist turns the rays, yaw spins the body | extended fingers = number of rays, open = ray length | – | radial symmetry |
| **Off** | the glove's MIDI goes to the parameters (MIDI page) as before | | | |

Gestures (derived from motion, the glove sends no triggers): **flick** → strike / impulse · **fist** →
harden / ossify · **spread** → burst / quill volley · **push** → strike · **pinch** → split, pinch again →
merge.  Sliders: intensity, smoothing (One-Euro filter: smooth at rest, instant in motion), gesture
sensitivity; *invert fingers* if an open hand closes the organism.

Works with every organism: Black Nanomaterial (turn, fingers = its limbs and their curl, rear up /
crouch), Mimetic Polyalloy, Colony, Hive, the Osseous line and the Cyber Hive (every body of a flock
turns with the hand).  While Shepherd holds a flock, the organism does not merge or split it by itself.
