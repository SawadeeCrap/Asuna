# Живой режим: Ableton / VCV Rack → персонаж в реальном времени

Главный сценарий проекта: включаешь Ableton (или VCV Rack), и персонаж в Blender
**сразу, вживую** идёт, реагирует на удары и держит ритм. Шаги ставятся на долю,
камера сама меняет планы на сильных долях, а при остановке музыки персонаж не
замирает статуей: встаёт в позу, дышит, переносит вес и ждёт.

```
Ableton Live ──Link / Remote Script / MIDI (IAC) / аудио──┐
VCV Rack 2 ───MIDI (IAC) / OSC──────────────────────────────┤
                                                            ▼
                myrmex live  (Python, 120 Гц: музыка → поведение → тело → поза)
                                                            │  позы + камера, UDP :9101, 60 кадров/с
                                                            ▼
                Blender + аддон Myrmex: арматура, живая камера, свет, бесконечный пол
                                                            │  (по желанию) запись дубля
                                                            ▼
                финальный рендер дубля в EEVEE / Cycles (офлайн, максимальное качество)
```

Движок `myrmex live` использует тот же шаг симуляции (`PerformanceCore`), что и
офлайн-генерация, поэтому вживую персонаж ведёт себя так же, как в отрендеренных
роликах.

---

## 0. Что нужно на Mac (M4 Pro, 24 ГБ)

| Что | Зачем |
|---|---|
| Blender 5.2 LTS (или 4.5 LTS) | показ персонажа в реальном времени (EEVEE, Metal) |
| Python 3.12 | для `myrmex live`. На 3.13 у `python-rtmidi` пока нет готовых сборок под macOS |
| Ableton Live 11/12 и/или VCV Rack 2 | источник музыки |
| BlackHole 2ch (необязательно) | если сет собран из аудиоклипов и нужно «слушать» мастер |

Всё ставится одной командой: `bash tools/mac_setup.sh`. Скрипт создаёт окружение
Python 3.12, ставит зависимости, подключает аддон Blender и копирует Remote Script
в библиотеку Ableton. Вручную то же самое:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh           # менеджер Python (или: brew install uv)
cd ~/Asuna/myrmex                                         # папка проекта
uv venv --python 3.12 ~/.venvs/myrmex
source ~/.venvs/myrmex/bin/activate
uv pip install -e ".[live]"                               # numpy, mido, python-rtmidi, aalink (Link), sounddevice
myrmex ports                                              # проверка: MIDI-порты, аудиовходы, Ableton Link
```

---

## 1. Подготовить персонажа (один раз на каждый GLB из Hunyuan3D)

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --python blender/scripts/prepare_character.py -- \
    --glb ~/Downloads/character.glb --out ~/Myrmex/character_live.blend --height 1.70
```

Скрипт импортирует и чистит модель, анализирует форму (конечности находятся по
самой геометрии), подбирает риг, считает веса, добавляет Armature и Corrective
Smooth, сглаживает поверхность, ставит материал «чёрный хром» и студию. Результат:

* `character_live.blend`: персонаж со встроенным описанием рига;
* `character_live.rig.json`: то же описание для `myrmex live --rig`.

Флаги: `--material keep` сохраняет текстуры из GLB, `--material chrome|liquid_metal|gunmetal|ceramic|clay|iridescent`
ставит другой материал, `--smooth 0` отключает сглаживание.

---

## 2. Аддон в Blender

```bash
ln -s ~/Asuna/myrmex/blender/myrmex_blender \
   "$HOME/Library/Application Support/Blender/5.2/scripts/addons/myrmex_blender"
```

Blender → Settings → Add-ons → включить **Myrmex**. Открыть `character_live.blend`,
в 3D-окне нажать **N**, вкладка **Myrmex**:

1. **Armature**: выбрать `Rig`.
2. **Engine**:
   * *Separate process* (рекомендуется): движок запускается в Терминале (шаг 3).
     Так Blender не делит с движком одно ядро и один GIL.
   * *Inside Blender*: движок работает потоком внутри Blender, всё в одно нажатие.
     Для Link и MIDI нужно поставить `aalink`/`mido` в Python самого Blender.
     OSC (Remote Script, VCV) работает без установки.
3. **Start Live**, затем кнопка с камерой: взгляд через живую камеру.
4. Режим отображения: *Material Preview* или *Rendered* (EEVEE).

Кнопки **Live camera / Lights follow / Endless floor** определяют, что следует за
персонажем. Он идёт вперёд бесконечно, а пол и свет едут вместе с ним.

---

## 3. Запустить движок

```bash
source ~/.venvs/myrmex/bin/activate
myrmex live --rig ~/Myrmex/character_live.rig.json
```

Раз в 2 секунды печатается строка состояния:

```
[  12.2s] clock=link  bpm=124.0 beat=24.59 PLAY walk peers=1 notes=34 section=build cam=side_track tick=2.1ms …
```

* `clock`: откуда берётся ритм (см. шаг 4);
* `PLAY/stop` и `walk/HOLD`: идёт или стоит в позе;
* `notes`: сколько событий пришло (если 0, реакций на удары не будет);
* `tick`: время шага симуляции (бюджет 8.3 мс при 120 Гц).

Полезные флаги: `--style catwalk|swagger|heels|natural`, `--latency 45` (мс, см. §7),
`--record ~/Myrmex/takes` (запись дубля), `--midi auto`, `--audio "BlackHole 2ch"`,
`--config mapping.json` (своя раскладка), `--clock link|osc|midi|onsets|internal`.

---

## 4. Подключить Ableton (можно комбинировать)

### A. Ableton Link: ритм без настройки
В Live: Settings → **Link, Tempo & MIDI** → *Show Link Toggle*, затем включить
**LINK** в левом верхнем углу. Там же включить **Start Stop Sync**: тогда Play и
Stop в Live запускают и останавливают походку. Движок видит `peers=1`, темп и фазу
такта. Link даёт только ритм. Для реакций на удары добавь B, C или D.

### B. Remote Script «Myrmex» (рекомендуется)
Скопируй папку `ableton/remote_script/Myrmex` в
`~/Music/Ableton/User Library/Remote Scripts/`, перезапусти Live, затем Settings →
Link, Tempo & MIDI → **Control Surface: Myrmex** (Input/Output: None).

Скрипт шлёт по OSC на `127.0.0.1:9100`:
* транспорт: позицию, темп, Play/Stop, размер;
* **ноты играющих MIDI-клипов на 2 такта вперёд**. Персонаж «знает» удары заранее,
  и реакции попадают точно в звук, а не с опозданием.

Группу дорожки (кик, бас, лид…) скрипт определяет по названию: *Kick*, *Snare*,
*Hat*, *Bass*, *Lead*, *Pad*, *FX*… Для Drum Rack каждая пэд-нота разбирается по
General MIDI (36 кик, 38 снейр, 42 хэт…). Аудиодорожки нот не дают: для них вариант D.

### C. MIDI через IAC (для живой игры на клавишах и контроллерах)
1. macOS: «Настройка Audio-MIDI» → Окно → Студия MIDI → **IAC Driver** → *Устройство подключено*.
2. В Live: новая MIDI-дорожка «to Myrmex». *MIDI From*: дорожка с барабанами
   (Post FX), *MIDI To*: **IAC Driver Bus 1**, канал 1. Так инструмент продолжает
   звучать, а копия нот уходит в движок.
3. Settings → Link, Tempo & MIDI → MIDI Ports → Output *IAC Driver*: включить **Sync**
   (MIDI clock) и **Track**.
4. `myrmex live --rig … --midi auto`

Раскладка по умолчанию (меняется через `--config`):

| MIDI-канал | группа | | CC | ручка |
|---|---|---|---|---|
| 1, 10 | Drum Rack / GM: 36 кик, 38/40 снейр, 42/46 хэты… | | 1, 16 | energy (энергия) |
| 2 | bass | | 2, 17 | stride (длина шага) |
| 3 | melody | | 3, 18 | sway (бёдра) |
| 4 | harmony | | 4, 19 | style (стиль походки) |
| 5 | fx | | 5, 20, 64 | hold (встать в позу) |
| 6 | texture | | 6, 21 | camera (следующий план) |
| 7 / 8 / 9 | kick / snare / hats | | 22 | pose (эффектная поза) |
| 11–16 | perc, melody, harmony, texture, fx | | 23 | flourish (жест) |

### D. Аудио: если трек собран из аудиоклипов
1. Установи **BlackHole 2ch**. В «Настройке Audio-MIDI» создай *Multi-Output Device*
   из динамиков и BlackHole и выбери его выходом в Live.
2. `myrmex live --rig … --audio "BlackHole 2ch"`

Движок слушает мастер и сам выделяет кик (низы, огибающая), снейр (1–4.5 кГц) и
хэты (6–16 кГц). Задержка анализа ~5–10 мс. Удары вне доли при включённом Link
считаются басом, а не киком. Если клока нет, ритм берётся из самих ударов
(«follow kicks»).

---

## 5. VCV Rack 2

* **Удары**: модуль Core **CV-Gate** → устройство *IAC Driver*, канал 1. Назначь входам
  ноты 36 (кик), 38 (снейр), 42 (хэт): клик по ячейке и нота.
* **Ручки**: модуль Core **CV-CC** → IAC, CC 16–23 (energy, stride, sway, style, hold,
  camera, pose, flourish), CV 0–10 В.
* **Ритм**:
  * тактовый сигнал 24 PPQN на вход **CLK** модуля **CV-MIDI** (→ IAC) даёт MIDI clock;
  * либо без клока: движок сам подстроится под кики (`clock=onsets`);
  * либо модуль Ableton Link из VCV Library.
* **OSC** вместо MIDI: модуль trowaSoft **cvOSCcv** → `127.0.0.1:9100`. Адреса
  `/ch/1`…`/ch/8` по умолчанию сопоставлены с kick, snare, hats, perc, bass, energy,
  stride, camera. Гейт > 0.5 В — удар, остальное — ручки. Адреса меняются в модуле
  и в `mapping.json`.

---

## 6. Ручки и триггеры

| Имя | Диапазон | Что делает |
|---|---|---|
| `energy` | 0…1 | интенсивность: длина шага, покачивание, размах рук, «полтемпа» на малых значениях |
| `stride` | 0…1 | длина шага |
| `sway` | 0…1 | амплитуда бёдер |
| `style` | 0…1 | catwalk → swagger → heels → natural (плавный переход ~1 с) |
| `hold` | 0/1 | стоять в позе, пока включено |
| `camera` | триггер | склейка на следующий план на ближайшей сильной доле |
| `pose` / `pose:look_back` | триггер | эффектная остановка на 4 доли |
| `flourish` / `flourish:hair_touch` | триггер | жест на ходу (рука на бедро, волосы, плечо, взгляд) |

Кроме CC и гейтов, можно слать OSC напрямую: `/myrmex/control energy 0.8`,
`/myrmex/trigger camera`, `/myrmex/trigger pose:look_back`.

Свой `mapping.json` (передаётся через `--config`) дополняет таблицы по умолчанию:

```json
{"midi_channels": {"1": "gm", "3": "bass"},
 "midi_cc": {"74": "energy", "71": "sway"},
 "osc": {"/vcv/kick": "kick", "/vcv/knob1": "energy"}}
```

---

## 7. Задержка и точность

* Шаги ставятся на сетку долей, взятую **на `--latency` мс вперёд**. Каблук
  появляется на экране ровно в долю, хотя кадр рисуется с задержкой. На M4 Pro
  подойдёт 40–70 мс. Настройка: включи простой бочонок 4/4 и подбирай, пока удар
  каблука не совпадёт с киком на слух.
* С Remote Script реакции на ноты тоже точные: ноты известны заранее. MIDI и
  аудио приходят в момент звука, поэтому реакция видна чуть позже (физика
  мышц ~30–80 мс плюс кадр).
* У каждого персонажа своя «подача»: шаг ложится на 5–30 мс позади доли, как у
  живого танцора (задаётся `--seed`).

## 8. Производительность вьюпорта

* Движок держи отдельным процессом: Blender тогда тратит на приём позы ~1 мс на кадр.
* Галочка **Fast viewport** (включена по умолчанию) на время живого режима отключает
  во вьюпорте Corrective Smooth. На меше в 150k вершин он стоит десятки миллисекунд
  на кадр, а сама деформация Armature с сохранением объёма занимает единицы мс.
  В рендере сглаживание суставов остаётся.
* Если FPS всё равно мал: Scene → Simplify, или готовь персонажа с `--work-faces 150000`.
* Material Preview быстрее Rendered. Motion blur в EEVEE есть только в рендере, не во вьюпорте.

## 9. Проверка без Ableton

```bash
myrmex live --rig character_live.rig.json      # терминал 1
myrmex simulate --mode both                    # терминал 2: «виртуальный Ableton» играет демо-трек
myrmex monitor --seconds 20                    # терминал 3 (или сразу Blender): что приходит
```

## 10. Дубль → финальный рендер

```bash
myrmex live --rig character_live.rig.json --record ~/Myrmex/takes      # Ctrl+C сохранит take_*.npz
/Applications/Blender.app/Contents/MacOS/Blender -b ~/Myrmex/character_live.blend \
  --python blender/scripts/render_cinematic.py -- \
  --performance ~/Myrmex/takes/take_20260924_210000.npz --audio mix.wav --out film.mp4 --engine eevee
```

## 11. Неполадки

| Симптом | Что проверить |
|---|---|
| В панели «waiting for poses» | запущен ли `myrmex live`; совпадает ли порт (9101); не блокирует ли файрвол Python |
| `peers=0` при включённом Link | Link включён в Live? оба приложения в одной сети? файрвол macOS пропускает Python? |
| Персонаж стоит (HOLD), музыка играет | нет нот (`notes=0`) и транспорта: включи Start Stop Sync в Link, или Remote Script, или MIDI |
| `MIDI … not found` | IAC Driver не включён. Посмотри точное имя порта: `myrmex ports` |
| не ставится `python-rtmidi` | используй Python 3.12 (`uv venv --python 3.12`) |
| рывки в Blender | движок во внешнем процессе, Corrective Smooth выключен во вьюпорте, Material Preview |
| шаги «плывут» относительно звука | подстрой `--latency`; для MIDI clock в Live есть *MIDI Clock Sync Delay* |
