# Myrmex + Blender + TouchDesigner

Схема связи:

```
Ableton / MIDI / перчатка ──> Myrmex ──OSC 7000/7001──> TouchDesigner ──> окно на 2-м экране / Syphon / запись .mov
                                │                          ▲
                                └──> Blender ──Syphon "Myrmex"──┘
                  TouchDesigner ──OSC 9100──> Myrmex (статус, события существу)
```

- **Myrmex** каждый кадр шлёт в TouchDesigner (TD) всё, что происходит:
  - музыку (бит, фаза, бочка, энергия);
  - существо (форма, скорость, свечение, где оно на экране, удары, смены формы);
  - камеру (склейки, объектив).
- **Blender** отдаёт в TD картинку своей камеры через **Syphon**. Библиотека уже внутри Myrmex.app, ставить ничего не нужно.
- **TD** накладывает эффекты. Все они привязаны к данным: бочка бьёт аберрацией, удар существа запускает ударную волну *из точки, где оно на экране*, смена формы искажает картинку. Эффектами управляет страница **TouchDesigner** в приложении, в том числе MIDI-ручками.

Сделано под **TouchDesigner 2025.3x**, подходит и бесплатная Non-Commercial. Её ограничения:
- картинка до 1280×1280, поэтому по умолчанию 1280×720;
- запись только в ProRes .mov, H.264 доступен лишь в платной лицензии.

## Первый запуск (один раз)

1. В Myrmex откройте страницу **TouchDesigner**, включите **Sync with TouchDesigner** и нажмите **Set up TouchDesigner…**. Строка-команда окажется в буфере обмена.
2. Откройте TouchDesigner (новый проект).
3. Откройте **Dialogs → Textport and DATs** (или Alt+T), вставьте строку (⌘V) и нажмите Enter.
4. TD построит сеть `/project1/myrmex` и сохранит её как `~/Myrmex/touchdesigner/Myrmex_FX.toe`.

Дальше кнопка **Open TouchDesigner** открывает этот проект сразу.

## Каждый раз

1. **Start engine** в Myrmex, затем **Open in Blender**. Blender сам включит Syphon.
2. **Open TouchDesigner**. Статус на странице покажет «TouchDesigner connected · 60 fps».
3. Выберите пресет или двигайте ползунки. TD следует за приложением, пока в TD включён параметр **Follow**.

Если Myrmex не запущен, TD продолжает работать сам на «демо-пульсе», так что эффекты можно настраивать без музыки.

## Пресеты

| Пресет | Что делает |
|---|---|
| Clean | чистая кинокартинка: мягкое свечение, зерно, виньетка |
| Neon Trails | светящиеся эхо каждого движения, дрейф цвета |
| Glitch Storm | картинка рвётся на ударах, сильная аберрация, HUD |
| Dream | длинные мягкие шлейфы, жидкое свечение |
| Kaleido | калейдоскоп вокруг существа |
| Scanner | неоновый контур и прицельный HUD вокруг существа |
| Liquid | картинка течёт вокруг существа |

Регуляторы:
- **Bloom**, **Trails**, **Chromatic aberration**, **Glitch**, **Liquid warp**, **Shockwaves**, **Kaleidoscope**, **Neon edges**, **Grain**, **Vignette**, **HUD**;
- **Music moves the effects** — насколько эффекты «дышат» музыкой и существом;
- **Exposure / Contrast / Saturation / Colour drift / Trails mix**.

MIDI: назначьте ручку на цель `td_bloom`, `td_trails`, `td_glitch` и т. д. на странице MIDI.

## Вывод и запись

- **Fullscreen output window on display N** — TD откроет полноэкранное окно на выбранном мониторе (проектор, второй экран).
- **Record what TouchDesigner shows** — запись финальной картинки TD в `~/Myrmex/td_recordings/*.mov` (ProRes).
- В TD, в параметрах компонента `myrmex` (выделить и нажать P):
  - **Syphon out** — отдаёт картинку как «Myrmex FX» в OBS, Resolume или MadMapper;
  - **Record** — запись без приложения.
- **Transparent background** в Blender-разделе: Blender отдаёт только существо на прозрачном фоне, а фон рисует TD.

## Тейки

Каждый тейк Myrmex записывает и те же каналы, что уходили в TD. Если открыть тейк в Blender при включённой синхронизации, Blender:
- отдаёт в TD картинку тейка;
- шлёт записанные каналы кадр за кадром.

Поэтому эффекты повторяют живое выступление. Нажмите Play в Blender и запишите результат в TD.

## Что приходит в TD

OSC In CHOP `data_in`, порт 7000, *Strip Prefix Segments* = 1. Каналы:

| Группа | Каналы |
|---|---|
| Музыка | `beat phase bar bpm playing energy bass mid high flux kick snare hats` |
| Существо | `glow arousal surface instab speed size x y z heading sx sy ssize visible regime variant intent tension impact morph event` |
| Камера | `cut shot lens focus fstop` |
| Драйверы эффектов 0..1 | `fx_bloom fx_trails fx_glitch fx_chroma fx_warp fx_flash fx_shake fx_shock fx_hue fx_strobe shock_x shock_y` |
| Стойка эффектов приложения | `c_bloom … c_mix c_preset rec window monitor` |

Текст (OSC In DAT `text_in`, порт 7001): `organism regime intent shot preset recfile`.

Обратно в Myrmex (порт 9100) идут:
- `/myrmex/td/alive f fps` — статус соединения;
- `/myrmex/trigger s creature:<событие>` и `/myrmex/control s <параметр> f <значение>`.

Страница **Events** компонента в TD отправляет события существу.

Логика эффектов — Text DAT `fx_logic` в сети TD: обычный Python, его можно править. Шейдеры лежат в `shader_*`. Повторный запуск строки из шага 3 пересобирает сеть и сохраняет ваши настройки.

## Если что-то не так

- **Чёрная картинка в TD.** Нужны включённые Sync и Send Blender's camera, и Blender, открытый из Myrmex. В TD нажмите **Find Blender picture** или переключите **Picture → Test pattern**, чтобы проверить эффекты.
- **Картинка вверх ногами.** Параметр **Flip** в TD.
- **«waiting for TouchDesigner».** Откройте проект Myrmex_FX.toe. Порты 7000, 7001 и 9100 не должны быть заняты другими программами.
