# Myrmex: персонажи из Hunyuan3D, которые живут под музыку из Ableton и VCV

**Запускаешь Ableton или VCV Rack, и персонаж в Blender в реальном времени идёт,
реагирует на удары и держит ритм.** Шаги ложатся на долю, камера меняет планы
на сильных долях. Когда музыка останавливается, персонаж встаёт в позу и ждёт.

```
Hunyuan3D GLB ─► prepare_character (Blender: чистка, авто-риг, веса, look-dev)
Ableton / VCV ─► myrmex live (120 Гц) ─► UDP ─► Blender: аддон Myrmex Live (EEVEE, живая камера)
                                  └──────► запись дубля ─► финальный рендер (EEVEE / Cycles)
```

## Быстрый старт (Mac)

```bash
uv venv --python 3.12 ~/.venvs/myrmex && source ~/.venvs/myrmex/bin/activate
uv pip install -e ".[live]"

# 1) персонаж: GLB -> готовый .blend + описание рига
/Applications/Blender.app/Contents/MacOS/Blender -b --python blender/scripts/prepare_character.py -- \
    --glb character.glb --out character_live.blend

# 2) движок (Ableton: включи Link или Remote Script «Myrmex»)
myrmex live --rig character_live.rig.json

# 3) Blender: открыть character_live.blend -> N -> Myrmex -> Start Live
```

Без Ableton проверяется так: `myrmex simulate --mode both` («виртуальный Ableton») и `myrmex monitor`.

**Подробно: [docs/REALTIME.md](docs/REALTIME.md)**: Ableton (Link, Remote Script, MIDI/IAC,
аудио через BlackHole), VCV Rack (CV-Gate/CV-CC/cvOSCcv), ручки и триггеры, задержка, неполадки.

## Что умеет

- **Живой режим**: часы из Ableton Link, транспорта Remote Script, MIDI clock или по самим кикам.
  Ноты клипов приходят заранее, поэтому персонаж предвосхищает удары. MIDI (каналы, GM-барабаны,
  CC-ручки), OSC (VCV, TouchOSC, Max), аудио-анализ мастера. Стиль походки меняется на лету.
  Поток поз 60 кадров/с, живая камера, свет и бесконечный пол едут за персонажем, дубль пишется.
- **Подиумная походка** (по умолчанию): шаги на сетке долей, перекрёстный шаг, бёдра,
  противовращение плеч, акценты: снейр даёт hip-pop, кик пружинит, хэты дают shimmy.
  Позы на брейках, заморозка перед дропом и рывок на дропе, жесты на ходу.
- **Персонажи из Hunyuan3D**: импорт и нормализация, анализ формы (конечности по геометрии),
  авто-риг гуманоида, геодезические веса и Corrective Smooth, сглаживание поверхности.
  Материалы: чёрный хром, хром, жидкий металл и другие.
- **Музыкальный слой**: признаки, память, структура (интро/билд/дроп/брейк), предиктор грува
  (предвосхищение, удивление, пропуск удара); чтение `.als` (Live 8–12), MIDI, стемов.
- **Кино-камера**: 9 типов планов в системе координат идущего персонажа, склейки на сильных долях.
  Офлайн-версия сглаживает без фазового сдвига, живая идёт с упреждением.

Документы: [живой режим](docs/REALTIME.md) · [исследование](docs/RESEARCH.md) · [улучшенный промпт v2](docs/PROMPT_v2.md)

```bash
python -m pytest -q      # тесты
```
