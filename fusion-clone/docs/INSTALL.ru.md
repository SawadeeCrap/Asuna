# Установка Fusion Clone (VCV Rack 2, macOS на Apple silicon; Linux и Windows — по аналогии)

*(English version: `docs/INSTALL.md`.)*

**Готового бинарного файла в репозитории нет.** Плагин для Rack — это скомпилированный код под конкретную платформу, а среда, в которой я
писал код, — облачный Linux-контейнер без вашего Mac и без Rack. Поэтому есть два пути: **один раз собрать плагин из исходников (около
5 минут)** или **скачать готовый пакет, который собирает GitHub** (`.github/workflows/fusion-clone.yml`, артефакт `FusionClone-mac-arm64`, см. раздел Б).

## А. Сборка из исходников на Mac

1. **Компилятор.** Xcode Command Line Tools (если `clang --version` уже работает — пропустите):
   ```sh
   xcode-select --install
   ```
2. **Rack SDK.** На https://vcvrack.com/downloads скачайте **Rack SDK 2.x для Mac (arm64)** — той же версии 2.x, что и ваш Rack
   (Rack → Help → About) — и распакуйте в `~/Rack-SDK`. (SDK — это только заголовки и makefile'ы, исходники Rack не нужны.)
3. **Исходники, сборка, установка.**
   ```sh
   git clone https://github.com/SawadeeCrap/Asuna.git
   cd Asuna && git checkout claude/jolly-einstein-e9h02j
   cd fusion-clone
   export RACK_DIR="$HOME/Rack-SDK"
   make -j"$(sysctl -n hw.ncpu)"
   make install
   ```
   `make install` копирует плагин в `~/Library/Application Support/Rack2/plugins-mac-arm64/FusionClone/`.
   `make dist` вместо этого собирает пакет `dist/FusionClone-<версия>-mac-arm64.vcvplugin`.
4. **Перезапустите Rack.** Добавьте модуль: правый клик по стойке → поиск **Fusion Clone** (бренд *Asuna*).

## Б. Готовый пакет, собранный GitHub (компилятор не нужен)

Откройте репозиторий на GitHub → **Actions** → последний запуск *fusion-clone* для ветки `claude/jolly-einstein-e9h02j` → **Artifacts** →
`FusionClone-mac-arm64`. Распакуйте и дважды кликните по `.vcvplugin` (Rack установит его сам) либо распакуйте содержимое архива в
`~/Library/Application Support/Rack2/plugins-mac-arm64/`. Если macOS не даёт загрузить плагин, потому что он скачан из интернета:
```sh
xattr -dr com.apple.quarantine "$HOME/Library/Application Support/Rack2/plugins-mac-arm64/FusionClone"
```
В том же запуске на Apple-silicon-раннере выполняются DSP-тесты и бенчмарк CPU (вкладка Summary).

## Первое использование

* Подключите **OUT** вашего Fusion VCO2 (через аудиоинтерфейс, в модуль *Audio* в Rack) к входу **AUDIO IN**, а **OUT L / OUT R** — на микшер.
  Номинальный уровень ±5 В; DSP от уровня не зависит, но не допускайте клиппинга.
* Крутите **VOICES** (1 = только оригинал, 16 = оригинал + 15 клонов). На дисплее появится **LOCK xx.x Hz**, загорится лампа LOCK через
  несколько периодов после начала ноты; до этого слышен только оригинал. Модуль монофонический (у полифонических кабелей берётся канал 1).
* Остальное — в `README.md` (панель) и `docs/MANIFEST.md`; расширенные настройки — в контекстном меню модуля.

## Если что-то не работает

| Симптом | Что делать |
|---|---|
| `make`: *plugin.mk / RACK_DIR not found* | `RACK_DIR` должен указывать на распакованную папку SDK (в ней лежит `plugin.mk`). |
| Модуля нет в Rack / в логе «plugin built for another version» | Возьмите SDK той же версии 2.x, что и Rack. Лог: `~/Library/Application Support/Rack2/log.txt`. |
| Ошибка линковки про `pffft_…` | SDK не экспортирует pffft: в `Makefile` в `FLAGS` замените на `-DFC_FFT_RACK` (обёртка FFT из Rack) или уберите `-DFC_FFT_PFFFT` (встроенный FFT, медленнее). |
| Подозрение на проблему со SIMD | Пересоберите с переносимым скалярным кодом: `FLAGS=-DFC_NO_SIMD make -j … && make install`. |
| macOS блокирует скачанный плагин | `xattr -dr com.apple.quarantine …` как выше. |
| Дисплей висит на *ACQUIRING* / *PASS-THRU* | На входе должен быть стабильный монофонический периодический сигнал (saw/tri/pulse/sine, можно с суб-осциллятором); шум, аккорды и постоянная составляющая по замыслу не захватываются. Проверьте уровень входа. |
| Слишком высокая загрузка CPU | Снизьте QUALITY (ECO/BALANCED) или VOICES; цифры в `docs/BENCHMARKS.md` получены на x86-Linux, бенчмарк в запуске GitHub даёт данные для Apple silicon. |

## Что мне важно узнать после первого запуска

Этот код ни разу не запускался внутри Rack (там, где он писался, нет ни Rack, ни экрана, ни звука). Полезнее всего: лог Rack (`log.txt`),
если плагин не появился; скриншот панели; загрузка CPU при VOICES = 16 для каждого QUALITY; и то, как это **звучит** рядом с настоящими
осцилляторами. Измерения, которые закрывают открытые вопросы про реальный Fusion VCO2, описаны в `docs/REFERENCE_PROTOCOL.md`.
