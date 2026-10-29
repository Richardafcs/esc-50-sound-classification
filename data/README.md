# Data

`ESC-50/meta/esc50.csv`, the original dataset README and LICENSE are retained. `python -m esc50_sound prepare` downloads pinned WAVs into `ESC-50/audio/` and produces a validated feature cache. Raw WAVs and large cache arrays are generated data, excluded from version control. Standardization is fitted on training clips only. The current source-strict roles are 1200 train / 398 validation / 400 designated test clips.
