# Experiment outputs and regression fixtures

New experiments write to `results/runs/`, which is local and ignored by Git. Full matrix checkpoints, optimizer states, process logs and bulk historical studies also stay local.

Only the 43 historical numerical files listed in [the fixture manifest](../tests/historical_fixtures.json) are shared here, together with two archived reference source files elsewhere. They are small fixtures needed for the complete regression suite, not a complete training-data publication.

Team-facing result summaries are in [curated report evidence](../reports/evidence/matrix_20261028/). Published reports and figures are indexed in [reports](../reports/README.md). Do not alter original fixture metrics to make a model appear better; generate a new experiment directory instead.
