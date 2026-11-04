# Final ESC-50 research report - 5 November 2026

Later on 5 November, the separately authorized [PANNs transfer phase](../panns_20261105/README.md) was executed. The decision and unexecuted proposal below document the earlier closure of the original study.

[English PDF](../pdf/ESC50_Final_Research_Report_EN.pdf) - 35 pages, all 19 scientific figures, complete numerical appendices.

[Editable English source](Final_Research_Report_EN.md) · [Figure atlas](figure_atlas.json) · [Six-role model-decision meeting](../../docs/research_meetings/2026-11-05_model_decision.md)

## Model decision

The existing architecture study already includes nine attention-specific runs: temporal, channel, and combined attention, each with three training seeds. Channel attention achieved 74.37% mean validation accuracy versus 73.87% for ordinary BiGRU. The +0.503 pp paired difference has a conditional 95% source-bootstrap interval of [-1.217, +2.167] pp. These results do not establish a reliable attention advantage.

No additional attention training is required to complete the current research questions. If the goal changes to higher predictive accuracy, frozen AudioSet-pretrained audio embeddings plus a deterministic linear head are a separate proposed phase. No such model was downloaded, trained or evaluated in this study. Repeating an otherwise identical deterministic head fit with nominal seed changes would not provide independent evidence.

## Included evidence

- 65 original tasks, including 60 neural fits; Appendix C lists all 62 fitted models (including two classical baselines).
- Completed three-seed robustness: all 54 model/seed/condition combinations.
- Temporal-order probe: all 30 model/seed/condition combinations.
- All 19 scientific figures: 11 development/follow-up figures and eight historical baseline figures, with separate evaluation scopes.
- Attention paired effects: [numerical output](attention_paired_effects.json).
- Completed robustness overview: [PNG](three_seed_robustness.png) · [SVG](three_seed_robustness.svg).

## Interpretation and verification

This round reused recorded predictions and existing checkpoints; it did not train a new model. The report source was drafted and reviewed through six Chrome ChatGPT roles, then integrated locally. Role agreement is not independent experimental replication. The PDF was rendered and visually inspected, including its result tables and figure atlas. No new ML tests or file-hash checks were performed.

All official folds have historical exposure. Findings remain development-validation results on 398 clips from 295 recording sources. Bootstrap intervals condition on the fitted models and fixed perturbation realizations; they do not cover future training seeds or noise types. Historical fold-5 results are retained separately and are not fresh confirmation.
