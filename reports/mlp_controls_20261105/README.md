# Earlier MLP capacity controls

Recovered per-fit metrics and predictions from the completed fixed-recipe study. These are existing experiments, not new training. All use the same 398 development-validation clips and seeds 101/202/303. The CNN and BiGRU reference fits overlap the later matrix; do not count them as independent replications.

The pooled and tokenwise MLPs share the CNN encoder and have 252,036 total parameters. They do not use PANNs features. Their results cannot determine whether a nonlinear classifier helps frozen PANNs.

The results notebook presents the full comparison and paired source intervals.
