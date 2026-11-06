# Executed research notebooks

- [Complete ESC-50 study](ESC50_Results.ipynb): integrated comparisons, experiment ledger, literature review and all 40 scientific figures.
- [Training-noise distribution transfer](ESC50_Noise_Distribution.ipynb): one recorded-background LR fit, six reserved recordings, a fixed SNR grid, paired comparisons and four editable figures.
- [Gaussian-to-recorded-noise transfer](ESC50_Noise_Transfer.ipynb): two matched linear fits on frozen PANNs, six recorded-background conditions and clean-performance tradeoffs.
- [Transfer robustness and matched heads](ESC50_Transfer_Followup.ipynb): clean/10-dB comparison and six fixed-budget PANNs-head fits.
- [External FSD transfer](ESC50_External_Transfer.ipynb): frozen-system evaluation on 259 mapped external recordings after known ESC-50 recording-ID exclusion.
- [Short-recording input-policy diagnostic](ESC50_Duration_Diagnostic.ipynb): fixed right/native/left comparison, uncertainty, prediction transitions and architecture exclusions.

All six retain actual Colab tables and image outputs. Saved-evidence analysis is the default. Fresh model computation requires official audio/backbone inputs and a GPU; completed outputs are reused. Historical scratch fits are identified separately and are not retrained by these notebooks. Raw audio, feature caches and document-production utilities stay outside Git.
