# Executed research notebooks

- [Complete ESC-50 study](ESC50_Results.ipynb): the full comparison, earlier MLP controls, all 36 scientific figures, experiment ledger and literature review. Saved-result analysis is the default; original PANNs extraction/refitting is optional.
- [Transfer robustness and matched heads](ESC50_Transfer_Followup.ipynb): the clean/10 dB comparison and six fixed-budget PANNs-head fits.
- [External FSD transfer](ESC50_External_Transfer.ipynb): frozen-system evaluation on 259 mapped external recordings after known ESC-50 recording-ID exclusion.
- [Short-recording input-policy diagnostic](ESC50_Duration_Diagnostic.ipynb): fixed right/native/left comparison, paired uncertainty, prediction transitions, architecture exclusions and identical-input controls.

All five retain executed Colab tables and image outputs. Fresh model runs require a GPU and download official audio/backbone inputs. Completed outputs are reused. Historical scratch-model fits are identified separately and are not retrained by these notebooks. The external benchmark remains unchanged by the subsequent exploratory input-policy diagnostic.

- [Gaussian-to-recorded-noise transfer](ESC50_Noise_Transfer.ipynb): two matched linear head fits on frozen PANNs, six recorded-background conditions, paired effects and clean-performance trade-off.
