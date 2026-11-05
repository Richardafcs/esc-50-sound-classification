# Executed research notebooks

- [Complete ESC-50 study](ESC50_Results.ipynb): all main results, restored earlier MLP controls, all 22 original scientific figures and four new follow-up figures. Original PANNs extraction/refitting is optional; the default recomputes saved results.
- [Transfer robustness and matched heads](ESC50_Transfer_Followup.ipynb): the executable clean/10 dB comparison and six fixed-budget PANNs-head fits, with complete outputs.

All three were executed in Colab and retain tables and image outputs. A fresh follow-up run requires a Colab GPU runtime and downloads the official audio/backbone; completed outputs are reused without repeating head fits. Historical scratch-model fits are identified separately and are not retrained by either notebook.

- [External FSD transfer](ESC50_External_Transfer.ipynb): one-pass frozen-system evaluation on 259 mapped external recordings after known ESC-50 recording-ID exclusion. No training or added noise.
