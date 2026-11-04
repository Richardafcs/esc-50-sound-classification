# Follow-up: formal properties and implementation boundary

This work verifies small mathematical specifications used by the follow-up analysis and waveform perturbation. It does not infer model robustness from correct algebra. There is no file-hash check in this command.

```sh
python scripts/prove_followup_properties.py
# Optional, when retaining the receipt:
python scripts/prove_followup_properties.py --output reports/followup_20261104/formal_properties.json
```

The optional dependency is `z3-solver`. Z3 4.15.2 discharged nine counterexample queries as `unsat` and found the three expected negative-control counterexamples as `sat` on 4 November 2026. Every assumption set is also checked as `sat`, excluding proof by inconsistent premises. Each query has a time limit; `unknown` or an unexpected answer fails the command.

## Paired correctness

For the same ordered set of \(N>0\) clips, let \(c,a,b,z\) count both-correct, A-only-correct, B-only-correct and neither-correct outcomes. They are nonnegative integers with \(N=c+a+b+z\). Correct counts are \(C_A=c+a\) and \(C_B=c+b\). Consequently,

\[
C_B-C_A=b-a,\qquad \operatorname{Acc}_B-\operatorname{Acc}_A=\frac{b-a}{N}\in[-1,1].
\]

The solver verifies the integer count identity, count bounds and real-valued common-denominator normalization. Accuracy is dimensionless; multiply the difference by 100 to report percentage points. Example: \((166-9)/398=0.394472\ldots\), or **39.45 pp**. The theorem requires aligned clips and labels; CSV membership/label checks in the analysis implementation establish that input contract. It does not determine whether the predictions are useful or correct in the real world.

## Whole-source resampling with clip weighting

For original source \(j\), let \(n_j>0\) be its clip count and \(d_j=\sum_{i\in j}(\mathbf{1}[B_i=y_i]-\mathbf{1}[A_i=y_i])\). Thus \(-n_j\le d_j\le n_j\). In one bootstrap draw, integer \(w_j\ge0\) records how often source \(j\) is sampled; at least one multiplicity is positive. The resampled, clip-weighted accuracy difference is

\[
\Delta^*=\frac{\sum_jw_jd_j}{\sum_jw_jn_j}.
\]

The denominator is positive, but it usually differs from the original clip count. It must be recomputed for each draw. Since each weighted summand obeys \(-w_jn_j\le w_jd_j\le w_jn_j\), summing proves \(\Delta^*\in[-1,1]\) for any finite number of nonempty groups. Joint reordering of all group attributes preserves both sums.

The executable SMT model covers **three arbitrary groups** with unbounded integer sizes and multiplicities, including positivity, bounds and a cyclic joint permutation. The general finite-sum argument above extends to 295 sources; the script does not claim to verify the Python loop for 295 groups.

Two `sat` controls demonstrate meaningful implementation mistakes:

- Source sizes \((1,2)\) and paired sums \((1,0)\) yield clip-weighted difference \(1/3\), whereas averaging source-specific accuracies equally yields \(1/2\). These are different estimands.
- Source sizes \((1,2,3)\) and draw multiplicities \((3,0,0)\) contain three clips, not the original six. Holding the denominator fixed is incorrect.

For three **fixed trained models per family**, use identical source multiplicities across seeds before taking their mean. Averaging then commutes with division by the shared resampled denominator; the script verifies this real-arithmetic identity. Repeated predictions on the same clips are not additional independent recordings. The conditional source-bootstrap interval does not integrate over the distribution of future training seeds or fresh noise realizations. Even identical outputs from three seeds remain possible, so the existence of three outputs alone cannot prove statistical independence.

## Ideal added-noise SNR

For waveform \(x\) and centered noise vector \(e\), define mean-square energies \(P=\operatorname{mean}(x^2)>0\), \(Q=\operatorname{mean}(e^2)>0\). These have squared digital-amplitude units, rather than physical sound-pressure units. Set \(u=\sqrt P>0\), \(v=\sqrt Q>0\), \(g=10^{s/20}>0\), and scale \(\alpha=u/(vg)\). Here \(s\) is the requested added-noise SNR in dB, and \(R=g^2=10^{s/10}\) is the dimensionless power ratio. With output \(x+\alpha e\),

\[
\operatorname{mean}((\alpha e)^2)=\alpha^2Q=\frac P R,\qquad
\frac{P}{\operatorname{mean}((\alpha e)^2)}=R.
\]

The SMT proof uses exact real constraints \(u^2=P\), \(v^2=Q\), \(\alpha vg=u\), and verifies positive added-noise power and \(\alpha^2Qg^2=P\). The exponential dB conversion is an external definition, not an SMT-proved transcendental function. The noise must be centered **before** measuring \(Q\). No clipping or subsequent amplitude renormalization is permitted for this identity. A third `sat` control shows why unnormalized noise is insufficient: \(P=4,Q=1,\alpha=1,R=1\) gives actual ratio 4, not 1.

If \(P=0\), preserving silence is a sensible implementation convention, but a finite SNR is undefined. If \(Q=0\), the normalization precondition fails. Original ambient noise is unknown; the formula describes only artificially added noise. It says nothing about every real-world noise type.

## Connection to executable behavior

The inference implementation needs numerical checks of the measured added-noise SNR for nonzero finite float32 waveforms, silence handling, unchanged clean input, paired use of the same corrupted clip across models/seeds, and invariance to task ordering or batching where claimed. These test the correspondence between an implementation and the specification; they remain finite numerical checks, not proofs of all IEEE-754 executions. Dataset overlap, label validity, prediction alignment and source grouping require their own semantic checks. Confidence-interval coverage and learned-model robustness remain empirical/statistical questions.
