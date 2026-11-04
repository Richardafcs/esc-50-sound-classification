"""Discharge small ESC-50 analysis specifications with executable Z3 queries.

An UNSAT result excludes a counterexample to the stated ideal-arithmetic
specification. It does not verify Python, floating-point kernels, a trained
network, statistical coverage, or the provenance/independence of recordings.
No experiment, audio, checkpoint, or file-hash verification is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import z3
except ImportError as exc:
    raise SystemExit("This optional proof command requires z3-solver (python -m pip install z3-solver).") from exc


def queries():
    """Yield (name, assumptions, alleged counterexample, expected, scope)."""
    both, only_a, only_b, neither = z3.Ints("both only_a only_b neither")
    n = both + only_a + only_b + neither
    valid = [both >= 0, only_a >= 0, only_b >= 0, neither >= 0, n > 0]
    correct_a, correct_b = both + only_a, both + only_b
    yield (
        "paired_net_correct_count", valid,
        correct_b - correct_a != only_b - only_a, "unsat",
        "All nonnegative integer 2x2 paired-correctness tables with N>0.",
    )
    yield (
        "paired_difference_bounds", valid,
        z3.Or(only_b - only_a < -n, only_b - only_a > n), "unsat",
        "The paired accuracy difference, after division by N, lies in [-1,1].",
    )
    size, acc_a, acc_b, count_a, count_b = z3.Reals("N accuracy_a accuracy_b count_a count_b")
    yield (
        "paired_accuracy_normalization",
        [size > 0, size * acc_a == count_a, size * acc_b == count_b],
        size * (acc_b - acc_a) != count_b - count_a, "unsat",
        "Exact real normalization of two counts by the same positive denominator.",
    )

    # Three groups are an explicit finite model. Positivity for arbitrary G
    # follows by the ordinary finite-sum argument documented alongside it.
    lengths = z3.Ints("n0 n1 n2")
    weights = z3.Ints("w0 w1 w2")
    differences = z3.Ints("d0 d1 d2")
    group_valid = [x > 0 for x in lengths] + [x >= 0 for x in weights]
    group_valid += [z3.Sum(weights) > 0]
    group_valid += [z3.And(-lengths[j] <= differences[j], differences[j] <= lengths[j]) for j in range(3)]
    denominator = z3.Sum([weights[j] * lengths[j] for j in range(3)])
    numerator = z3.Sum([weights[j] * differences[j] for j in range(3)])
    yield (
        "cluster_resample_positive_denominator_3_groups", group_valid,
        denominator <= 0, "unsat",
        "Three nonempty sources, nonnegative integer multiplicities, at least one draw.",
    )
    yield (
        "cluster_resample_difference_bounds_3_groups", group_valid,
        z3.Or(numerator < -denominator, numerator > denominator), "unsat",
        "Clip-weighted paired difference remains in [-1,1] after whole-source resampling.",
    )
    permutation = (2, 0, 1)
    perm_num = z3.Sum([weights[j] * differences[j] for j in permutation])
    perm_den = z3.Sum([weights[j] * lengths[j] for j in permutation])
    yield (
        "joint_source_permutation_invariance_3_groups", group_valid,
        z3.Or(perm_num != numerator, perm_den != denominator), "unsat",
        "Jointly permuting source size, paired sum, and draw multiplicity changes neither sum.",
    )
    yield (
        "same_cluster_draws_commute_with_fixed_seed_mean",
        [size > 0],
        (sum(z3.Reals("delta_seed0 delta_seed1 delta_seed2")) / 3) / size
        != sum([d / size for d in z3.Reals("delta_seed0 delta_seed1 delta_seed2")]) / 3,
        "unsat",
        "A fixed three-seed average shares the resampled denominator; no seed-independence claim.",
    )

    # P and Q are mean-square energies; u and v are their positive roots.
    # g=10**(snr_db/20)>0 is externally defined, R=g*g. The solver does not
    # reason about the transcendental dB conversion itself.
    p, q, u, v, gain, amplitude = z3.Reals("P Q signal_rms noise_rms ratio_root noise_amplitude")
    noise_valid = [p > 0, q > 0, u > 0, v > 0, gain > 0,
                   u * u == p, v * v == q, amplitude * v * gain == u]
    noise_power = amplitude * amplitude * q
    ratio = gain * gain
    yield (
        "normalized_added_noise_energy", noise_valid,
        noise_power * ratio != p, "unsat",
        "Ideal real arithmetic with positive signal/noise RMS and no clipping or renormalization.",
    )
    yield (
        "normalized_added_noise_positive_power", noise_valid,
        z3.Or(amplitude <= 0, noise_power <= 0), "unsat",
        "A nonzero signal and nonzero centered noise have positive scaled-noise power.",
    )

    # Negative controls must be SAT: tempting stronger claims are false.
    a, b, da, db = z3.Reals("size_a size_b delta_a delta_b")
    yield (
        "negative_control_source_mean_equals_clip_mean",
        [a == 1, b == 2, da == 1, db == 0],
        (da + db) / (a + b) != (da / a + db / b) / 2,
        "sat",
        "Counterexample: clip-weighted difference 1/3 versus equally weighted source mean 1/2.",
    )
    yield (
        "negative_control_constant_bootstrap_denominator",
        [lengths[0] == 1, lengths[1] == 2, lengths[2] == 3,
         weights[0] == 3, weights[1] == 0, weights[2] == 0],
        denominator != z3.Sum(lengths), "sat",
        "Drawing three sources with replacement need not preserve six original clips: here it gives three.",
    )
    yield (
        "negative_control_unnormalized_noise_hits_requested_snr",
        [p == 4, q == 1, gain == 1, amplitude == 1],
        noise_power * ratio != p, "sat",
        "Without RMS normalization, signal power 4 plus noise power 1 misses requested ratio 1.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the machine-readable proof receipt")
    parser.add_argument("--timeout-ms", type=int, default=10000, help="Per-query solver time limit")
    args = parser.parse_args(argv)
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")
    checks = []
    for name, assumptions, counterexample, expected, scope in queries():
        solver = z3.Solver()
        solver.set(timeout=args.timeout_ms)
        solver.add(*assumptions)
        premises = solver.check()
        solver.add(counterexample)
        answer = solver.check()
        result = {"property": name, "expected": expected, "actual": str(answer),
                  "premises": str(premises),
                  "passed": str(answer) == expected and premises == z3.sat, "scope": scope}
        if answer == z3.sat:
            model = solver.model()
            result["counterexample"] = {str(k): str(model[k]) for k in model.decls()}
        elif answer == z3.unknown:
            result["reason_unknown"] = solver.reason_unknown()
        checks.append(result)
    receipt = {
        "schema_version": 1,
        "solver": "Z3", "solver_version": z3.get_version_string(),
        "verification_kind": "SMT counterexample queries over stated abstract specifications",
        "scope_limit": "Not a proof of Python/PyTorch correspondence, floating-point behavior, bootstrap coverage, dataset independence, or learned-model robustness.",
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "proved_properties": sum(c["passed"] and c["expected"] == "unsat" for c in checks),
        "confirmed_negative_controls": sum(c["passed"] and c["expected"] == "sat" for c in checks),
    }
    encoded = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
