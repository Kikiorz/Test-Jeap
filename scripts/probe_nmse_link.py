#!/usr/bin/env python3
"""Does the correction help most where the latent is predicted most accurately?

Across arms we found that improving the latent head's accuracy does not improve
the action. That is a between-arm comparison, which is coarse. This is the
within-experiment version: for each of the 200 probe batches, compare the
correction's benefit (flow with the correction silenced minus flow with it) to
that batch's latent NMSE.

If accuracy mediates the correction, the correlation should be clearly negative -
the correction should help most on the batches it predicts best. If it is
approximately zero, the correction is using the latent as an identification
signal rather than as a prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-correction", type=Path,
                   default=Path("/workspace/artifacts/con2/nmselink/robotwin_ab_a.json"),
                   help="run that records latent_nmse_b0.05 per batch")
    p.add_argument("--silenced", type=Path,
                   default=Path("/workspace/artifacts/con2/hp/robotwin_ab_base.json"))
    p.add_argument("--reference", type=Path,
                   default=Path("/workspace/artifacts/con2/hp/robotwin_ab_a.json"),
                   help="earlier run of the same row, used as a consistency check")
    return p.parse_args()


def series(path: Path, key: str):
    records = json.loads(path.read_text())["records"]
    records.sort(key=lambda r: r["batch"])
    return np.asarray([r[key] for r in records], dtype=float)


def main() -> None:
    args = parse()
    nmse = series(args.with_correction, "latent_nmse_b0.05")
    flow_on = series(args.with_correction, "flow_b0.05_true")
    flow_off = series(args.silenced, "flow_b0.05_true")
    flow_ref = series(args.reference, "flow_b0.05_true")

    if not (len(nmse) == len(flow_on) == len(flow_off) == len(flow_ref)):
        raise SystemExit("batch counts disagree; the runs are not paired")

    print(f"batches: {len(nmse)}")
    print(f"consistency: this run's flow vs the earlier run of the same row: "
          f"max |diff| = {np.abs(flow_on - flow_ref).max():.3e}")
    benefit = flow_off - flow_on          # >0 means the correction helps
    print(f"correction benefit per batch: mean {benefit.mean():+.3e} "
          f"({benefit.mean() / flow_off.mean() * 100:+.2f}% of the silenced flow)")
    print(f"latent NMSE per batch:        mean {nmse.mean():.4f} "
          f"range {nmse.min():.4f}..{nmse.max():.4f}")
    print()

    for name, x in (("latent NMSE", nmse),):
        r, p = scipy.stats.pearsonr(x, benefit)
        rho, p_s = scipy.stats.spearmanr(x, benefit)
        print(f"benefit vs {name}: pearson r = {r:+.3f} (p = {p:.3g})   "
              f"spearman rho = {rho:+.3f} (p = {p_s:.3g})")

    # Split by median NMSE for an effect size that does not assume linearity.
    median = np.median(nmse)
    good = benefit[nmse <= median]
    bad = benefit[nmse > median]
    print()
    print(f"benefit on the better-predicted half ({len(good)} batches): {good.mean():+.3e}")
    print(f"benefit on the worse-predicted half  ({len(bad)} batches): {bad.mean():+.3e}")
    t, p = scipy.stats.ttest_ind(good, bad, equal_var=False)
    print(f"difference between halves: {good.mean() - bad.mean():+.3e}  (t = {t:+.2f}, p = {p:.3g})")


if __name__ == "__main__":
    main()
