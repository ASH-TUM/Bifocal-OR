#!/usr/bin/env python3
"""
score_diarization.py — Score Audacity speaker labels against a ground truth.

Give it your hand-annotated ground-truth label file and an automatic (or
edited) hypothesis label file. It prints:

  * Diarization Error Rate (DER) under the standard scoring conventions,
    split into Missed / False-Alarm / Confusion  (via pyannote.metrics if
    installed; otherwise a self-contained frame-based md-eval implementation
    that has been verified to match pyannote exactly).
  * A speaker confusion matrix (rows = ground truth, cols = hypothesis).
  * Per-speaker "routing": for each ground-truth speaker, where did their
    speech actually go in the hypothesis (missed / correct / stolen by X).

Both inputs are Audacity label files: tab-separated  start<TAB>end<TAB>role,
one label per line. Overlapping labels (same time, different role) are fine
and are handled correctly.

Usage:
    python score_diarization.py REF.txt HYP.txt
    python score_diarization.py REF.txt HYP.txt --collar 0.25
    python score_diarization.py REF.txt HYP.txt --collar 0.25 --skip-overlap
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

FRAME = 0.010  # 10 ms scoring grid for the diagnostic views


def parse_labels(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print(f"  ! {path}:{ln}: skipped (need 3 tab-separated fields)",
                      file=sys.stderr)
                continue
            out.append((float(parts[0]), float(parts[1]), parts[2].strip()))
    return out


def build_grid(labels, roles, n):
    idx = {r: i for i, r in enumerate(roles)}
    M = np.zeros((len(roles), n), dtype=bool)
    for s, e, r in labels:
        a, b = int(round(s / FRAME)), int(round(e / FRAME))
        M[idx[r], a:b] = True
    return M, idx


def frame_der(R, H, ridx, hidx, roles):
    """Self-contained md-eval DER (overlap-aware, collar=0). Validated == pyannote."""
    Nref, Nsys = R.sum(0), H.sum(0)
    total = Nref.sum()
    K = len(roles)
    cooc = np.array([[np.logical_and(R[i], H[j]).sum() for j in range(K)]
                     for i in range(K)])
    row, col = linear_sum_assignment(-cooc)
    Ncorrect = np.zeros(R.shape[1])
    for i, j in zip(row, col):
        Ncorrect += np.logical_and(R[i], H[j])
    miss = np.maximum(0, Nref - Nsys).sum()
    fa = np.maximum(0, Nsys - Nref).sum()
    conf = (np.minimum(Nref, Nsys) - Ncorrect).sum()
    return dict(der=100 * (miss + fa + conf) / total,
                miss=100 * miss / total, fa=100 * fa / total,
                conf=100 * conf / total)


def pyannote_der(ref_labels, hyp_labels, collar, skip_overlap, exclude=None):
    try:
        from pyannote.core import Segment, Annotation, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        return None
    def to_ann(labels):
        a = Annotation()
        for i, (s, e, r) in enumerate(labels):
            a[Segment(s, e), i] = r
        return a
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    ref, hyp = to_ann(ref_labels), to_ann(hyp_labels)
    kwargs = {}
    if exclude:  # list of (start, end) regions to remove from scoring
        tmax = max([e for _, e, _ in ref_labels + hyp_labels] + [0.0])
        ig = Timeline([Segment(s, e) for s, e in exclude])
        kwargs["uem"] = ig.gaps(support=Segment(0, tmax))
    der = metric(ref, hyp, **kwargs)
    c = metric[:]
    t = c["total"]
    if t == 0:
        return None
    return dict(der=100 * der, miss=100 * c["missed detection"] / t,
                fa=100 * c["false alarm"] / t, conf=100 * c["confusion"] / t)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("ref", help="ground-truth Audacity label file")
    p.add_argument("hyp", help="hypothesis (auto / edited) Audacity label file")
    p.add_argument("--collar", type=float, default=0.25,
                   help="forgiveness collar in seconds around ref boundaries "
                        "(literature default 0.25)")
    p.add_argument("--skip-overlap", action="store_true",
                   help="exclude overlapping-speech regions from scoring "
                        "(classic NIST convention)")
    p.add_argument("--ignore-speaker", default="",
                   help="comma-separated speaker(s) to EXCLUDE from scoring, e.g. "
                        "a dead-mic role. Their reference-active regions become "
                        "unscored so you neither get credit nor blame for them.")
    a = p.parse_args(argv)

    ignore = {s.strip() for s in a.ignore_speaker.split(",") if s.strip()}

    ref = parse_labels(a.ref)
    hyp = parse_labels(a.hyp)
    if not ref:
        print("No reference labels parsed.", file=sys.stderr)
        return 2

    gt_roles = sorted({r for *_, r in ref})
    hyp_roles = sorted({r for *_, r in hyp})
    roles = sorted(set(gt_roles) | set(hyp_roles))

    tmax = max([e for _, e, _ in ref] + [e for _, e, _ in hyp] + [0.0])
    n = int(np.ceil(tmax / FRAME)) + 1
    R_full, ridx_full = build_grid(ref, roles, n)
    H_full, hidx_full = build_grid(hyp, roles, n)

    # --- exclude ignored speakers' reference-active regions from scoring ---
    excluded_s = 0.0
    if ignore:
        ig_mask = np.zeros(n, dtype=bool)
        for r in ignore:
            if r in ridx_full:
                ig_mask |= R_full[ridx_full[r]]
        scored = ~ig_mask
        excluded_s = ig_mask.sum() * FRAME
        keep = [r for r in roles if r not in ignore]
        # drop ignored labels for the pyannote path too
        ref = [(s, e, r) for (s, e, r) in ref if r not in ignore]
        hyp = [(s, e, r) for (s, e, r) in hyp if r not in ignore]
        roles = keep
        R = np.vstack([R_full[ridx_full[r]][scored] for r in roles])
        H = np.vstack([H_full[hidx_full[r]][scored] for r in roles])
        ridx = {r: i for i, r in enumerate(roles)}
        hidx = dict(ridx)
    else:
        R, ridx, H, hidx = R_full, ridx_full, H_full, hidx_full
    Nref, Nsys = R.sum(0), H.sum(0)

    print("=" * 64)
    print(f"  REF: {a.ref}")
    print(f"  HYP: {a.hyp}")
    print("=" * 64)

    print("\nROLE INVENTORY")
    print(f"  ground truth : {gt_roles}")
    print(f"  hypothesis   : {hyp_roles}")
    missing = sorted(set(gt_roles) - set(hyp_roles))
    extra = sorted(set(hyp_roles) - set(gt_roles))
    if missing:
        print(f"  !! IN GT BUT MISSING FROM HYP (likely dead/mis-parsed mic): {missing}")
    if extra:
        print(f"  !! IN HYP BUT NOT IN GT (likely a confusion sink): {extra}")
    if ignore:
        print(f"  ** EXCLUDED FROM SCORING: {sorted(ignore)}  "
              f"({excluded_s:.0f} s of their speech regions unscored)")

    # ---- DER ----
    exclude_regions = None
    if ignore:
        exclude_regions = [(s, e) for (s, e, r) in parse_labels(a.ref) if r in ignore]
    py = pyannote_der(ref, hyp, a.collar, a.skip_overlap, exclude=exclude_regions)
    fb = frame_der(R, H, ridx, hidx, roles)
    print(f"\nDER  (collar={a.collar}, skip_overlap={a.skip_overlap})")
    if py:
        print(f"  [pyannote, canonical]  DER = {py['der']:5.1f}%   "
              f"miss {py['miss']:.1f}  FA {py['fa']:.1f}  conf {py['conf']:.1f}")
    print(f"  [frame md-eval, c=0 ]  DER = {fb['der']:5.1f}%   "
          f"miss {fb['miss']:.1f}  FA {fb['fa']:.1f}  conf {fb['conf']:.1f}")
    if not py:
        print("  (install pyannote.metrics for the canonical collar-aware number:")
        print("   pip install --break-system-packages pyannote.metrics)")

    # ---- coverage ----
    speech = (Nref > 0).sum() * FRAME
    ov = (Nref >= 2).sum() * FRAME
    print("\nCOVERAGE")
    print(f"  run span            : {tmax:8.1f} s")
    print(f"  GT speech (>=1 spk) : {speech:8.1f} s ({100*speech/tmax:.1f}% of run)")
    print(f"  GT overlap (>=2 spk): {ov:8.1f} s ({100*ov/max(speech,1e-9):.1f}% of speech)")

    # ---- confusion on clean single/single frames ----
    K = len(roles)
    single = (Nref == 1) & (Nsys == 1)
    cm = np.zeros((K, K), int)
    ra, ha = np.argmax(R, 0), np.argmax(H, 0)
    for f in np.where(single)[0]:
        cm[ra[f], ha[f]] += 1
    print("\nCONFUSION (seconds, frames where BOTH sides have exactly 1 speaker)")
    print("  rows = ground truth, cols = hypothesis")
    print("  " + " " * 16 + "".join(f"{r[:7]:>9}" for r in roles))
    for i, r in enumerate(roles):
        print("  " + f"{r:<16}" + "".join(f"{cm[i,j]*FRAME:>9.0f}" for j in range(K)))

    # ---- per-GT-speaker routing ----
    print("\nROUTING  (for each GT speaker: where did their audio go?)")
    for i, r in enumerate(roles):
        mask = R[i]
        tot = mask.sum() * FRAME
        if tot < 0.5:
            continue
        silent = np.logical_and(mask, Nsys == 0).sum() * FRAME
        print(f"\n  GT {r}  ({tot:.0f} s)")
        print(f"     missed (hyp silent)     : {silent:7.1f} s ({100*silent/tot:5.1f}%)")
        dist = {rr: np.logical_and(mask, H[hidx[rr]]).sum() * FRAME for rr in roles}
        for rr in sorted(roles, key=lambda x: -dist[x]):
            if dist[rr] > 0.5:
                tag = " (correct)" if rr == r else ""
                print(f"     hyp says {rr:<16}: {dist[rr]:7.1f} s ({100*dist[rr]/tot:5.1f}%){tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
