"""Bifocal-OR: score the oracle-speaker-count pyannote runs exactly like Table V.

Same convention as the paper:
  - 250 ms collar (pyannote convention: half before, half after each reference boundary)
  - optimal one-to-one speaker mapping, overlap-aware, 10 ms grid
  - Run 3 without the Anesthesiologist (dead wearable channel)

It scores both the original pyannote output (sanity check: must give Table V,
54.7 / 76.9 / 77.2 % DER) and the new oracle-count output, then prints the
means and a sentence for the paper. Reads only; writes oracle_count_scores.csv.

Usage (Windows PowerShell):
    python score_oracle.py --sync "E:\\Ophthalmo_4D\\Ophthalmo_4D_DVR_Sync" --new "$HOME\\Desktop\\bifocal\\diar"
Needs numpy and scipy.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

TABLE_V = {3: (4.3, 40.4, 10.0, 54.7), 4: (3.6, 59.6, 13.7, 76.9), 7: (3.0, 65.1, 9.2, 77.2)}


def parse(path):
    out = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            try:
                out.append((float(p[0]), float(p[1]), p[2].strip()))
            except ValueError:
                pass
    return out


def der_collar(ref, hyp, collar=0.25, ignore=(), frame=0.01):
    tmax = max(e for _, e, _ in ref + hyp)
    n = int(math.ceil(tmax / frame)) + 2
    uem = np.ones(n, bool)
    for s, e, r in ref:
        if r in ignore:
            uem[int(round(s / frame)):int(round(e / frame))] = False
    ref = [x for x in ref if x[2] not in ignore]
    hyp = [x for x in hyp if x[2] not in ignore]
    for s, e, _ in ref:
        for b in (s, e):
            uem[max(0, int(round((b - collar / 2) / frame))):int(round((b + collar / 2) / frame))] = False

    def grid(labels):
        roles = sorted({r for *_, r in labels})
        m = np.zeros((len(roles), n), bool)
        for s, e, r in labels:
            m[roles.index(r), int(round(s / frame)):int(round(e / frame))] = True
        return m[:, uem]

    R, H = grid(ref), grid(hyp)
    co = R.astype(np.int64) @ H.T.astype(np.int64)
    ri, hi = linear_sum_assignment(-co)
    nref, nhyp = R.sum(0), H.sum(0)
    corr = sum((R[a] & H[b]).astype(np.int64) for a, b in zip(ri, hi))
    total = nref.sum()
    fa = np.maximum(0, nhyp - nref).sum()
    miss = np.maximum(0, nref - nhyp).sum()
    conf = (np.minimum(nref, nhyp) - corr).sum()
    return tuple(float(round(100 * v / total, 1)) for v in (fa, miss, conf, fa + miss + conf))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync", required=True, help="Ophthalmo_4D_DVR_Sync folder (references + original pyannote files)")
    ap.add_argument("--new", required=True, help="folder with Run_N_pyannote_n5.txt")
    ap.add_argument("--suffix", default="_pyannote_n5.txt")
    a = ap.parse_args()

    rows, ok = [], True
    print(f"{'Run':>4} {'version':<16} {'FA':>5} {'Miss':>5} {'Conf':>5} {'DER':>6}  clusters")
    for r in (3, 4, 7):
        d = os.path.join(a.sync, f"Run_{r}")
        ref_p = os.path.join(d, f"Run_{r}_Labels_GroundTruth.txt")
        files = {"original": os.path.join(d, f"Run_{r}_Ambient_pyannote.txt"),
                 "oracle count": os.path.join(a.new, f"Run_{r}{a.suffix}")}
        if not os.path.exists(ref_p):
            sys.exit(f"Missing reference: {ref_p}")
        ref = parse(ref_p)
        ign = ("Anesthesiologist",) if r == 3 else ()
        for name, p in files.items():
            if not os.path.exists(p):
                print(f"{r:>4} {name:<16} missing: {p}")
                ok = False
                continue
            hyp = parse(p)
            fa, miss, conf, der = der_collar(ref, hyp, 0.25, ign)
            k = len({s for *_, s in hyp})
            flag = ""
            if name == "original":
                same = all(abs(x - y) <= 0.11 for x, y in zip((fa, miss, conf, der), TABLE_V[r]))
                flag = "  = Table V" if same else f"  != Table V {TABLE_V[r]}"
            print(f"{r:>4} {name:<16} {fa:5.1f} {miss:5.1f} {conf:5.1f} {der:6.1f}  {k}{flag}")
            rows.append(dict(run=r, version=name, fa=fa, miss=miss, conf=conf, der=der, clusters=k))
    if not ok:
        sys.exit("\nSome files are missing; nothing summarized.")

    print()
    for name in ("original", "oracle count"):
        v = np.array([[x["fa"], x["miss"], x["conf"], x["der"]] for x in rows if x["version"] == name])
        print(f"mean {name:<13} FA {v[:,0].mean():.1f}  Miss {v[:,1].mean():.1f}  Conf {v[:,2].mean():.1f}  "
              f"DER {v[:,3].mean():.1f} ± {v[:,3].std(ddof=1):.1f}")
    with open("oracle_count_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nWritten oracle_count_scores.csv - send me this file or the lines above.")


if __name__ == "__main__":
    main()
