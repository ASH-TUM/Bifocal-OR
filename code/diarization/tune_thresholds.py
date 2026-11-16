#!/usr/bin/env python3
"""
tune_thresholds.py — Automatic threshold sweep for the Micro-OR diarizer.

For each (vad-threshold, min-contrast-db) combination it:
  1. re-runs auto_diarize_v2.py on ONE run with those thresholds,
  2. scores the result against your ground truth with score_diarization.py,
  3. records the DER.
Finally it prints a table ranked by DER so you can read off the best config.

This automates the boring part of tuning — you set it going and walk away.

NOTE: every iteration overwrites the run's Run_X_AntiFragLabels.txt. That's
fine during the sweep, but once you've chosen the best config, do ONE final
plain run of auto_diarize_v2.py with those thresholds to leave the good
labels in place.

Example (Windows, all on one line):
  python "G:\\...\\code\\tune_thresholds.py"
      --script  "G:\\...\\code\\auto_diarize_v2.py"
      --scorer  "G:\\...\\code\\score_diarization.py"
      --base-dir "G:\\Ophthalmo_4D_DVR_Sync"
      --run Run_3
      --gt  "G:\\Ophthalmo_4D_DVR_Sync\\Run_3\\Run_3_Labels_GroundTruth.txt"
      --hyp "G:\\Ophthalmo_4D_DVR_Sync\\Run_3\\Run_3_AntiFragLabels.txt"
      --ignore-speaker Anesthesiologist
"""
from __future__ import annotations
import argparse
import itertools
import re
import subprocess
import sys

DER_RE = re.compile(
    r"\[frame md-eval[^\]]*\]\s*DER\s*=\s*([\d.]+)%\s+miss\s+([\d.]+)"
    r"\s+FA\s+([\d.]+)\s+conf\s+([\d.]+)"
)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--python", default=sys.executable,
                   help="python executable (default: the one running this)")
    p.add_argument("--script", required=True, help="path to auto_diarize_v2.py")
    p.add_argument("--scorer", required=True, help="path to score_diarization.py")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--run", required=True, help="run folder name, e.g. Run_3")
    p.add_argument("--gt", required=True, help="ground-truth label file")
    p.add_argument("--hyp", required=True,
                   help="label file the script writes (…_AntiFragLabels.txt)")
    p.add_argument("--ignore-speaker", default="",
                   help="dead-mic speaker(s) to exclude from scoring")
    p.add_argument("--vad", default="0.20,0.25,0.30,0.35",
                   help="comma-separated VAD thresholds to try")
    p.add_argument("--contrast", default="2.0,3.0,4.0",
                   help="comma-separated min-contrast-db values to try")
    p.add_argument("--gain-clip", default="1.0,1.15,1.25",
                   help="comma-separated gain-clip values to try (1.0 = raw energy)")
    a = p.parse_args()

    vads = [float(x) for x in a.vad.split(",")]
    cons = [float(x) for x in a.contrast.split(",")]
    clips = [float(x) for x in a.gain_clip.split(",")]
    combos = list(itertools.product(vads, cons, clips))
    print(f"Sweeping {len(combos)} configs on {a.run}. "
          f"This re-runs the diarizer each time, so grab a coffee.\n")

    results = []
    for i, (v, c, g) in enumerate(combos, 1):
        gen = [a.python, a.script, "--base-dir", a.base_dir, "--run", a.run,
               "--vad-threshold", str(v), "--min-contrast-db", str(c),
               "--gain-clip", str(g)]
        r1 = run(gen)
        if r1.returncode != 0:
            print(f"[{i}/{len(combos)}] vad={v} contrast={c} gain-clip={g}  GENERATION FAILED")
            print("   " + (r1.stderr or r1.stdout)[-400:].replace("\n", "\n   "))
            continue
        score = [a.python, a.scorer, a.gt, a.hyp]
        if a.ignore_speaker:
            score += ["--ignore-speaker", a.ignore_speaker]
        r2 = run(score)
        m = DER_RE.search(r2.stdout)
        if not m:
            print(f"[{i}/{len(combos)}] vad={v} contrast={c} gain-clip={g}  COULD NOT PARSE DER")
            print("   " + (r2.stderr or r2.stdout)[-400:].replace("\n", "\n   "))
            continue
        der, miss, fa, conf = map(float, m.groups())
        results.append((der, miss, fa, conf, v, c, g))
        print(f"[{i}/{len(combos)}] vad={v:<5} contrast={c:<5} gain-clip={g:<5} "
              f"DER={der:5.1f}%  (miss {miss:4.1f}  FA {fa:4.1f}  conf {conf:4.1f})")

    if not results:
        print("\nNo successful runs — check the --script / --scorer / paths.")
        return 1

    results.sort()
    print("\n" + "=" * 60)
    print("RANKED BY DER (lowest = best)")
    print("=" * 60)
    print(f"{'DER':>6}{'miss':>7}{'FA':>6}{'conf':>7}   vad   contrast  gain-clip")
    for der, miss, fa, conf, v, c, g in results[:10]:
        print(f"{der:>6.1f}{miss:>7.1f}{fa:>6.1f}{conf:>7.1f}   {v:<6}{c:<10}{g}")
    best = results[0]
    print(f"\nBEST: --vad-threshold {best[4]} --min-contrast-db {best[5]} "
          f"--gain-clip {best[6]}   (DER {best[0]:.1f}%)")
    print("Now do ONE final run of auto_diarize_v2.py with those three values\n"
          "to leave the best labels in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
