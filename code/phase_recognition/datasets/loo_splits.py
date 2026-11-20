"""
Shared Leave-One-Out (LOO) cross-validation split logic.

Backward-compatible defaults reproduce the original 9-video LOO
(test=1, val=2, train=6) used by the Phase-1/2 baseline.

The Experiment A/B setup adds two parameters:

* `pool` — restricts the LOO rotation to a subset of video indices
  (e.g. `[0..7]` for Runs 1-8 only, leaving Run 9 = video_8 as
  fixed out-of-domain test).
* `val_size` — number of cyclically-following videos used for
  validation (default 2 = legacy behaviour; set to 1 for the
  Experiment A/B 6-train / 1-val / 1-test split).

The held-out OOD video (e.g. Run 9) is *not* part of `pool` and is
therefore guaranteed never to appear in train, val, or in-domain
test — Experiment B's zero-shot evaluation then uses the resulting
models directly.
"""


def loo_split(all_vids, test_fold, pool=None, val_size=2):
    """
    Parameters
    ----------
    all_vids : iterable[int]
        Every video index present in the dataset (used for sanity check
        and to derive `pool` if it isn't given).
    test_fold : int
        The video index to hold out for in-domain testing.
    pool : iterable[int] | None
        The set of video indices the LOO rotation operates over. If
        `None`, defaults to `all_vids` (legacy behaviour). Must contain
        `test_fold`. Videos in `all_vids` that are *not* in `pool` are
        reserved (e.g. for OOD test) and excluded from train/val.
    val_size : int
        How many videos cyclically following `test_fold` (inside the
        pool) are used as validation. Must be < len(pool).

    Returns
    -------
    (train_vids, val_vids, test_vids) : tuple[list[int], list[int], list[int]]
    """
    all_vids = sorted(int(v) for v in all_vids)
    if pool is None:
        pool = list(all_vids)
    pool = sorted(int(v) for v in pool)

    if test_fold not in pool:
        raise ValueError(
            f"test_fold {test_fold} not in pool {pool}. "
            f"(All available videos: {all_vids}.)"
        )
    if val_size < 0 or val_size >= len(pool):
        raise ValueError(
            f"val_size={val_size} invalid for pool of size {len(pool)}"
        )

    n = len(pool)
    ti = pool.index(test_fold)
    val = [pool[(ti + 1 + i) % n] for i in range(val_size)]
    test = [test_fold]
    train = [v for v in pool if v not in val + test]
    return train, val, test


def ood_videos(all_vids, pool):
    """Return the videos in `all_vids` that are excluded from `pool`
    — i.e. reserved as out-of-domain (e.g. Run 9 = video_8)."""
    all_vids = sorted(int(v) for v in all_vids)
    pool = set(int(v) for v in pool)
    return [v for v in all_vids if v not in pool]
