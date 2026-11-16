"""Bifocal-OR: upload the release to Hugging Face straight from the clean dataset.

Nothing is copied or written on the data drive. Frame folders are packed into one
.tar per run, camera and modality in a temporary folder (default: your temp folder on
C:), uploaded, and deleted again, one at a time. Files that are already in the repository
are skipped, so the script can be stopped (Ctrl+C) and restarted at any time.

Before the first run (once):
    pip install huggingface_hub
    hf auth login            (or: huggingface-cli login) with a token that has write access

Usage (Windows PowerShell):
    python upload_release.py release_manifest.csv --root "E:\\Ophthalmo_4D" --repo USER/Bifocal-OR --dry-run
    python upload_release.py release_manifest.csv --root "E:\\Ophthalmo_4D" --repo USER/Bifocal-OR --metadata .\\Bifocal-OR
    python upload_release.py release_manifest.csv --root "E:\\Ophthalmo_4D" --repo USER/Bifocal-OR --only runs/run03

    --dry-run        check every source and print sizes; uploads nothing
    --metadata DIR   also upload the metadata folder (annotations, calibration, README, ...) first
    --only TEXT      only rows whose target contains TEXT (repeatable)
    --temp DIR       where the .tar files are built (needs ~10 GB free)
    --to-folder DIR  write the release layout into a local folder instead of uploading

It refuses to upload into a public repository that is not gated (the data must stay behind
the access request until publication); --allow-ungated overrides this.

Manifest columns: kind, source, pattern, target, group
    kind=file  one file (source relative to --root)
    kind=tar   all files in the source folder that match the pattern, as one uncompressed .tar
    group      files with the same group are uploaded in one commit
A log (upload_log.csv) records every upload with size, file count and time.
"""
import argparse
import csv
import fnmatch
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from collections import OrderedDict


class FolderTarget:
    """Writes the release layout into a local folder (for testing or for a spare drive)."""

    def __init__(self, folder):
        self.folder = folder

    def existing(self):
        out = {}
        for dp, _, fns in os.walk(self.folder):
            for f in fns:
                full = os.path.join(dp, f)
                out[os.path.relpath(full, self.folder).replace("\\", "/")] = os.path.getsize(full)
        return out

    def put(self, items, message):
        for src, target in items:
            dst = os.path.join(self.folder, *target.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

    def put_folder(self, folder, message):
        shutil.copytree(folder, self.folder, dirs_exist_ok=True)


class HubTarget:
    def __init__(self, repo, allow_ungated=False):
        from huggingface_hub import HfApi, CommitOperationAdd
        self.api, self.Op, self.repo = HfApi(), CommitOperationAdd, repo
        try:
            info = self.api.dataset_info(repo)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Can't open the dataset repository {repo} ({str(exc)[:150]}).\n"
                     "Create it on huggingface.co first (New -> Dataset) and log in with a write token "
                     "(hf auth login).")
        gated, private = getattr(info, "gated", None), getattr(info, "private", None)
        print(f"Repository {repo}: {'private' if private else 'public'}, gated: {gated or 'no'}")
        if not private and not gated and not allow_ungated:
            sys.exit("The repository is public and not gated, so anyone could download the data.\n"
                     "Enable access requests with manual approval (Settings -> Access requests) and run again.")

    def existing(self):
        out = {}
        for f in self.api.list_repo_tree(self.repo, repo_type="dataset", recursive=True):
            if getattr(f, "size", None) is not None:
                out[f.path] = f.size
        return out

    def put(self, items, message):
        ops = [self.Op(path_in_repo=target, path_or_fileobj=src) for src, target in items]
        for attempt in range(4):
            try:
                self.api.create_commit(self.repo, operations=ops, commit_message=message, repo_type="dataset")
                return
            except Exception as exc:  # noqa: BLE001 - retry network hiccups
                if attempt == 3:
                    raise
                wait = 60 * (attempt + 1)
                print(f"    upload failed ({str(exc)[:120]}); retrying in {wait} s")
                time.sleep(wait)

    def put_folder(self, folder, message):
        self.api.upload_folder(repo_id=self.repo, folder_path=folder, repo_type="dataset", commit_message=message)


def matching(folder, pattern):
    with os.scandir(folder) as it:
        files = [(e.name, e.path, e.stat().st_size) for e in it if e.is_file() and fnmatch.fnmatch(e.name, pattern)]
    return sorted(files)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--root", required=True, help="clean dataset folder, e.g. E:\\Ophthalmo_4D")
    ap.add_argument("--repo", help="Hugging Face dataset repository, e.g. USER/Bifocal-OR")
    ap.add_argument("--metadata", help="metadata folder to upload first (README, annotations, ...)")
    ap.add_argument("--only", action="append", default=[], help="only targets containing this text")
    ap.add_argument("--temp", default=tempfile.gettempdir(), help="folder for the temporary .tar files")
    ap.add_argument("--to-folder", help="write into this local folder instead of uploading")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-ungated", action="store_true", help="upload even if the public repo is not gated")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8-sig"))
            if not args.only or any(o in r["target"] for o in args.only)]
    if not args.dry_run and not (args.repo or args.to_folder):
        sys.exit("Give --repo (Hugging Face) or --to-folder (local), or use --dry-run.")
    target = None
    if not args.dry_run:
        target = FolderTarget(args.to_folder) if args.to_folder else HubTarget(args.repo, args.allow_ungated)
    existing = target.existing() if target else {}
    print(f"{len(rows)} manifest rows; {len(existing)} files already at the destination.")

    if args.metadata and target and not args.dry_run:
        print(f"Uploading metadata folder {args.metadata} ...")
        target.put_folder(args.metadata, "Add metadata layer")

    log_path = "upload_log.csv"
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["time", "target", "kind", "bytes", "files", "seconds", "status"])

    groups = OrderedDict()
    total_bytes = n_missing = n_skip = n_up = 0
    try:
        for r in rows:
            kind, tgt = r["kind"].strip(), r["target"].strip()
            src = os.path.join(args.root, *[s for s in re.split(r"[\\/]", r["source"].strip()) if s])
            if kind == "file":
                if not os.path.isfile(src):
                    print(f"  MISSING  {r['source']}")
                    n_missing += 1
                    continue
                size = os.path.getsize(src)
                total_bytes += size
                if tgt in existing and existing[tgt] == size:
                    n_skip += 1
                    continue
                groups.setdefault(r.get("group") or tgt, []).append((src, tgt, size))
            elif kind == "tar":
                if not os.path.isdir(src):
                    print(f"  MISSING  {r['source']}")
                    n_missing += 1
                    continue
                if tgt in existing:
                    n_skip += 1
                    total_bytes += existing[tgt]
                    continue
                files = matching(src, r["pattern"])
                size = sum(s for _, _, s in files)
                total_bytes += size
                if args.dry_run:
                    print(f"  tar  {tgt}: {len(files):,} files, {human(size)}")
                    continue
                need = size + 2048 * len(files) + 100_000_000
                if shutil.disk_usage(args.temp).free < need:
                    sys.exit(f"Not enough space in {args.temp} for {tgt} ({human(need)} needed). Use --temp.")
                t0 = time.time()
                tmp = os.path.join(args.temp, "bifocal_" + tgt.replace("/", "_"))
                with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tf:
                    for name, path, _ in files:
                        tf.add(path, arcname=name)
                try:
                    target.put([(tmp, tgt)], f"Add {tgt}")
                finally:
                    os.remove(tmp)
                n_up += 1
                secs = time.time() - t0
                log.writerow([time.strftime("%Y-%m-%d %H:%M"), tgt, "tar", size, len(files), f"{secs:.0f}", "ok"])
                logf.flush()
                print(f"  done {tgt}: {len(files):,} files, {human(size)}, {secs / 60:.1f} min")
        for grp, items in groups.items():
            size = sum(s for _, _, s in items)
            if args.dry_run:
                print(f"  {len(items)} file(s) in {grp}: {human(size)}")
                continue
            t0 = time.time()
            target.put([(s, t) for s, t, _ in items], f"Add {grp}")
            n_up += len(items)
            secs = time.time() - t0
            for s, t, sz in items:
                log.writerow([time.strftime("%Y-%m-%d %H:%M"), t, "file", sz, 1, f"{secs:.0f}", "ok"])
            logf.flush()
            print(f"  done {grp}: {len(items)} file(s), {human(size)}, {secs / 60:.1f} min")
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it stopped.")
    finally:
        logf.close()
    print(f"\nSelected rows: {len(rows)} ({human(total_bytes)}); uploaded now: {n_up}; "
          f"already at the destination: {n_skip}; missing sources: {n_missing}")
    if args.dry_run:
        print("Dry run only: nothing was uploaded.")


if __name__ == "__main__":
    main()
