"""Bifocal-OR: collect the small files that describe timing, sync and calibration (read-only).

Copies about 20-120 MB of small files from the clean dataset into one zip, keeping
their folder structure, so that the timing of all streams can be worked out:

  - METADATA.csv (dataset root)
  - per recording folder (rec_*): capture01/pcpd_config.yml and fsmap.yml,
    export_full/calibration/*.json, export_full/tables_timestamps.arrow (frame timestamps)
  - per microscope recording (*Manual*, basler_grata_and_microphone/output/*):
    sync_log.txt and ffmpeg_log.txt
  - per run in Ophthalmo_4D_DVR_Sync: *_pyannote.txt, *_diarize_summary.json,
    *_Labels_GroundTruth.txt, *_AntiFragLabels.txt
  - from the audio-sync working folder next to it (E:\\Ophthalmo_4D__Audio_Synced, found
    automatically, or --audio-synced DIR): README.txt and, per run, notes.txt, the Reaper
    project (*.rpp) and the PreDTW/PostDTW timelines (*.xml)

It never opens node_hostvars.yml (it can contain credentials) and never reads
the frame folders, videos or WAV files. If pyarrow is installed, the .arrow tables are
written as compact .csv.gz instead of the raw files (please install it, see below).

Usage (Windows PowerShell):
    python collect_sync_files.py "E:\\Ophthalmo_4D"
    python collect_sync_files.py "E:\\Ophthalmo_4D" --out sync_files.zip
    python collect_sync_files.py "E:\\Ophthalmo_4D" --audio-synced "E:\\Ophthalmo_4D__Audio_Synced"

Needs only the Python standard library, plus pyarrow (pip install pyarrow) to convert the
.arrow tables; without pyarrow they are copied raw, which I can't read on my side.
"""
import argparse
import fnmatch
import gzip
import io
import os
import sys
import zipfile

try:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
except ImportError:  # optional
    feather = None

CSV_SAFE = ("is_integer", "is_floating", "is_boolean", "is_string", "is_large_string",
            "is_timestamp", "is_date", "is_time", "is_null")

SKIP_ALWAYS = {"node_hostvars.yml"}


def add(zf, root, rel, log, prefix=""):
    """Add root/rel to the zip as prefix/rel. prefix keeps files from a second folder apart."""
    full = os.path.join(root, rel)
    if os.path.basename(rel) in SKIP_ALWAYS or not os.path.isfile(full):
        return 0
    size = os.path.getsize(full)
    arc = (prefix + "/" if prefix else "") + rel.replace("\\", "/")
    if feather is not None and rel.endswith(".arrow"):
        try:  # a compact CSV (+ a description of the table) instead of the raw file
            table = read_arrow(full)
            zf.writestr(arc + ".schema.txt", describe(table))
            zf.writestr(arc + ".csv.gz", gzip.compress(to_csv(table)))
            log.append((arc + ".csv.gz", size))
            return size
        except Exception as exc:  # noqa: BLE001 - fall back to the raw file
            print(f"  could not convert {rel} ({str(exc)[:100]}); copied as it is")
    zf.write(full, arcname=arc)
    log.append((arc, size))
    return size


def read_arrow(path):
    """An Arrow table from a Feather/IPC file or an IPC stream."""
    try:
        return feather.read_table(path)
    except Exception:  # noqa: BLE001
        pass
    try:
        with ipc.open_file(path) as r:
            return r.read_all()
    except Exception:  # noqa: BLE001
        with open(path, "rb") as fh:
            return ipc.open_stream(fh).read_all()


def describe(table):
    """Schema, metadata and size of a table, as text."""
    lines = [f"rows: {table.num_rows}", "schema:", str(table.schema)]
    meta = table.schema.metadata or {}
    if meta:
        lines.append("metadata:")
        for k, v in meta.items():
            lines.append(f"  {k.decode('utf-8', 'replace')}: {v.decode('utf-8', 'replace')[:2000]}")
    return "\n".join(lines) + "\n"


def to_csv(table):
    """CSV bytes; nested or unusual columns are flattened or written as text."""
    for _ in range(3):
        if any(pa.types.is_struct(f.type) for f in table.schema):
            table = table.flatten()
    cols = []
    for col in table.columns:
        t = col.type
        if pa.types.is_dictionary(t):
            col, t = col.cast(t.value_type), t.value_type
        if pa.types.is_duration(t):
            col, t = col.cast(pa.int64()), pa.int64()
        if not any(getattr(pa.types, f)(t) for f in CSV_SAFE):
            col = pa.array(["" if x is None else str(x) for x in col.to_pylist()], pa.string())
        cols.append(col)
    buf = io.BytesIO()
    pacsv.write_csv(pa.table(cols, names=table.column_names), buf)
    return buf.getvalue()


def listdir(path):
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="clean dataset folder, e.g. E:\\Ophthalmo_4D")
    ap.add_argument("--out", default="sync_files.zip")
    ap.add_argument("--audio-synced", help="audio-sync working folder (default: Ophthalmo_4D__Audio_Synced "
                                           "next to the dataset folder, if it exists)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"Not a folder: {root}")
    synced = args.audio_synced or os.path.join(os.path.dirname(root), "Ophthalmo_4D__Audio_Synced")
    log = []
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        add(zf, root, "METADATA.csv", log)
        for d in listdir(root):
            if d.startswith("rec_") and os.path.isdir(os.path.join(root, d)):
                for f in ("pcpd_config.yml", "fsmap.yml"):
                    add(zf, root, os.path.join(d, "capture01", f), log)
                ef = os.path.join(d, "export_full")
                for f in listdir(os.path.join(root, ef)):
                    # frame timestamps of every run; the (larger) file table only as a sample
                    if f.endswith(".arrow") and ("timestamps" in f or "bodytracking" in f or "Charuco" in d):
                        add(zf, root, os.path.join(ef, f), log)
                for f in listdir(os.path.join(root, ef, "calibration")):
                    add(zf, root, os.path.join(ef, "calibration", f), log)
                for sub in listdir(os.path.join(root, d)):
                    if "manual" in sub.lower() or "robotic" in sub.lower():
                        for f in ("sync_log.txt", "ffmpeg_log.txt"):
                            add(zf, root, os.path.join(d, sub, f), log)
        out_dir = os.path.join("basler_grata_and_microphone", "output")
        for sub in listdir(os.path.join(root, out_dir)):
            for f in ("sync_log.txt", "ffmpeg_log.txt"):
                add(zf, root, os.path.join(out_dir, sub, f), log)
        dvr = "Ophthalmo_4D_DVR_Sync"
        for run in listdir(os.path.join(root, dvr)):
            for f in listdir(os.path.join(root, dvr, run)):
                if any(fnmatch.fnmatch(f, p) for p in ("*_pyannote.txt", "*_diarize_summary.json",
                                                         "*_Labels_GroundTruth.txt", "*_AntiFragLabels.txt")):
                    add(zf, root, os.path.join(dvr, run, f), log)
        if os.path.isdir(synced):
            name = os.path.basename(os.path.normpath(synced))
            add(zf, synced, "README.txt", log, name)
            for run in listdir(synced):
                for f in listdir(os.path.join(synced, run)):
                    if f.lower() == "notes.txt" or f.lower().endswith((".rpp", ".xml")):
                        add(zf, synced, os.path.join(run, f), log, name)
        else:
            print(f"Audio-sync working folder not found ({synced}); skipped. Use --audio-synced DIR.")
    for rel, size in log:
        print(f"  {rel}  ({size:,})")
    total = sum(s for _, s in log)
    print(f"\n{len(log)} entries, {total / 1e6:.1f} MB read. Zip: {os.path.abspath(args.out)} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    if feather is None:
        print("NOTE: pyarrow is not installed, so the .arrow tables were copied as they are, and I can't\n"
              "read them on my side. Please run:  pip install pyarrow  and then this script again.")


if __name__ == "__main__":
    main()
