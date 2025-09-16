import argparse
import os
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare small AUDETER CSVs from Hugging Face (streaming)")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to store downloaded wavs and CSVs")
    parser.add_argument(
        "--subsets",
        type=str,
        default="mls-tts-bark,mls-vocoders-hifigan",
        help="Comma-separated AUDETER subsets to sample from (see HF card)",
    )
    parser.add_argument("--per_subset", type=int, default=200, help="Number of items per subset")
    parser.add_argument(
        "--bona_subsets",
        type=str,
        default="",
        help="Comma-separated subsets to label as bona-fide (0). Others labeled fake (1)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    import datasets
    import fsspec
    import tarfile

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    bona = set([s.strip() for s in args.bona_subsets.split(",") if s.strip()])
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    rows = []
    for subset in subsets:
        print({"info": f"streaming subset {subset}"})
        ds = datasets.load_dataset("wqz995/AUDETER", subset, split="dev", streaming=True)
        # avoid audio decoding (torchcodec); we'll use __url__/__key__ to stream tars
        try:
            from datasets.features import Audio
            ds = ds.cast_column("wav", Audio(decode=False))
        except Exception:
            pass
        count = 0
        for ex in ds:
            # Avoid Audio feature decoding; extract directly from tar with __url__/__key__
            url = ex.get("__url__")
            key = ex.get("__key__")  # e.g., dev/1055_0
            if not url or not key:
                continue
            wav_member_suffix = f"{key}.wav"
            rel = f"{subset.replace('/', '_')}_{count}.wav"
            out_path = wav_dir / rel
            try:
                with fsspec.open(url, mode="rb") as fobj:
                    # stream tar and find the wav member
                    with tarfile.open(fileobj=fobj, mode="r|*") as tf:
                        found = False
                        for member in tf:
                            if member.name.endswith(wav_member_suffix):
                                with tf.extractfile(member) as mf, open(out_path, "wb") as wf:
                                    wf.write(mf.read())
                                found = True
                                break
                        if not found:
                            continue
            except Exception:
                continue
            label = 0 if subset in bona else 1
            rows.append({"audio_path": str(out_path), "label": label})
            count += 1
            if count >= args.per_subset:
                break

    # split
    n_total = len(rows)
    n_val = max(1, int(n_total * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    with open(out_dir / "audeter_train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "label"])
        w.writeheader()
        w.writerows(train_rows)
    with open(out_dir / "audeter_val.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "label"])
        w.writeheader()
        w.writerows(val_rows)

    print({"status": "ok", "train": len(train_rows), "val": len(val_rows), "csv_dir": str(out_dir)})


if __name__ == "__main__":
    main()


