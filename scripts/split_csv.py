import argparse
import csv
import random


def main():
    parser = argparse.ArgumentParser(description="Split a CSV (audio_path,label) into train/val")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("audio_path") or row.get("path") or row.get("file")) is None:
                continue
            if row.get("label") is None:
                continue
            rows.append(row)

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    fieldnames = ["audio_path", "label"]
    with open(args.train_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in train_rows:
            p = r.get("audio_path") or r.get("path") or r.get("file")
            w.writerow({"audio_path": p, "label": r["label"]})

    with open(args.val_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in val_rows:
            p = r.get("audio_path") or r.get("path") or r.get("file")
            w.writerow({"audio_path": p, "label": r["label"]})

    print({"status": "ok", "train": len(train_rows), "val": len(val_rows)})


if __name__ == "__main__":
    main()


