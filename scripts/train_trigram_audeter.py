import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


def load_waveform(path: str, sample_rate: int = 16000) -> torch.Tensor:
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    # ensure shape (L,)
    if waveform.ndim == 2:
        if waveform.shape[0] > 1:
            waveform = waveform[:1]
        waveform = waveform[0]
    return waveform


class CsvAudioDataset(Dataset):
    def __init__(self, csv_path: str, sample_rate: int = 16000):
        import csv

        self.sample_rate = sample_rate
        self.items = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            # expect columns: audio_path,label
            for row in reader:
                p = row.get("audio_path") or row.get("path") or row.get("file")
                l = row.get("label")
                if p is None or l is None:
                    continue
                self.items.append((p, int(l)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        x = load_waveform(path, self.sample_rate)
        return {"audio": x, "label": torch.tensor(label, dtype=torch.long)}


def collate_batch(batch):
    # pad to max length in the batch
    lengths = [item["audio"].shape[0] for item in batch]
    max_len = max(lengths)
    padded = []
    labels = []
    for item in batch:
        x = item["audio"]
        if x.shape[0] < max_len:
            x = torch.nn.functional.pad(x, (0, max_len - x.shape[0]))
        padded.append(x.unsqueeze(0))  # (1, L)
        labels.append(item["label"])
    audio = torch.stack(padded, dim=0).squeeze(1)  # (B, L)
    labels = torch.stack(labels, dim=0)
    return {"audio": audio, "label": labels}


class TrigramLit(pl.LightningModule):
    def __init__(self, use_gat: int = 1, n_edges: int = 10, ngram: int = 3, min_segments: int = 3, lr: float = 1e-4):
        super().__init__()
        from phoneme_GAT.modules import Phoneme_GAT

        self.model = Phoneme_GAT(
            backbone='wavlm',
            use_raw=1,
            use_GAT=use_gat,
            n_edges=n_edges,
            ngram=ngram,
            min_segments=min_segments,
        )
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, audio: torch.Tensor):
        B, L = audio.shape
        num_frames = torch.full((B,), L // 320 - 1, device=audio.device)
        out = self.model(audio, num_frames, profiler=None, use_aug=False, stage='train')
        return out

    def training_step(self, batch, batch_idx):
        audio, label = batch["audio"], batch["label"].to(self.device)
        out = self.forward(audio)
        logit = out["logit"].to(self.device)
        loss = self.loss(logit, label.float())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, label = batch["audio"], batch["label"].to(self.device)
        out = self.forward(audio)
        logit = out["logit"].to(self.device)
        loss = self.loss(logit, label.float())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # AUC/AP hooks can be added with sklearn if desired
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


def main():
    parser = argparse.ArgumentParser(description="Train trigram-only deepfake detector on AUDETER CSV")
    parser.add_argument("--train_csv", type=str, required=True, help="CSV with columns audio_path,label")
    parser.add_argument("--val_csv", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--min_segments", type=int, default=3)
    parser.add_argument("--use_gat", type=int, default=1)
    parser.add_argument("--n_edges", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_ds = CsvAudioDataset(args.train_csv)
    val_ds = CsvAudioDataset(args.val_csv) if args.val_csv else None

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch) if val_ds else None

    lit = TrigramLit(use_gat=args.use_gat, n_edges=args.n_edges, ngram=args.ngram, min_segments=args.min_segments, lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )
    trainer.fit(lit, train_dl, val_dl)


if __name__ == "__main__":
    main()


