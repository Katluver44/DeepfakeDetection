import argparse
import os

import torch


def load_waveform(path: str, sample_rate: int = 16000) -> torch.Tensor:
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        if waveform.ndim == 2:
            # mono
            if waveform.shape[0] > 1:
                waveform = waveform[:1]
            waveform = waveform[0]
        return waveform
    except Exception:
        # fallback: generate 3s of random noise
        return torch.randn(sample_rate * 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None, help="Path to a wav file (optional)")
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--min_segments", type=int, default=3)
    parser.add_argument("--use_gat", type=int, default=1)
    args = parser.parse_args()

    from phoneme_GAT.modules import Phoneme_GAT

    model = Phoneme_GAT(
        backbone='wavlm',
        use_raw=1,
        use_GAT=int(args.use_gat),
        n_edges=10,
        ngram=int(args.ngram),
        min_segments=int(args.min_segments),
    )
    model.eval()

    if args.wav is not None and os.path.exists(args.wav):
        x = load_waveform(args.wav)
    else:
        x = load_waveform(None)

    # batch size 1
    x = x.unsqueeze(0)
    num_frames = torch.full((x.shape[0],), x.shape[1] // 320 - 1)

    with torch.no_grad():
        out = model(x, num_frames, profiler=None, use_aug=False, stage='val')

    print({
        'status': 'ok',
        'ngram': model.ngram,
        'min_segments': model.min_segments,
        'use_GAT': model.use_GAT,
        'logit': out['logit'].tolist(),
        'kept': None if out.get('keep_indices', None) is None else out['keep_indices'].tolist()
    })


if __name__ == "__main__":
    main()


