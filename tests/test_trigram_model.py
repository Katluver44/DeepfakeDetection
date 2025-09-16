import torch


def _import_modules_py():
    import importlib.util
    import sys
    import types
    from pathlib import Path

    # ensure local package root is importable for `ay2`
    sys.path.insert(0, str(Path(__file__).parents[1]))

    # create a dummy top-level package `phoneme_GAT` so relative imports work
    pkg_name = 'phoneme_GAT'
    pkg_path = Path(__file__).parents[1] / pkg_name
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        sys.modules[pkg_name] = pkg

    mod_name = f'{pkg_name}.modules'
    mod_path = pkg_path / 'modules.py'
    spec = importlib.util.spec_from_file_location(mod_name, str(mod_path))
    pg_modules = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = pg_modules
    spec.loader.exec_module(pg_modules)
    return pg_modules


def test_model_forward_shapes_ngram3(monkeypatch):
    pg_modules = _import_modules_py()

    # Monkeypatch load_phoneme_model to avoid importing heavy deps (e.g., librosa)
    class _Id(torch.nn.Module):
        def forward(self, x):
            return (x,)

    class _StubLM(torch.nn.Module):
        def forward(self, x):
            B, T, C = x.shape
            logits = torch.zeros(B, T, 5)
            for b in range(B):
                for t in range(T):
                    logits[b, t, (t % 5)] = 1.0
            return logits

    class _StubTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Id()
            # used in _mask_hidden_states
            self.masked_spec_embed = torch.zeros(768)

        def feature_extractor(self, x):
            # return (B, C, T)
            B = x.shape[0]
            return torch.randn(B, 768, 10)

        def feature_projection(self, feat1):
            # project to (B, T, C)
            return feat1, None

    class _StubPhonemeModel:
        def __init__(self):
            self.model = type('MM', (), {
                'model': type('Inner', (), {
                    'wavlm': _StubTransformer(),
                    'lm_head': _StubLM(),
                })()
            })()
        def requires_grad_(self, flag: bool):
            return self
        def eval(self):
            return self

    monkeypatch.setattr(pg_modules, 'load_phoneme_model', lambda **kwargs: _StubPhonemeModel(), raising=True)

    # create a small model and stub its encoder to identity
    model = pg_modules.Phoneme_GAT(backbone='wavlm', use_raw=1, use_GAT=0, n_edges=2, ngram=3)
    model.encoder = _Id()

    B = 2
    x = torch.randn(B, 3200)
    num_frames = torch.full((B,), 3200 // 320 - 1)
    out = model(x, num_frames, profiler=None, use_aug=False, stage='val')

    assert 'logit' in out and out['logit'].shape == (B,)

