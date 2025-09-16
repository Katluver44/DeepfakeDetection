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


def test_reduce_feat_trigram_shapes():
    pg_modules = _import_modules_py()

    B, T, C = 2, 10, 8
    hidden_states = torch.randn(B, T, C)
    # sample 0: ids -> 5 segments => 3 trigrams
    # sample 1: ids -> 5 segments => 3 trigrams
    phoneme_ids = torch.tensor([
        [1, 1, 2, 2, 3, 4, 4, 5, 5, 5],
        [7, 7, 7, 8, 9, 9, 10, 10, 11, 11],
    ], dtype=torch.int64)
    num_frames = torch.tensor([T, T], dtype=torch.int64)

    reduced_hidden_states, reduced_num_frames, reduced_ids = pg_modules.reduce_feat(
        hidden_states, num_frames, phoneme_ids, ngram=3
    )

    assert reduced_num_frames.tolist() == [3, 3]
    assert reduced_hidden_states.shape[0] == 6
    # ids are int64 hashed n-grams
    assert reduced_ids.dtype == torch.int64


def test_reduce_feat_fallback_short_sequences():
    pg_modules = _import_modules_py()

    B, T, C = 1, 4, 8
    hidden_states = torch.randn(B, T, C)
    # only 2 segments -> S < 3 so fallback to unigram segments
    phoneme_ids = torch.tensor([[1, 1, 2, 2]], dtype=torch.int64)
    num_frames = torch.tensor([T], dtype=torch.int64)

    reduced_hidden_states, reduced_num_frames, reduced_ids = pg_modules.reduce_feat(
        hidden_states, num_frames, phoneme_ids, ngram=3
    )

    assert reduced_num_frames.tolist() == [2]
    assert reduced_hidden_states.shape[0] == 2

