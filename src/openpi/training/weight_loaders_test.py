import flax.traverse_util
import numpy as np
import pytest

from openpi.training import weight_loaders


def test_merge_explicitly_allowed_new_vjepa_parameters():
    reference = {
        "base": {"kernel": np.asarray(1.0)},
        "vjepa_query_tokens": {"value": np.asarray(2.0)},
        "vjepa_alignment_out": {"kernel": np.asarray(3.0)},
    }
    loaded = {"base": {"kernel": np.asarray(4.0)}}

    merged = weight_loaders._merge_params(  # noqa: SLF001
        loaded, reference, missing_regex=".*vjepa_.*"
    )
    flat = flax.traverse_util.flatten_dict(merged, sep="/")
    assert flat["base/kernel"] == 4.0
    assert flat["vjepa_query_tokens/value"] == 2.0
    assert flat["vjepa_alignment_out/kernel"] == 3.0


def test_complete_continuation_preserves_new_change_weights(monkeypatch):
    reference = {
        "base": {"kernel": np.zeros(2, dtype=np.float32)},
        "change_in_proj": {"kernel": np.zeros((2, 2), dtype=np.float32)},
    }
    restored = {
        "base": {"kernel": np.ones(2, dtype=np.float32)},
        "change_in_proj": {"kernel": np.full((2, 2), 7, dtype=np.float32)},
    }
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: restored)  # noqa: SLF001
    result = weight_loaders.CheckpointWeightLoader("unused", require_complete=True).load(reference)
    np.testing.assert_array_equal(result["change_in_proj"]["kernel"], restored["change_in_proj"]["kernel"])


def test_complete_continuation_rejects_missing_change_even_if_regex_allows_it(monkeypatch):
    reference = {
        "base": {"kernel": np.zeros(2, dtype=np.float32)},
        "future_context_proj": {"kernel": np.zeros((2, 2), dtype=np.float32)},
    }
    restored = {"base": {"kernel": np.ones(2, dtype=np.float32)}}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: restored)  # noqa: SLF001
    loader = weight_loaders.CheckpointWeightLoader("unused", missing_regex=".*", require_complete=True)
    with pytest.raises(KeyError, match="future_context_proj/kernel"):
        loader.load(reference)


@pytest.mark.parametrize("saved_bias", [True, False])
def test_complete_checkpoint_preserves_disabled_bias(monkeypatch, saved_bias):
    reference = {"layer": {"kernel": np.zeros(2, dtype=np.float32), "bias": None}}
    restored = {"layer": {"kernel": np.ones(2, dtype=np.float32)}}
    if saved_bias:
        restored["layer"]["bias"] = None
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: restored)
    result = weight_loaders.CheckpointWeightLoader("unused", require_complete=True).load(reference)
    assert result["layer"]["bias"] is None
    np.testing.assert_array_equal(result["layer"]["kernel"], restored["layer"]["kernel"])


def test_complete_checkpoint_rejects_none_for_required_array(monkeypatch):
    reference = {"layer": {"kernel": np.zeros(2, dtype=np.float32)}}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: {"layer": {"kernel": None}})
    with pytest.raises(KeyError, match="layer/kernel"):
        weight_loaders.CheckpointWeightLoader("unused", missing_regex=".*", require_complete=True).load(reference)


def test_checkpoint_rejects_array_for_disabled_parameter():
    with pytest.raises(ValueError, match="disabled parameter"):
        weight_loaders._merge_params({"bias": np.zeros(2)}, {"bias": None}, missing_regex=".*")
