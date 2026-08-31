import flax.traverse_util
import numpy as np

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


def test_split_scanned_layers_is_an_exact_axis_partition():
    layers = np.arange(18 * 3, dtype=np.float32).reshape(18, 3)
    params = {
        "PaliGemma": {"llm": {"layers": {"kernel": layers}, "embedder": {"input": np.ones(2)}}}
    }

    split = weight_loaders._split_scanned_layers(params, 16)  # noqa: SLF001
    flat = flax.traverse_util.flatten_dict(split, sep="/")

    np.testing.assert_array_equal(flat["PaliGemma/llm/early_layers/kernel"], layers[:16])
    np.testing.assert_array_equal(flat["PaliGemma/llm/late_layers/kernel"], layers[16:])
    np.testing.assert_array_equal(flat["PaliGemma/llm/embedder/input"], np.ones(2))
