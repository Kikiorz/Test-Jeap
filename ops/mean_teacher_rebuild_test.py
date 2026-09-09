import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from precompute_vjepa_multiview_targets import pool_features


def test_pool_matches_original_bfloat16_mean_before_half_cast():
    generator=torch.Generator().manual_seed(42)
    dense=torch.randn((2,576,1408),generator=generator).to(torch.bfloat16)
    reference=dense.reshape(2,8,3,8,3,1408).mean(dim=(2,4)).reshape(2,64,1408)
    actual=pool_features(dense)
    assert actual.dtype==torch.bfloat16
    np.testing.assert_array_equal(actual.to(torch.float16).numpy(),reference.to(torch.float16).numpy())
