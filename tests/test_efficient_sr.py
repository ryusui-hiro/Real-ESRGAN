import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efficient_sr import ECALite, ERBlock, EfficientSRNet


def test_erblock_shape_and_residual():
    torch.manual_seed(0)
    block = ERBlock(32)
    x = torch.randn(1, 32, 64, 64)
    y = block(x)

    assert y.shape == x.shape
    # Residual path should introduce non-zero differences for random input
    assert torch.allclose(y, x) is False
    assert torch.count_nonzero(y - x) > 0


def test_attention_outputs_between_zero_and_one():
    torch.manual_seed(1)
    attn = ECALite(16)
    x = torch.randn(1, 16, 8, 8)
    _ = attn(x)
    attention_map = attn.latest_attention
    assert attention_map is not None
    assert torch.all(attention_map >= 0.0)
    assert torch.all(attention_map <= 1.0)
    # The attention should not collapse to a constant value
    assert torch.var(attention_map) > 0


def test_model_shapes_and_nan_free_output():
    torch.manual_seed(2)
    model = EfficientSRNet()
    x = torch.randn(1, 3, 16, 16)
    y = model(x)
    assert y.shape == (1, 3, 64, 64)
    assert torch.isfinite(y).all()


def test_parameter_budget():
    model = EfficientSRNet()
    total_params = EfficientSRNet.count_parameters(model)
    # Ensure the parameter count remains well below multi-million levels typical of Real-ESRGAN
    assert total_params < 2_000_000


def test_infer_shape_helper():
    model = EfficientSRNet(upscale=2)
    shape = EfficientSRNet.infer_shape(model, (1, 3, 10, 10))
    assert shape == (1, 3, 20, 20)
