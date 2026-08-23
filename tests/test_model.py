import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import AttentionPooling, LastFramePooling, MaxPooling, MeanPooling, TemporalTurnClassifier


def test_attention_pooling_shapes_mask_and_gradients():
    states = torch.randn(2, 5, 8, requires_grad=True)
    pooled = AttentionPooling(8)(states, torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]))
    assert pooled.shape == (2, 8)
    pooled.sum().backward()
    assert torch.isfinite(states.grad).all()


def test_attention_rejects_all_padded_item():
    with pytest.raises(ValueError, match="unmasked"):
        AttentionPooling(4)(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.long))


def test_mean_and_max_and_last_pooling_shape_and_masking():
    states = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    ])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    mean_out = MeanPooling()(states, mask)
    assert mean_out.shape == (2, 2)
    assert torch.allclose(mean_out[0], torch.tensor([2.0, 3.0]))

    max_out = MaxPooling()(states, mask)
    assert max_out.shape == (2, 2)
    assert torch.allclose(max_out[0], torch.tensor([3.0, 4.0]))

    last_out = LastFramePooling()(states, mask)
    assert last_out.shape == (2, 2)
    assert torch.allclose(last_out[0], torch.tensor([3.0, 4.0]))


def test_temporal_classifier_output_shape():
    assert TemporalTurnClassifier(8, (4,))(torch.randn(3, 6, 8)).shape == (3,)
    assert TemporalTurnClassifier(8, (4,), pooling="mean")(torch.randn(3, 6, 8)).shape == (3,)
    assert TemporalTurnClassifier(8, (4,), pooling="last")(torch.randn(3, 6, 8)).shape == (3,)
    assert TemporalTurnClassifier(8, (4,), pooling="max")(torch.randn(3, 6, 8)).shape == (3,)
