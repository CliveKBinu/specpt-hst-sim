import inspect

import numpy as np
import torch

from src.specpt.losses import BinnedRedshiftLoss
from src.specpt.model import BinnedRedshiftHead, EnhancedSpecPTForRedshift, binned_predict
from src.specpt.training.eval import decode_redshift_output


def test_bins_are_uniform_in_log1p_space_and_outputs_are_bounded():
    head = BinnedRedshiftHead(8, num_bins=4, z_bin_max=3.0, dropout_rate=0.0)
    expected = np.linspace(0.0, np.log1p(3.0), 5)
    actual = head.bin_left_log1p.detach().numpy()
    np.testing.assert_allclose(actual, expected[:-1])
    assert np.isclose(head.bin_width, expected[1] - expected[0])

    outputs = head(torch.randn(6, 8))
    assert outputs["conf_logits"].shape == (6, 4)
    assert outputs["within_z"].shape == (6, 4)
    assert torch.all(outputs["within_z"] >= 0)
    assert torch.all(outputs["within_z"] <= 3.0)


def test_binned_loss_returns_components_and_backpropagates():
    head = BinnedRedshiftHead(8, num_bins=4, z_bin_max=3.0, dropout_rate=0.0)
    criterion = BinnedRedshiftLoss(num_bins=4, z_bin_max=3.0)
    loss_out = criterion(head(torch.randn(5, 8)), torch.tensor([[0.1], [0.7], [1.4], [2.1], [2.9]]))

    assert set(loss_out) == {"total", "loss_cls", "loss_refine", "loss_nmad"}
    assert loss_out["total"].ndim == 0
    loss_out["total"].backward()
    assert head.conf_logits.weight.grad is not None
    assert head.refine.weight.grad is not None


def test_argmax_decode_is_headline_prediction():
    outputs = {
        "conf_logits": torch.tensor([[0.0, 5.0, 0.0]]),
        "within_z": torch.tensor([[0.1, 1.2, 2.5]]),
    }
    z_argmax, z_soft, probabilities = binned_predict(outputs)
    assert torch.equal(z_argmax, torch.tensor([1.2]))
    assert z_soft.item() != z_argmax.item()
    assert torch.equal(decode_redshift_output(outputs, "binned"), z_argmax)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1))


def test_point_head_default_remains_disabled():
    signature = inspect.signature(EnhancedSpecPTForRedshift.__init__)
    assert signature.parameters["binned_output"].default is False
