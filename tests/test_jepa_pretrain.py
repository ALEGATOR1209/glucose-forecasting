"""Guards for the JEPA self-supervised stage.

The failure modes here are quiet: a mask that leaks target patches into the
context makes the task trivial, an EMA that never updates makes the target
encoder a frozen random projection, and representation collapse drives the loss
to zero while the encoder learns nothing. Each gets a test.
"""
from __future__ import annotations

import copy
import random

import pytest
import torch

from sugar_jepa.jepa_pretrain import (
    JepaPredictor,
    _forward_loss,
    collapse_metrics,
    ema_update,
    momentum_at,
    sample_block_mask,
    variance_penalty,
)
from sugar_jepa.sugar_jepa_model import JepaEncoder

N_PATCHES, EMBED_DIM, PATCH, STEPS = 16, 32, 8, 128


def _encoder() -> JepaEncoder:
    return JepaEncoder(
        n_time_steps=STEPS, patch_size=PATCH, embed_dim=EMBED_DIM, n_layers=2, n_heads=4
    )


def _predictor() -> JepaPredictor:
    return JepaPredictor(embed_dim=EMBED_DIM, n_patches=N_PATCHES, n_layers=1, n_heads=2)


# --- masking ----------------------------------------------------------------

def test_context_and_target_are_disjoint_and_non_empty():
    rng = random.Random(0)
    for _ in range(200):
        ctx, tgt = sample_block_mask(N_PATCHES, n_targets=4, min_block=2, max_block=4, rng=rng)
        assert ctx and tgt, "a run with an empty context or target teaches nothing"
        assert not set(ctx) & set(tgt), "target patches leaked into the context"
        assert set(ctx) | set(tgt) == set(range(N_PATCHES))


def test_mask_blocks_are_contiguous_runs():
    rng = random.Random(1)
    _ctx, tgt = sample_block_mask(N_PATCHES, n_targets=2, min_block=3, max_block=3, rng=rng)
    # Targets are unions of contiguous blocks, so every run length is >= min_block.
    runs, run = [], 1
    for a, b in zip(tgt, tgt[1:]):
        run = run + 1 if b == a + 1 else (runs.append(run) or 1)
    runs.append(run)
    assert all(r >= 3 for r in runs)


def test_impossible_mask_raises_instead_of_looping_forever():
    rng = random.Random(2)
    with pytest.raises(RuntimeError, match="Could not place"):
        sample_block_mask(N_PATCHES, n_targets=8, min_block=8, max_block=8, rng=rng)


# --- EMA target encoder ------------------------------------------------------

def test_ema_moves_target_toward_online():
    online, target = _encoder(), None
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad = False

    with torch.no_grad():  # make the online encoder differ
        for p in online.parameters():
            p.add_(1.0)

    before = [p.clone() for p in target.parameters()]
    ema_update(target, online, momentum=0.9)

    for p_before, p_after, p_online in zip(before, target.parameters(), online.parameters()):
        assert not torch.allclose(p_before, p_after), "EMA did not update the target encoder"
        expected = 0.9 * p_before + 0.1 * p_online
        torch.testing.assert_close(p_after, expected)


def test_momentum_ramps_to_one():
    assert momentum_at(0, 100, base=0.996) == pytest.approx(0.996, abs=1e-6)
    assert momentum_at(99, 100, base=0.996) == pytest.approx(1.0, abs=1e-6)
    assert momentum_at(0, 100, 0.996) < momentum_at(50, 100, 0.996) < 1.0


# --- collapse detector -------------------------------------------------------

def test_collapse_metrics_flag_a_collapsed_representation():
    collapsed = torch.ones(64, N_PATCHES, EMBED_DIM)  # every window the same vector
    std, rank = collapse_metrics(collapsed)
    assert std < 1e-3
    assert rank < 1.5

    healthy = torch.randn(64, N_PATCHES, EMBED_DIM)
    std, rank = collapse_metrics(healthy)
    assert std > 0.5
    assert rank > EMBED_DIM / 2  # isotropic noise uses all directions


# --- the objective itself ----------------------------------------------------

def test_predictor_output_matches_target_shape():
    torch.manual_seed(0)
    encoder, predictor = _encoder(), _predictor()
    target_encoder = copy.deepcopy(encoder)

    ctx, tgt = sample_block_mask(N_PATCHES, 4, 2, 4, random.Random(0))
    ctx_idx = torch.tensor(ctx)
    tgt_idx = torch.tensor(tgt)

    loss, _pred, _var, full = _forward_loss(
        encoder, target_encoder, predictor, torch.randn(8, STEPS), ctx_idx, tgt_idx
    )
    assert full.shape == (8, N_PATCHES, EMBED_DIM)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_loss_decreases_on_a_tiny_overfit_batch():
    """If the objective cannot be driven down on one fixed batch and one fixed
    mask, the wiring is wrong — this is the cheapest end-to-end check there is."""
    torch.manual_seed(0)
    encoder, predictor = _encoder(), _predictor()
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    glucose = torch.randn(16, STEPS)
    ctx, tgt = sample_block_mask(N_PATCHES, 4, 2, 4, random.Random(0))
    ctx_idx, tgt_idx = torch.tensor(ctx), torch.tensor(tgt)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    first = None
    for i in range(60):
        loss, _pred, _var, _full = _forward_loss(
            encoder, target_encoder, predictor, glucose, ctx_idx, tgt_idx
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # target encoder frozen here on purpose: with EMA chasing the online
        # encoder, a falling loss could just be the two collapsing together.
        if i == 0:
            first = loss.item()

    assert loss.item() < first * 0.5, f"loss barely moved: {first:.5f} -> {loss.item():.5f}"


def test_encoder_gets_gradients_through_the_context_path():
    """The `keep` gather must not sever the graph — if it does, only the
    predictor trains and the encoder stays at its random init forever."""
    torch.manual_seed(0)
    encoder, predictor = _encoder(), _predictor()
    target_encoder = copy.deepcopy(encoder)

    ctx, tgt = sample_block_mask(N_PATCHES, 4, 2, 4, random.Random(0))
    loss, _pred, _var, _full = _forward_loss(
        encoder, target_encoder, predictor, torch.randn(8, STEPS),
        torch.tensor(ctx), torch.tensor(tgt),
    )
    loss.backward()

    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads, "no gradient reached the context encoder"
    assert any(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in target_encoder.parameters()), "target encoder must not train"


# --- variance floor (anti-collapse) -----------------------------------------

def test_variance_penalty_is_zero_when_healthy():
    """No gradient at all while the representation is fine — this is a floor,
    not a force acting during normal training."""
    healthy = torch.randn(64, N_PATCHES, EMBED_DIM)  # per-dim std ~1.0
    assert variance_penalty(healthy, target_std=0.5).item() == 0.0


def test_variance_penalty_grows_as_representation_contracts():
    torch.manual_seed(0)
    base = torch.randn(64, N_PATCHES, EMBED_DIM)
    mild = variance_penalty(base * 0.4, target_std=0.5).item()   # std ~0.4
    severe = variance_penalty(base * 0.05, target_std=0.5).item()  # std ~0.05
    collapsed = variance_penalty(torch.ones(64, N_PATCHES, EMBED_DIM), target_std=0.5).item()

    assert 0.0 < mild < severe < collapsed
    assert collapsed == pytest.approx(0.5, abs=1e-3)  # std 0 -> full hinge


def test_variance_penalty_pushes_std_back_up():
    """The whole point: optimising it must *undo* a contraction.

    Uses AdamW because that is what pretraining uses. The hinge's raw gradient is
    small (it is divided by batch size and by embed_dim), so under plain SGD it
    barely moves — which is precisely why VICReg pairs it with a large coefficient
    and why --var-reg-weight defaults to 25.
    """
    torch.manual_seed(0)
    z = (torch.randn(128, EMBED_DIM) * 0.1).requires_grad_(True)  # collapsing
    opt = torch.optim.AdamW([z], lr=0.05)
    before = z.std(dim=0).mean().item()

    for _ in range(200):
        loss = variance_penalty(z, target_std=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()

    after = z.std(dim=0).mean().item()
    assert after > before * 2, f"std did not recover: {before:.3f} -> {after:.3f}"
    assert variance_penalty(z, target_std=0.5).item() < 0.01  # floor reached


def test_var_weight_zero_leaves_the_loss_untouched():
    torch.manual_seed(0)
    encoder, predictor = _encoder(), _predictor()
    target_encoder = copy.deepcopy(encoder)
    ctx, tgt = sample_block_mask(N_PATCHES, 4, 2, 4, random.Random(0))
    args = (encoder, target_encoder, predictor, torch.randn(8, STEPS),
            torch.tensor(ctx), torch.tensor(tgt))

    total, pred, var, _ = _forward_loss(*args, var_weight=0.0)
    assert var.item() == 0.0
    torch.testing.assert_close(total, pred)
