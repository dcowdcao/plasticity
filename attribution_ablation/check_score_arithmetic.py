"""ENTRYPOINT — pin the two-sided score arithmetic on synthetic tensors. No GPU or model; seconds.

These are the parts most likely to be silently wrong: a sign flip would put trait-amplifying edges at
the top of the ranking while the sweep still produced a tidy table.

  python -m attribution_ablation.check_score_arithmetic
"""
import torch

from attribution_ablation.sheaf_attribution import coupling, lm_loss, spread, two_sided_score


def check_sign_convention() -> None:
    """An edge whose ablation reduces misalignment (negative Δ) must score above one that raises it."""
    delta_mis = torch.tensor([-1.0, 0.0, 1.0])
    delta_ben = torch.tensor([1.0, 1.0, 1.0])
    s = two_sided_score(delta_mis, delta_ben, m_mis=10.0, m_ben=10.0)
    assert s.score[0] > s.score[1] > s.score[2], s.score
    assert s.score[0] > 0 > s.score[2], s.score
    assert torch.allclose(s.removal, torch.tensor([0.1, 0.0, -0.1], dtype=torch.float64))
    print(f"sign convention        score(reduces)={s.score[0]:+.4f} > score(no effect)="
          f"{s.score[1]:+.4f} > score(amplifies)={s.score[2]:+.4f}")


def check_descending_order() -> None:
    torch.manual_seed(0)
    delta_mis, delta_ben = torch.randn(5000), torch.randn(5000)
    s = two_sided_score(delta_mis, delta_ben, m_mis=1e4, m_ben=8.0)
    order = s.score.argsort(descending=True)
    ranked = s.score[order]
    assert bool((ranked[:-1] >= ranked[1:]).all())
    top = order[0].item()
    assert s.removal[top] > 0, "the top-ranked edge must reduce misalignment"
    print(f"descending order       top edge removal={s.removal[top]:+.3e} "
          f"perturbation={s.perturbation[top]:.3e}")


def check_floor_bounds_a_zero_denominator() -> None:
    """One edge with exactly zero benign perturbation must stay finite, and must not outrank an
    edge that removes ten times as much misalignment for a comparable cost."""
    delta_mis = torch.cat([torch.tensor([-1.0, -10.0]), -torch.rand(998)])
    delta_ben = torch.cat([torch.tensor([0.0, 1.0]), torch.rand(998) + 0.5])
    s = two_sided_score(delta_mis, delta_ben, m_mis=10.0, m_ben=10.0)
    assert torch.isfinite(s.score).all()
    assert s.floor > 0, s.floor
    assert s.score[1] > s.score[0], (s.score[0], s.score[1])
    without_floor = float(-delta_mis[0] / 10.0) / (abs(float(delta_ben[0])) / 10.0 + 1e-30)
    print(f"floor                  {s.floor:.4e}; zero-benign edge scores {s.score[0]:.3f} "
          f"(unfloored {without_floor:.3e}), below the 10x-better edge at {s.score[1]:.3f}")


def check_scale_free() -> None:
    """Rescaling either metric rescales its Δ by the same factor, so the score cannot move: that is
    what normalising per side buys, and it is why a logit probe and a nats loss can be divided."""
    torch.manual_seed(1)
    delta_mis = torch.randn(2000, dtype=torch.float64) * 1e4
    delta_ben = torch.randn(2000, dtype=torch.float64) * 1e-3
    base = two_sided_score(delta_mis, delta_ben, m_mis=1e4, m_ben=8.0)
    scaled = two_sided_score(delta_mis * 137.0, delta_ben * 0.004,
                             m_mis=1e4 * 137.0, m_ben=8.0 * 0.004)
    gap = float((base.score - scaled.score).abs().max() / base.score.abs().max())
    assert gap < 1e-12, gap
    print(f"scale-free             max relative Δscore under a 137x / 0.004x rescale = {gap:.3e}")


def check_normalisers_must_be_positive() -> None:
    try:
        two_sided_score(torch.randn(4), torch.randn(4), m_mis=0.0, m_ben=8.0)
    except ValueError as e:
        print(f"zero normaliser        raises: {e}")
        return
    raise AssertionError("a zero normaliser must raise, not divide")


def check_coupling_detects_proportionality() -> None:
    """The go/no-go diagnostic must read slope 1 / r² 1 on a proportional pair and near-zero
    correlation on an independent one."""
    torch.manual_seed(2)
    mis = torch.randn(20000)
    tied = coupling(mis, 0.67 * mis)
    free = coupling(mis, torch.randn(20000))
    assert abs(tied.slope - 1.0) < 1e-6 and tied.r2 > 0.9999, tied
    assert abs(free.pearson_log) < 0.05 and abs(free.spearman) < 0.05, free
    print(f"coupling               proportional: slope={tied.slope:.4f} r2={tied.r2:.5f} "
          f"spearman={tied.spearman:.3f} | independent: r2={free.r2:.5f} "
          f"spearman={free.spearman:+.3f}")


def check_spread_separates_flat_from_tailed() -> None:
    removal = torch.arange(5000.0)
    flat = spread(torch.full((5000,), 3.0), removal)
    tailed = spread(torch.cat([torch.full((4900,), 1.0), torch.arange(100.0) + 50.0]), removal)
    assert abs(flat.top100_over_median - 1.0) < 1e-9, flat
    assert tailed.top100_over_median > 50, tailed
    print(f"spread                 flat: top100/median={flat.top100_over_median:.4f} | "
          f"tailed: {tailed.top100_over_median:.1f}")


def check_rank_vs_removal_detects_an_inert_denominator() -> None:
    """The denominator earns its place only if it reorders. A perturbation proportional to the
    removal leaves the ranking untouched — that is the method reducing to necessity attribution."""
    torch.manual_seed(4)
    delta_mis = -torch.rand(4000) - 0.1
    inert = two_sided_score(delta_mis, 0.67 * delta_mis, m_mis=1e4, m_ben=8.0)
    active = two_sided_score(delta_mis, (torch.randn(4000) * 3).exp(), m_mis=1e4, m_ben=8.0)
    s_inert = spread(inert.score, inert.removal)
    s_active = spread(active.score, active.removal)
    assert s_inert.rank_vs_removal > 0.999, s_inert
    assert abs(s_active.rank_vs_removal) < 0.6, s_active
    print(f"rank vs removal        proportional denominator: {s_inert.rank_vs_removal:.4f} | "
          f"independent: {s_active.rank_vs_removal:+.4f}")


def check_lm_loss_ignores_padding() -> None:
    """Right-padded batches: a pad target must not enter the mean, or the benign side measures the
    model's confidence in the pad token. Row 0 is 4 real tokens, row 1 is 2 real + 2 pad."""
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 5)
    label = torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]])
    input_lengths = torch.tensor([4, 2])
    got = float(lm_loss(logits, None, input_lengths, label))
    per_token = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 5), label[:, 1:].reshape(-1), reduction="none").reshape(2, 3)
    want = float(torch.cat([per_token[0, :3], per_token[1, :1]]).mean())
    assert abs(got - want) < 1e-6, (got, want)
    print(f"lm_loss padding        {got:.6f} over 4 real targets, not {float(per_token.mean()):.6f} "
          f"over all 6")


def main() -> None:
    for check in (check_sign_convention, check_descending_order,
                  check_floor_bounds_a_zero_denominator, check_scale_free,
                  check_normalisers_must_be_positive, check_coupling_detects_proportionality,
                  check_spread_separates_flat_from_tailed,
                  check_rank_vs_removal_detects_an_inert_denominator,
                  check_lm_loss_ignores_padding):
        check()
    print("\nall score-arithmetic checks passed")


if __name__ == "__main__":
    main()
