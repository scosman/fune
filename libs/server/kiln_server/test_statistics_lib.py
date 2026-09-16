"""Unit tests for the pure-stats helpers.

Regression anchors are known worked examples (e.g. McNemar discordant counts
b=13, c=6 -> exact two-sided p=0.1671).
"""

import math

import pytest

from kiln_server.statistics_lib import (
    bootstrap_difference_ci,
    mcnemar_chi2_cc,
    mcnemar_exact_p,
    paired_bootstrap_diff_ci,
    paired_proportion_diff_ci,
    percentile,
    wilcoxon_signed_rank_p,
    wilson_ci,
    wilson_difference_ci,
    z_for_confidence,
)


class TestZForConfidence:
    @pytest.mark.parametrize(
        "confidence,expected",
        [(0.95, 1.95996), (0.99, 2.57583), (0.90, 1.64485), (0.80, 1.28155)],
    )
    def test_known_values(self, confidence: float, expected: float) -> None:
        assert z_for_confidence(confidence) == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            z_for_confidence(bad)


class TestWilson:
    def test_proportion_cell(self) -> None:
        # 127/147 ~= 86.4%; SE = sqrt(p(1-p)/n) ~= 2.83pp.
        low, high = wilson_ci(127, 147)
        assert 0.79 < low < 0.81
        assert 0.90 < high < 0.92
        p = 127 / 147
        assert math.sqrt(p * (1 - p) / 147) == pytest.approx(0.0283, abs=5e-4)

    def test_zero_total(self) -> None:
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_difference_sign_and_none(self) -> None:
        # p_b > p_a -> positive delta.
        out = wilson_difference_ci(50, 100, 70, 100)
        assert out is not None
        delta, low, high = out
        assert delta == pytest.approx(0.2, abs=1e-9)
        assert low < high
        assert wilson_difference_ci(0, 0, 5, 10) is None


class TestPercentile:
    """Gold values are numpy.percentile's default (linear interpolation)."""

    def test_empty_is_none(self) -> None:
        # None, not 0.0 — a 0 would read downstream as a real datum.
        assert percentile([], 50) is None
        assert percentile([], 90) is None

    def test_single_value(self) -> None:
        assert percentile([7.5], 0) == 7.5
        assert percentile([7.5], 50) == 7.5
        assert percentile([7.5], 100) == 7.5

    def test_odd_length_median_is_middle(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 50) == pytest.approx(2.0)

    def test_even_length_median_interpolates(self) -> None:
        # Average of the two middle values, not the lower one.
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_unsorted_input(self) -> None:
        assert percentile([5.0, 1.0, 4.0, 2.0, 3.0], 50) == pytest.approx(3.0)

    @pytest.mark.parametrize(
        "p,expected",
        [(0, 1.0), (25, 3.25), (50, 5.5), (75, 7.75), (90, 9.1), (100, 10.0)],
    )
    def test_matches_numpy_defaults(self, p: float, expected: float) -> None:
        values = [float(v) for v in range(1, 11)]
        assert percentile(values, p) == pytest.approx(expected)

    def test_all_equal(self) -> None:
        assert percentile([2.0, 2.0, 2.0], 90) == pytest.approx(2.0)

    @pytest.mark.parametrize("bad", [-1, 101, 150.0])
    def test_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], bad)


class TestMcNemar:
    @pytest.mark.parametrize(
        "b,c,expected_p",
        [(13, 6, 0.1671), (10, 4, 0.1796)],
    )
    def test_exact_matches_gold(self, b: int, c: int, expected_p: float) -> None:
        assert mcnemar_exact_p(b, c) == pytest.approx(expected_p, abs=1e-4)

    def test_exact_symmetric(self) -> None:
        assert mcnemar_exact_p(13, 6) == mcnemar_exact_p(6, 13)

    def test_exact_no_discordant(self) -> None:
        assert mcnemar_exact_p(0, 0) == 1.0

    @pytest.mark.parametrize(
        "b,c,chi2,p",
        [(13, 6, 1.895, 0.1687), (10, 4, 1.786, 0.1814)],
    )
    def test_chi2_cc_matches_gold(self, b: int, c: int, chi2: float, p: float) -> None:
        got_chi2, got_p = mcnemar_chi2_cc(b, c)
        assert got_chi2 == pytest.approx(chi2, abs=1e-3)
        assert got_p == pytest.approx(p, abs=1e-3)

    def test_chi2_cc_no_discordant(self) -> None:
        assert mcnemar_chi2_cc(0, 0) == (0.0, 1.0)


class TestPairedProportionDiffCI:
    def test_paired_table(self) -> None:
        # 2x2: n11=51, n10=13, n01=6, n00=34 (n=104). delta = (6-13)/104 = -0.0673.
        out = paired_proportion_diff_ci(51, 13, 6, 34)
        assert out is not None
        delta, low, high = out
        assert delta == pytest.approx(-0.0673, abs=1e-4)
        # Not significant -> CI straddles zero (consistent with exact p=0.167).
        assert low < 0 < high

    def test_empty_table(self) -> None:
        assert paired_proportion_diff_ci(0, 0, 0, 0) is None


class TestWilcoxon:
    def test_too_few_nonzero(self) -> None:
        assert wilcoxon_signed_rank_p([1.0, 2.0, 3.0, 4.0]) is None

    def test_zeros_dropped_below_threshold(self) -> None:
        # Only 4 non-zero after dropping zeros -> None.
        assert wilcoxon_signed_rank_p([0, 0, 1.0, 2.0, 3.0, 4.0]) is None

    def test_consistent_shift_significant(self) -> None:
        p = wilcoxon_signed_rank_p([1.0, 1.0, 2.0, 1.5, 2.0, 1.0])
        assert p is not None
        assert p < 0.05

    def test_symmetric_sample_p_is_one(self) -> None:
        # Perfectly balanced ranks (w_plus == mu) must give z = 0 / p = 1.0,
        # not a spurious p < 1 from the continuity correction overshooting 0.
        p = wilcoxon_signed_rank_p([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
        assert p == 1.0


class TestBootstrapDeterminism:
    def test_paired_bootstrap_deterministic(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.5, 2.4, 3.6, 4.1, 5.9]
        assert paired_bootstrap_diff_ci(a, b) == paired_bootstrap_diff_ci(a, b)

    def test_paired_bootstrap_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap_diff_ci([1.0, 2.0], [1.0])

    def test_diff_bootstrap_deterministic(self) -> None:
        assert bootstrap_difference_ci(60, 100, 70, 100) == bootstrap_difference_ci(
            60, 100, 70, 100
        )

    def test_diff_bootstrap_zero_total(self) -> None:
        assert bootstrap_difference_ci(0, 0, 5, 10) is None
