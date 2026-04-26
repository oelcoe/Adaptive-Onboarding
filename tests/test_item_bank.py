"""Tests for src/item_bank.py — Item dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from src.item_bank import BASE_OBSERVATION_NOISE_VARIANCE, Item


A = np.array([1.0, 0.0])
THR = np.array([-1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# is_sensitive flag
# ---------------------------------------------------------------------------


class TestItemIsSensitive:
    def test_defaults_to_false(self) -> None:
        item = Item("q", a=A, thresholds=THR)
        assert item.is_sensitive is False

    def test_can_be_set_true(self) -> None:
        item = Item("q", a=A, thresholds=THR, is_sensitive=True)
        assert item.is_sensitive is True

    def test_flag_does_not_affect_noise_variance(self) -> None:
        item_a = Item("q", a=A, thresholds=THR, is_sensitive=False)
        item_b = Item("q", a=A, thresholds=THR, is_sensitive=True)
        assert item_a.observation_noise_variance == item_b.observation_noise_variance


# ---------------------------------------------------------------------------
# sensitivity_level field
# ---------------------------------------------------------------------------


class TestItemSensitivityLevel:
    def test_defaults_to_zero(self) -> None:
        item = Item("q", a=A, thresholds=THR)
        assert item.sensitivity_level == 0.0

    def test_can_be_set(self) -> None:
        item = Item("q", a=A, thresholds=THR, sensitivity_level=0.7)
        assert item.sensitivity_level == pytest.approx(0.7)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="sensitivity_level"):
            Item("q", a=A, thresholds=THR, sensitivity_level=-0.1)

    def test_zero_is_valid(self) -> None:
        Item("q", a=A, thresholds=THR, sensitivity_level=0.0)

    def test_independent_of_is_sensitive(self) -> None:
        Item("q", a=A, thresholds=THR, is_sensitive=True,  sensitivity_level=0.0)
        Item("q", a=A, thresholds=THR, is_sensitive=False, sensitivity_level=0.8)
        Item("q", a=A, thresholds=THR, is_sensitive=True,  sensitivity_level=0.8)

    def test_does_not_affect_observation_noise_variance(self) -> None:
        item_low  = Item("q", a=A, thresholds=THR, sensitivity_level=0.0)
        item_high = Item("q", a=A, thresholds=THR, sensitivity_level=1.0)
        assert item_low.observation_noise_variance == item_high.observation_noise_variance

    def test_noise_inflation_happens_at_simulation_time_not_construction(self) -> None:
        """
        Noise is only inflated when simulate_episode is called with
        sensitivity_noise_scale > 0. The item's own stored value stays at
        BASE_OBSERVATION_NOISE_VARIANCE regardless of sensitivity_level.
        """
        item = Item("q", a=A, thresholds=THR, is_sensitive=True, sensitivity_level=1.0)
        assert item.observation_noise_variance == pytest.approx(BASE_OBSERVATION_NOISE_VARIANCE)


# ---------------------------------------------------------------------------
# response_noise_variance field
# ---------------------------------------------------------------------------


class TestItemResponseNoiseVariance:
    def test_defaults_to_base_constant(self) -> None:
        item = Item("q", a=A, thresholds=THR)
        assert item.response_noise_variance == pytest.approx(BASE_OBSERVATION_NOISE_VARIANCE)

    def test_can_be_overridden(self) -> None:
        item = Item("q", a=A, thresholds=THR, response_noise_variance=2.5)
        assert item.observation_noise_variance == pytest.approx(2.5)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="response_noise_variance"):
            Item("q", a=A, thresholds=THR, response_noise_variance=0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="response_noise_variance"):
            Item("q", a=A, thresholds=THR, response_noise_variance=-1.0)
