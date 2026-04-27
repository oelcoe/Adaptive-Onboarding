from __future__ import annotations

import numpy as np
import pytest

from src.item_bank import Item
from src.synthetic import (
    axis_alignment_score,
    axis_alignment_scores,
    generate_user_population,
    item_bank_to_dataframe,
    item_bank_to_records,
    sample_theta_true,
    select_sensitivity_examples,
    synthetic_item_bank,
)


class TestSyntheticItemBank:
    def test_returns_items_with_expected_shapes(self) -> None:
        bank = synthetic_item_bank(
            n_items=8,
            dim=2,
            n_categories=5,
            sensitive_fraction=0.25,
            rng_seed=11,
        )

        assert len(bank) == 8
        assert all(isinstance(item, Item) for item in bank)
        assert all(item.a.shape == (2,) for item in bank)
        assert all(item.thresholds.shape == (4,) for item in bank)
        assert all(item.n_categories == 5 for item in bank)

    def test_2d_directions_follow_even_angle_construction_without_jitter(self) -> None:
        bank = synthetic_item_bank(
            n_items=4,
            dim=2,
            n_categories=2,
            angle_jitter=0.0,
            threshold_perturbation=0.0,
        )

        expected_angles = np.array([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0])
        expected = np.column_stack((np.cos(expected_angles), np.sin(expected_angles)))

        assert np.allclose([item.a for item in bank], expected)
        assert np.allclose([np.linalg.norm(item.a) for item in bank], 1.0)

    def test_thresholds_are_ordered_and_can_be_exactly_symmetric(self) -> None:
        bank = synthetic_item_bank(
            n_items=3,
            n_categories=4,
            threshold_span=1.5,
            threshold_perturbation=0.0,
        )

        for item in bank:
            assert np.allclose(item.thresholds, np.array([-1.5, 0.0, 1.5]))
            assert np.all(np.diff(item.thresholds) > 0.0)

    def test_sensitive_fraction_marks_exact_rounded_count(self) -> None:
        bank = synthetic_item_bank(
            n_items=10,
            sensitive_fraction=0.3,
            rng_seed=7,
        )

        assert sum(item.is_sensitive for item in bank) == 3
        assert [item.sensitivity_level for item in bank if item.is_sensitive] == [1.0] * 3
        assert all(item.sensitivity_level == 0.0 for item in bank if not item.is_sensitive)

    def test_sensitive_fraction_accepts_percentages(self) -> None:
        bank = synthetic_item_bank(
            n_items=20,
            sensitive_fraction=25.0,
            rng_seed=7,
        )

        assert sum(item.is_sensitive for item in bank) == 5

    def test_vary_sensitivity_levels_samples_only_sensitive_items(self) -> None:
        bank = synthetic_item_bank(
            n_items=12,
            sensitive_fraction=0.5,
            vary_sensitivity_levels=True,
            sensitivity_level_range=(0.2, 0.4),
            rng_seed=22,
        )

        sensitive_levels = [item.sensitivity_level for item in bank if item.is_sensitive]
        assert len(sensitive_levels) == 6
        assert all(0.2 <= level <= 0.4 for level in sensitive_levels)
        assert len(set(sensitive_levels)) > 1
        assert all(item.sensitivity_level == 0.0 for item in bank if not item.is_sensitive)

    def test_axis_aligned_assignment_labels_most_aligned_items_sensitive(self) -> None:
        bank = synthetic_item_bank(
            n_items=8,
            dim=2,
            sensitive_fraction=0.25,
            sensitivity_assignment="axis_aligned",
            angle_jitter=0.0,
            threshold_perturbation=0.0,
        )

        alignment = np.array([axis_alignment_score(item.a) for item in bank])
        sensitive_indices = {index for index, item in enumerate(bank) if item.is_sensitive}
        expected_indices = set(np.argsort(-alignment, kind="stable")[:2])

        assert sensitive_indices == expected_indices

    def test_axis_aligned_assignment_can_couple_levels_to_alignment(self) -> None:
        bank = synthetic_item_bank(
            n_items=8,
            dim=2,
            sensitive_fraction=1.0,
            sensitivity_assignment="axis_aligned",
            couple_sensitivity_level_to_axis_alignment=True,
            sensitivity_level_range=(0.2, 0.8),
            angle_jitter=0.0,
            threshold_perturbation=0.0,
        )

        alignment = np.array([axis_alignment_score(item.a) for item in bank])
        levels = np.array([item.sensitivity_level for item in bank])

        assert levels.min() == pytest.approx(0.2)
        assert levels.max() == pytest.approx(0.8)
        assert np.corrcoef(alignment, levels)[0, 1] > 0.99

    def test_seed_reproducibility(self) -> None:
        first = synthetic_item_bank(
            n_items=9,
            sensitive_fraction=0.4,
            vary_sensitivity_levels=True,
            rng_seed=123,
        )
        second = synthetic_item_bank(
            n_items=9,
            sensitive_fraction=0.4,
            vary_sensitivity_levels=True,
            rng_seed=123,
        )

        assert [item.item_id for item in first] == [item.item_id for item in second]
        assert [item.is_sensitive for item in first] == [item.is_sensitive for item in second]
        assert np.allclose([item.a for item in first], [item.a for item in second])
        assert np.allclose(
            [item.thresholds for item in first],
            [item.thresholds for item in second],
        )
        assert np.allclose(
            [item.sensitivity_level for item in first],
            [item.sensitivity_level for item in second],
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_items": 0}, "n_items"),
            ({"dim": 0}, "dim"),
            ({"n_categories": 1}, "n_categories"),
            ({"sensitive_fraction": -0.1}, "sensitive_fraction"),
            ({"sensitive_fraction": 101.0}, "sensitive_fraction"),
            ({"sensitivity_level_range": (-0.1, 1.0)}, "sensitivity_level_range"),
            ({"sensitivity_assignment": "unknown"}, "sensitivity_assignment"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs: dict[str, object], match: str) -> None:
        call_kwargs = {"n_items": 5}
        call_kwargs.update(kwargs)
        with pytest.raises(ValueError, match=match):
            synthetic_item_bank(**call_kwargs)


class TestGenerateUserPopulation:
    def test_returns_requested_shape(self) -> None:
        theta = generate_user_population(n_users=7, dim=3, rng_seed=10)

        assert theta.shape == (7, 3)

    def test_seed_reproducibility(self) -> None:
        first = generate_user_population(n_users=5, dim=2, rng_seed=123)
        second = generate_user_population(n_users=5, dim=2, rng_seed=123)

        assert np.allclose(first, second)

    def test_custom_mean_and_covariance(self) -> None:
        theta = generate_user_population(
            n_users=4,
            dim=2,
            mean=np.array([1.0, -1.0]),
            covariance=np.array([[2.0, 0.3], [0.3, 0.5]]),
            rng_seed=4,
        )

        assert theta.shape == (4, 2)
        assert np.all(np.isfinite(theta))

    def test_accepts_generator_and_advances_it(self) -> None:
        rng_a = np.random.default_rng(9)
        rng_b = np.random.default_rng(9)

        theta = generate_user_population(n_users=3, dim=2, rng=rng_a)
        expected = rng_b.multivariate_normal(
            mean=np.zeros(2),
            cov=np.eye(2),
            size=3,
        )

        assert np.allclose(theta, expected)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_users": 0}, "n_users"),
            ({"dim": 0}, "dim"),
            ({"mean": [0.0, 1.0, 2.0]}, "mean"),
            ({"mean": [0.0, np.inf]}, "mean"),
            ({"covariance": np.eye(3)}, "covariance"),
            ({"covariance": [[1.0, 0.2], [0.0, 1.0]]}, "symmetric"),
            ({"covariance": [[1.0, 0.0], [0.0, -0.1]]}, "positive semidefinite"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs: dict[str, object], match: str) -> None:
        call_kwargs = {"n_users": 5, "dim": 2}
        call_kwargs.update(kwargs)
        with pytest.raises(ValueError, match=match):
            generate_user_population(**call_kwargs)

    def test_rng_and_seed_together_raise(self) -> None:
        with pytest.raises(ValueError, match="rng or rng_seed"):
            generate_user_population(
                n_users=5,
                dim=2,
                rng=np.random.default_rng(1),
                rng_seed=1,
            )


class TestSampleThetaTrue:
    def test_returns_one_dimensional_theta(self) -> None:
        theta = sample_theta_true(dim=3, rng_seed=10)

        assert theta.shape == (3,)

    def test_matches_single_row_population_sample(self) -> None:
        theta = sample_theta_true(
            dim=2,
            mean=[1.0, -1.0],
            covariance=[[1.0, 0.2], [0.2, 0.5]],
            rng_seed=42,
        )
        population = generate_user_population(
            n_users=1,
            dim=2,
            mean=[1.0, -1.0],
            covariance=[[1.0, 0.2], [0.2, 0.5]],
            rng_seed=42,
        )

        assert np.allclose(theta, population[0])


class TestItemBankExperimentHelpers:
    def test_item_bank_to_dataframe_exposes_arrays_and_scalar_columns(self) -> None:
        bank = synthetic_item_bank(
            n_items=4,
            dim=2,
            n_categories=3,
            sensitive_fraction=0.5,
            angle_jitter=0.0,
            threshold_perturbation=0.0,
            rng_seed=3,
        )

        df = item_bank_to_dataframe(bank)

        assert len(df) == 4
        assert df.loc[0, "item_id"] == "item_0000"
        assert df.loc[0, "dim"] == 2
        assert df.loc[0, "n_categories"] == 3
        assert "a" in df.columns
        assert "thresholds" in df.columns
        assert "a_0" in df.columns
        assert "a_1" in df.columns
        assert "threshold_0" in df.columns
        assert "threshold_1" in df.columns
        assert "angle" in df.columns
        assert "axis_alignment" in df.columns

    def test_dataframe_copies_array_values(self) -> None:
        bank = synthetic_item_bank(n_items=1, threshold_perturbation=0.0)
        df = item_bank_to_dataframe(bank)

        df.loc[0, "a"][0] = 99.0
        df.loc[0, "thresholds"][0] = 99.0

        assert bank[0].a[0] != 99.0
        assert bank[0].thresholds[0] != 99.0

    def test_item_bank_to_records_still_available_for_plain_python_use(self) -> None:
        bank = synthetic_item_bank(n_items=2, threshold_perturbation=0.0)
        records = item_bank_to_records(bank)

        assert isinstance(records, list)
        assert records[0]["item_id"] == "item_0000"

    def test_select_sensitivity_examples_returns_one_of_each_type(self) -> None:
        bank = synthetic_item_bank(
            n_items=6,
            sensitive_fraction=0.5,
            rng_seed=5,
        )

        non_sensitive_item, sensitive_item = select_sensitivity_examples(bank)

        assert non_sensitive_item.is_sensitive is False
        assert sensitive_item.is_sensitive is True

    def test_select_sensitivity_examples_raises_when_one_type_missing(self) -> None:
        bank = synthetic_item_bank(
            n_items=3,
            sensitive_fraction=0.0,
            rng_seed=5,
        )

        with pytest.raises(ValueError, match="non-sensitive item and one sensitive item"):
            select_sensitivity_examples(bank)


class TestAxisAlignment:
    def test_axis_alignment_score_is_highest_on_axes(self) -> None:
        assert axis_alignment_score(np.array([1.0, 0.0])) == pytest.approx(1.0)
        assert axis_alignment_score(np.array([0.0, 1.0])) == pytest.approx(1.0)
        assert axis_alignment_score(np.array([1.0, 1.0])) == pytest.approx(1 / np.sqrt(2))

    def test_axis_alignment_scores_vectorizes_rows(self) -> None:
        directions = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )

        assert np.allclose(axis_alignment_scores(directions), [1.0, 1 / np.sqrt(2)])
