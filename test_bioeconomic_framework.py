"""Regression tests for the illustrative macroalgae bioeconomic model."""
import unittest
from dataclasses import replace

import numpy as np

from bioeconomic_framework import (
    Parameters,
    cohort_schedule,
    replacement_days,
    simulate,
    threshold_frontier,
)


class TerminalPolicyTests(unittest.TestCase):
    def test_only_harvestable_cohorts_are_sown(self) -> None:
        parameters = Parameters()
        schedule = cohort_schedule(parameters)

        self.assertTrue((schedule["harvest_day"] <= parameters.horizon_days).all())
        self.assertEqual(int(schedule["sowing_day"].max()), 3510)
        self.assertEqual(int(schedule["harvest_day"].max()), 3630)

        result = simulate(parameters)
        self.assertEqual(int(result["sowing_cohorts"].sum()), len(schedule))
        self.assertEqual(int(result["harvest_cohorts"].sum()), len(schedule))

    def test_terminal_reinvestment_is_excluded(self) -> None:
        parameters = Parameters()
        self.assertListEqual(replacement_days(parameters).tolist(), [1825])

        result = simulate(parameters)
        terminal = result.loc[result["day"] == parameters.horizon_days].iloc[0]
        self.assertEqual(float(terminal["reinvestment"]), 0.0)


class TaxTests(unittest.TestCase):
    def test_depreciation_reduces_taxable_income_without_being_a_cash_outflow(self) -> None:
        parameters = Parameters(
            horizon_days=2,
            cycle_days=1,
            sowing_interval_days=1,
            daily_fixed_cost=0.0,
            sowing_cost_per_cohort=0.0,
            harvest_cost_per_kg=0.0,
            initial_investment=200.0,
            replacement_interval_days=0,
            asset_depreciation_life_days=2,
            residual_value_fraction=0.0,
            annual_discount_rate=0.0,
            tax_rate=0.30,
        )
        result = simulate(parameters)
        day_one = result.loc[result["day"] == 1].iloc[0]

        expected_tax = max(
            day_one["revenue"] - day_one["operating_cost"] - day_one["depreciation"],
            0.0,
        ) * parameters.tax_rate
        self.assertAlmostEqual(float(day_one["depreciation"]), 100.0)
        self.assertAlmostEqual(float(day_one["tax"]), expected_tax)
        self.assertAlmostEqual(float(result["depreciation"].sum()), 200.0)
        expected_fcf = (
            day_one["revenue"]
            - day_one["operating_cost"]
            - day_one["tax"]
            - day_one["investment"]
            - day_one["reinvestment"]
            + day_one["residual_value"]
        )
        self.assertAlmostEqual(float(day_one["free_cash_flow"]), expected_fcf)

    def test_zero_tax_rate_leaves_tax_at_zero(self) -> None:
        result = simulate(Parameters(tax_rate=0.0))
        self.assertTrue(np.allclose(result["tax"].to_numpy(), 0.0))


class ThresholdTests(unittest.TestCase):
    def test_frontier_is_zero_npv_at_the_reported_break_even_price(self) -> None:
        parameters = Parameters()
        frontier = threshold_frontier(parameters, np.array([1.0]))
        price_multiplier = float(frontier.iloc[0]["break_even_price_multiplier"])
        result = simulate(
            replace(parameters, sale_price_per_kg=parameters.sale_price_per_kg * price_multiplier)
        )
        self.assertAlmostEqual(float(result["cumulative_npv"].iloc[-1]), 0.0, places=7)

    def test_frontier_marks_the_default_baseline_as_infeasible(self) -> None:
        parameters = Parameters()
        frontier = threshold_frontier(parameters, np.array([1.0]))
        self.assertGreater(float(frontier.iloc[0]["break_even_price_multiplier"]), 1.0)

    def test_frontier_remains_consistent_with_a_positive_tax_rate(self) -> None:
        parameters = Parameters(tax_rate=0.25)
        frontier = threshold_frontier(parameters, np.array([1.0]))
        price_multiplier = float(frontier.iloc[0]["break_even_price_multiplier"])
        result = simulate(
            replace(parameters, sale_price_per_kg=parameters.sale_price_per_kg * price_multiplier)
        )
        self.assertAlmostEqual(float(result["cumulative_npv"].iloc[-1]), 0.0, places=6)


class ValidationTests(unittest.TestCase):
    def test_misaligned_events_are_rejected(self) -> None:
        parameters = Parameters(time_step_days=2, sowing_interval_days=3)
        with self.assertRaisesRegex(ValueError, "sowing_interval_days"):
            simulate(parameters)


if __name__ == "__main__":
    unittest.main()
