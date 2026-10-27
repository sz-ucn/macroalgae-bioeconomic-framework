"""Illustrative daily-step bioeconomic model for macroalgae cultivation.

The model combines continuous Gompertz growth with discrete sowing, harvest,
capital-replacement and cash-flow events. The numerical implementation uses
a daily grid; it is therefore a discrete approximation to a continuous-time
bioeconomic framework, rather than a literally continuous solver.

Terminal policy
---------------
Only cohorts that can complete their cultivation cycle within the stated
project horizon are sown. Replacement investment is booked strictly before
the terminal day. These rules prevent terminal expenditures with no modeled
benefit during the evaluation period.

Tax policy
----------
Taxable income equals revenue minus operating cost minus straight-line tax
depreciation. Negative taxable income is not refunded and is not carried
forward. Users requiring loss carry-forwards, tax credits, or tax treatment
of asset disposals should extend the illustrative model accordingly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Parameters:
    """Illustrative biological, operational and financial assumptions."""

    horizon_days: int = 3650
    time_step_days: int = 1
    cycle_days: int = 120
    sowing_interval_days: int = 30
    initial_biomass_kg: float = 50.0
    carrying_capacity_kg: float = 500.0
    gompertz_rate_per_day: float = 0.025
    mortality_fraction: float = 0.05
    biomass_yield_multiplier: float = 1.0
    sale_price_per_kg: float = 1.20
    sowing_cost_per_cohort: float = 100.0
    harvest_cost_per_kg: float = 0.20
    daily_fixed_cost: float = 8.0
    initial_investment: float = 12000.0
    replacement_interval_days: int = 1825
    replacement_fraction: float = 0.35
    asset_depreciation_life_days: int = 1825
    residual_value_fraction: float = 0.10
    annual_discount_rate: float = 0.08
    tax_rate: float = 0.0


def _validate_parameters(p: Parameters) -> None:
    """Reject parameter combinations that would silently miss dated events."""
    positive_integer_fields = (
        "horizon_days",
        "time_step_days",
        "cycle_days",
        "sowing_interval_days",
        "asset_depreciation_life_days",
    )
    for name in positive_integer_fields:
        value = getattr(p, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if p.replacement_interval_days < 0:
        raise ValueError("replacement_interval_days must be zero or a positive integer.")
    if p.horizon_days % p.time_step_days != 0:
        raise ValueError("horizon_days must be divisible by time_step_days.")

    aligned_fields = (
        "cycle_days",
        "sowing_interval_days",
        "asset_depreciation_life_days",
    )
    if p.replacement_interval_days:
        aligned_fields += ("replacement_interval_days",)
    for name in aligned_fields:
        if getattr(p, name) % p.time_step_days != 0:
            raise ValueError(f"{name} must be divisible by time_step_days.")

    if not 0.0 <= p.mortality_fraction <= 1.0:
        raise ValueError("mortality_fraction must be between zero and one.")
    if not 0.0 <= p.tax_rate <= 1.0:
        raise ValueError("tax_rate must be between zero and one.")
    if p.biomass_yield_multiplier < 0.0:
        raise ValueError("biomass_yield_multiplier cannot be negative.")
    if p.replacement_fraction < 0.0 or p.residual_value_fraction < 0.0:
        raise ValueError("replacement and residual fractions cannot be negative.")


def gompertz_biomass(age_days: np.ndarray, p: Parameters) -> np.ndarray:
    """Return cohort biomass at age, including mortality and yield scaling."""
    age = np.maximum(age_days, 0.0)
    ratio = max(p.carrying_capacity_kg / p.initial_biomass_kg, 1.000001)
    displacement = math.log(math.log(ratio))
    biomass = p.carrying_capacity_kg * np.exp(
        -np.exp(displacement - p.gompertz_rate_per_day * age)
    )
    return biomass * (1.0 - p.mortality_fraction) * p.biomass_yield_multiplier


def cohort_schedule(p: Parameters) -> pd.DataFrame:
    """Return only sowing events whose harvest occurs no later than the horizon."""
    _validate_parameters(p)
    latest_sowing_day = p.horizon_days - p.cycle_days
    if latest_sowing_day < 0:
        return pd.DataFrame(
            {
                "sowing_day": pd.Series(dtype="int64"),
                "harvest_day": pd.Series(dtype="int64"),
            }
        )

    sowing_days = np.arange(
        0,
        latest_sowing_day + p.time_step_days,
        p.sowing_interval_days,
        dtype=int,
    )
    sowing_days = sowing_days[sowing_days <= latest_sowing_day]
    schedule = pd.DataFrame({"sowing_day": sowing_days})
    schedule["harvest_day"] = schedule["sowing_day"] + p.cycle_days
    return schedule


def replacement_days(p: Parameters) -> np.ndarray:
    """Return replacement dates strictly inside the project horizon."""
    _validate_parameters(p)
    if p.replacement_interval_days == 0:
        return np.array([], dtype=int)
    return np.arange(
        p.replacement_interval_days,
        p.horizon_days,
        p.replacement_interval_days,
        dtype=int,
    )


def simulate(p: Parameters) -> pd.DataFrame:
    """Simulate cash flows and return one row per numerical time step."""
    _validate_parameters(p)
    days = np.arange(0, p.horizon_days + p.time_step_days, p.time_step_days)
    day_index = {int(day): i for i, day in enumerate(days)}
    schedule = cohort_schedule(p)
    scheduled_replacements = replacement_days(p)

    df = pd.DataFrame({"day": days})
    df["sowing_cohorts"] = 0
    df["harvest_cohorts"] = 0
    df["active_cohorts"] = 0
    df["biomass_in_cultivation_kg"] = 0.0
    df["harvest_kg"] = 0.0
    df["revenue"] = 0.0
    df["fixed_cost"] = p.daily_fixed_cost * p.time_step_days
    df["sowing_cost"] = 0.0
    df["harvest_cost"] = 0.0
    df["operating_cost"] = 0.0
    df["investment"] = 0.0
    df["reinvestment"] = 0.0
    df["depreciation"] = 0.0
    df["residual_value"] = 0.0

    df.loc[df.index[0], "investment"] = p.initial_investment

    for event in schedule.itertuples(index=False):
        age = days - event.sowing_day
        active = (age >= 0) & (age < p.cycle_days)
        df.loc[active, "active_cohorts"] += 1
        df.loc[active, "biomass_in_cultivation_kg"] += gompertz_biomass(age[active], p)
        df.loc[day_index[event.sowing_day], "sowing_cohorts"] += 1
        df.loc[day_index[event.sowing_day], "sowing_cost"] += p.sowing_cost_per_cohort

        harvested_biomass = float(gompertz_biomass(np.array([p.cycle_days]), p)[0])
        harvest_row = day_index[event.harvest_day]
        df.loc[harvest_row, "harvest_cohorts"] += 1
        df.loc[harvest_row, "harvest_kg"] += harvested_biomass
        df.loc[harvest_row, "revenue"] += harvested_biomass * p.sale_price_per_kg
        df.loc[harvest_row, "harvest_cost"] += harvested_biomass * p.harvest_cost_per_kg

    replacement_amount = p.initial_investment * p.replacement_fraction
    for day in scheduled_replacements:
        df.loc[day_index[int(day)], "reinvestment"] += replacement_amount

    capital_purchases = [(0, p.initial_investment)]
    capital_purchases.extend((int(day), replacement_amount) for day in scheduled_replacements)
    depreciation_per_step_divisor = p.asset_depreciation_life_days / p.time_step_days
    for purchase_day, amount in capital_purchases:
        age = days - purchase_day
        depreciable = (age >= 0) & (age < p.asset_depreciation_life_days)
        df.loc[depreciable, "depreciation"] += amount / depreciation_per_step_divisor

    df.loc[df.index[-1], "residual_value"] = p.initial_investment * p.residual_value_fraction
    df["operating_cost"] = df["fixed_cost"] + df["sowing_cost"] + df["harvest_cost"]
    df["taxable_income"] = df["revenue"] - df["operating_cost"] - df["depreciation"]
    df["tax"] = np.maximum(df["taxable_income"], 0.0) * p.tax_rate
    df["free_cash_flow"] = (
        df["revenue"]
        - df["operating_cost"]
        - df["tax"]
        - df["investment"]
        - df["reinvestment"]
        + df["residual_value"]
    )

    continuous_rate = math.log1p(p.annual_discount_rate) / 365.0
    df["discount_factor"] = np.exp(-continuous_rate * df["day"])
    df["discounted_fcf"] = df["free_cash_flow"] * df["discount_factor"]
    df["cumulative_npv"] = df["discounted_fcf"].cumsum()
    return df


def _npv_affine_coefficients(base: Parameters) -> tuple[float, float, float]:
    """Return fixed NPV, discounted harvest cost, and discounted revenue.

    With yield multiplier ``b`` and price multiplier ``p``, the default model
    is affine: ``NPV(b, p) = fixed + b * (p * revenue - harvest_cost)``.
    This identity creates an exact threshold frontier without interpolation.
    """
    fixed = float(
        simulate(replace(base, biomass_yield_multiplier=0.0, sale_price_per_kg=0.0))
        ["cumulative_npv"].iloc[-1]
    )
    zero_price = float(
        simulate(replace(base, biomass_yield_multiplier=1.0, sale_price_per_kg=0.0))
        ["cumulative_npv"].iloc[-1]
    )
    unit_price = float(
        simulate(replace(base, biomass_yield_multiplier=1.0))
        ["cumulative_npv"].iloc[-1]
    )
    return fixed, fixed - zero_price, unit_price - zero_price


def _cashflow_template(base: Parameters) -> dict[str, np.ndarray]:
    """Precompute arrays needed to evaluate price-yield scenarios quickly.

    Yield affects only harvestable biomass, harvest revenue and harvest cost in
    this illustrative model. This template preserves the full daily tax logic,
    including the non-linearity introduced by a zero tax floor.
    """
    zero_yield = simulate(replace(base, biomass_yield_multiplier=0.0))
    unit_yield = simulate(replace(base, biomass_yield_multiplier=1.0))
    return {
        "revenue_at_base_price": unit_yield["revenue"].to_numpy(),
        "harvest_cost_at_unit_yield": unit_yield["harvest_cost"].to_numpy(),
        "non_yield_operating_cost": zero_yield["operating_cost"].to_numpy(),
        "depreciation": zero_yield["depreciation"].to_numpy(),
        "investment": zero_yield["investment"].to_numpy(),
        "reinvestment": zero_yield["reinvestment"].to_numpy(),
        "residual_value": zero_yield["residual_value"].to_numpy(),
        "discount_factor": zero_yield["discount_factor"].to_numpy(),
    }


def _final_npv_from_multipliers(
    template: dict[str, np.ndarray],
    base: Parameters,
    biomass_yield_multiplier: float,
    price_multiplier: float,
) -> float:
    """Evaluate the final NPV for a price-yield combination from a template."""
    revenue = (
        biomass_yield_multiplier
        * price_multiplier
        * template["revenue_at_base_price"]
    )
    operating_cost = (
        template["non_yield_operating_cost"]
        + biomass_yield_multiplier * template["harvest_cost_at_unit_yield"]
    )
    taxable_income = revenue - operating_cost - template["depreciation"]
    tax = np.maximum(taxable_income, 0.0) * base.tax_rate
    free_cash_flow = (
        revenue
        - operating_cost
        - tax
        - template["investment"]
        - template["reinvestment"]
        + template["residual_value"]
    )
    return float(np.sum(free_cash_flow * template["discount_factor"]))


def _numeric_break_even_price(
    template: dict[str, np.ndarray],
    base: Parameters,
    biomass_yield_multiplier: float,
) -> float:
    """Find the zero-NPV price multiplier while retaining nonlinear tax logic."""
    if biomass_yield_multiplier <= 0.0:
        return math.inf

    lower, upper = 0.0, 1.0
    while _final_npv_from_multipliers(template, base, biomass_yield_multiplier, upper) < 0.0:
        upper *= 2.0
        if upper > 1_000_000.0:
            raise RuntimeError("No break-even price was found within the search range.")

    for _ in range(50):
        midpoint = (lower + upper) / 2.0
        if _final_npv_from_multipliers(template, base, biomass_yield_multiplier, midpoint) >= 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def threshold_frontier(base: Parameters, biomass_multipliers: np.ndarray) -> pd.DataFrame:
    """Calculate the zero-NPV sale-price multiplier for each yield level."""
    template = _cashflow_template(base)
    if base.tax_rate == 0.0:
        fixed, discounted_harvest_cost, discounted_revenue = _npv_affine_coefficients(base)
    rows = []
    for multiplier in biomass_multipliers:
        b = float(multiplier)
        if base.tax_rate == 0.0:
            if b <= 0.0:
                break_even = math.inf
            else:
                break_even = (discounted_harvest_cost - fixed / b) / discounted_revenue
        else:
            break_even = _numeric_break_even_price(template, base, b)
        rows.append(
            {
                "biomass_yield_multiplier": b,
                "break_even_price_multiplier": break_even,
                "break_even_sale_price_per_kg": break_even * base.sale_price_per_kg,
                "final_npv_at_base_price": _final_npv_from_multipliers(template, base, b, 1.0),
            }
        )
    return pd.DataFrame(rows)


def threshold_grid(
    base: Parameters,
    biomass_multipliers: np.ndarray,
    price_multipliers: np.ndarray,
) -> pd.DataFrame:
    """Create a final-NPV grid for reporting or visualization."""
    template = _cashflow_template(base)
    rows = []
    for biomass_multiplier in biomass_multipliers:
        for price_multiplier in price_multipliers:
            final_npv = _final_npv_from_multipliers(
                template,
                base,
                float(biomass_multiplier),
                float(price_multiplier),
            )
            rows.append(
                {
                    "biomass_yield_multiplier": float(biomass_multiplier),
                    "price_multiplier": float(price_multiplier),
                    "final_npv": final_npv,
                    "feasible": bool(final_npv >= 0.0),
                }
            )
    return pd.DataFrame(rows)


def _scenario_parameters(base: Parameters) -> dict[str, Parameters]:
    return {
        "Baseline": base,
        "Conservative": replace(
            base,
            biomass_yield_multiplier=0.85,
            sale_price_per_kg=base.sale_price_per_kg * 0.90,
        ),
        "Favorable": replace(
            base,
            biomass_yield_multiplier=1.20,
            sale_price_per_kg=base.sale_price_per_kg * 1.15,
        ),
    }


def _plot_baseline_npv(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["day"], df["cumulative_npv"], color="black", linewidth=1.7)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative NPV (USD)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_figure_4(
    scenarios: dict[str, pd.DataFrame],
    frontier: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    styles = {
        "Baseline": {"color": "black", "linestyle": "-"},
        "Conservative": {"color": "black", "linestyle": "--"},
        "Favorable": {"color": "black", "linestyle": ":"},
    }
    for name, df in scenarios.items():
        axes[0].plot(
            df["day"],
            df["cumulative_npv"] / 1000.0,
            label=name,
            linewidth=1.8,
            **styles[name],
        )
    axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_title("A. Cumulative NPV under alternative scenarios")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Cumulative NPV (thousand USD)")
    axes[0].legend(frameon=False, loc="best")

    x = frontier["biomass_yield_multiplier"].to_numpy()
    y = frontier["break_even_price_multiplier"].to_numpy()
    scenario_points = {
        "Baseline": (1.00, 1.00, "o"),
        "Conservative": (0.85, 0.90, "s"),
        "Favorable": (1.20, 1.15, "^"),
    }
    lower = max(0.0, min(0.5, float(np.nanmin(y)) - 0.05))
    upper = max(float(np.nanmax(y)) + 0.10, 1.25)
    axes[1].fill_between(x, lower, y, color="0.85")
    axes[1].fill_between(x, y, upper, color="0.96")
    axes[1].plot(x, y, color="black", linewidth=1.8)
    axes[1].text(x[0] + 0.08, lower + 0.18, "Not feasible")
    axes[1].text(x[-1] - 0.30, upper - 0.16, "Feasible")
    for name, (bio, price, marker) in scenario_points.items():
        axes[1].plot(bio, price, marker=marker, color="black", markersize=5)
        axes[1].annotate(name, (bio, price), xytext=(5, 3), textcoords="offset points")
    axes[1].set_ylim(lower, upper)
    axes[1].set_title("B. Break-even price-yield frontier")
    axes[1].set_xlabel("Biomass yield multiplier")
    axes[1].set_ylabel("Sale-price multiplier")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_outputs(base: Parameters, output_dir: Path) -> dict[str, float | int]:
    """Write reproducible baseline, scenario and threshold outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = simulate(base)
    scenarios = {name: simulate(p) for name, p in _scenario_parameters(base).items()}
    frontier = threshold_frontier(base, np.linspace(0.6, 1.6, 101))
    grid = threshold_grid(base, np.linspace(0.6, 1.6, 11), np.linspace(0.6, 1.8, 13))

    baseline.to_csv(output_dir / "baseline_timeseries.csv", index=False)
    cohort_schedule(base).to_csv(output_dir / "cohort_schedule.csv", index=False)
    frontier.to_csv(output_dir / "price_productivity_frontier.csv", index=False)
    grid.to_csv(output_dir / "price_productivity_grid.csv", index=False)
    with (output_dir / "parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(base), handle, indent=2)

    _plot_baseline_npv(baseline, output_dir / "cumulative_npv.png")
    _plot_figure_4(scenarios, frontier, output_dir / "figure_4_sensitivity_thresholds.png")

    summary = {
        "final_npv_usd": float(baseline["cumulative_npv"].iloc[-1]),
        "maximum_active_cohorts": int(baseline["active_cohorts"].max()),
        "scheduled_cohorts": int(len(cohort_schedule(base))),
        "terminal_reinvestment_count": int(
            (baseline.loc[baseline["day"] == base.horizon_days, "reinvestment"] > 0).sum()
        ),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    base = Parameters()
    summary = save_outputs(base, Path("outputs"))
    print(f"Final NPV: {summary['final_npv_usd']:,.2f}")
    print(f"Maximum active cohorts: {summary['maximum_active_cohorts']}")
    print(f"Scheduled cohorts: {summary['scheduled_cohorts']}")


if __name__ == "__main__":
    main()
