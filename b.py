# app.py
# Simulator 361 Demo: User IRR + Bayesian Optimization Best Case + Trial-by-trial log
# Run:
#   pip install streamlit optuna numpy pandas
#   streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st

import optuna


# -----------------------------
# IRR helper (bisection on NPV)
# -----------------------------
def irr_from_cashflows(cashflows):
    cf = np.array(cashflows, dtype=float)

    # Must have at least one negative and one positive cashflow
    if not (np.any(cf < 0) and np.any(cf > 0)):
        return None

    def npv(rate):
        return float(np.sum(cf / ((1.0 + rate) ** np.arange(len(cf)))))

    lo, hi = -0.99, 5.0  # -99% to +500%
    f_lo, f_hi = npv(lo), npv(hi)

    # Try expanding hi if no sign change
    if f_lo * f_hi > 0:
        for hi2 in [10.0, 20.0, 50.0]:
            f_hi2 = npv(hi2)
            if f_lo * f_hi2 <= 0:
                hi, f_hi = hi2, f_hi2
                break
        else:
            return None

    for _ in range(120):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return (lo + hi) / 2.0


# -----------------------------
# Demo simulator (6-year YoY)
# -----------------------------
def simulate_demo_case(
    saleable_area_sqft,
    years_total,                 # fixed = 6
    construction_years,          # user input
    land_cost,
    approval_cost,
    res_rate,
    sales_velocity_value_per_year,  # ₹/year
    fsi_utilization_factor,
    construction_rate,
    admin_pct_of_construction,
    marketing_pct_of_revenue,
):
    years_total = int(years_total)
    construction_years = int(construction_years)
    construction_years = max(1, min(construction_years, years_total + 1))

    # Revenue (total) scales with FSI utilization factor
    total_revenue = float(saleable_area_sqft) * float(fsi_utilization_factor) * float(res_rate)

    # Linear revenue phasing by velocity (₹/year), force closure by taking leftover in final year
    rev = np.zeros(years_total + 1)  # Year 0..6 (Year 0 revenue is typically 0)
    remaining = total_revenue
    for y in range(1, years_total + 1):
        take = min(remaining, float(sales_velocity_value_per_year))
        rev[y] = take
        remaining -= take
        if remaining <= 1e-6:
            break

    # If not fully sold by year 6, take remaining as terminal receipt in year 6
    if remaining > 1e-6:
        rev[years_total] += remaining
        remaining = 0.0

    # Costs
    cost = np.zeros(years_total + 1)

    # Year 0 upfront
    cost[0] += float(land_cost) + float(approval_cost)

    # Construction cost scales with FSI utilization factor (your requirement: revenue + certain costs scale)
    total_construction_cost = float(saleable_area_sqft) * float(fsi_utilization_factor) * float(construction_rate)

    # Spread construction across construction years starting Year 0
    per_year_construction = total_construction_cost / float(construction_years)
    for y in range(0, construction_years):
        cost[y] += per_year_construction

    # Admin spread evenly across all years (0..6) as % of total construction
    total_admin = float(admin_pct_of_construction) * total_construction_cost
    per_year_admin = total_admin / float(years_total + 1)
    cost += per_year_admin

    # Marketing tied to revenue
    marketing = float(marketing_pct_of_revenue) * rev
    cost += marketing

    # Cashflow and IRR
    cashflow = rev - cost
    irr = irr_from_cashflows(cashflow.tolist())

    total_cost = float(np.sum(cost))
    profit = float(total_revenue - total_cost)

    df = pd.DataFrame({
        "Year": list(range(0, years_total + 1)),
        "Revenue": rev,
        "Cost": cost,
        "Cashflow": cashflow,
    })

    return {
        "irr": irr,
        "profit": profit,
        "total_revenue": float(total_revenue),
        "total_cost": total_cost,
        "yoy_table": df,
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Simulator 361 Demo – Bayesian Optimization", layout="wide")
st.title("Simulator 361 – Best Case Scenario Demo (Bayesian Optimization)")

YEARS_TOTAL = 6  # fixed per your requirement

left, right = st.columns([1.05, 1])

with left:
    st.subheader("Fixed Inputs (User Scenario)")
    saleable_area_sqft = st.number_input("Total Saleable Area (sq ft)", min_value=1000.0, value=200000.0, step=1000.0)
    construction_years = st.number_input("Construction Years (1–6)", min_value=1, max_value=6, value=3, step=1)

    land_cost = st.number_input("Land Cost (₹)", min_value=0.0, value=25_00_00_000.0, step=10_00_000.0)
    approval_cost = st.number_input("Approval Cost (₹)", min_value=0.0, value=2_50_00_000.0, step=5_00_000.0)

with right:
    st.subheader("User Scenario Inputs (Base)")
    res_rate_user = st.number_input("Residential Rate (₹/sq ft)", min_value=1000.0, value=9000.0, step=100.0)
    sales_vel_user = st.number_input("Sales Velocity (₹/year)", min_value=1_00_00_000.0, value=20_00_00_000.0, step=50_00_000.0)

    fsi_factor_user = st.number_input("FSI Utilization Factor (multiplier)", min_value=0.50, value=1.00, step=0.01)
    constr_rate_user = st.number_input("Construction Rate (₹/sq ft)", min_value=500.0, value=2500.0, step=50.0)

st.divider()

# Advanced assumptions (not optimized)
with st.expander("Assumptions (Kept Simple for Demo)", expanded=False):
    admin_pct = st.slider("Admin Cost (% of construction)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)
    marketing_pct = st.slider("Marketing Cost (% of revenue)", min_value=0.0, max_value=0.12, value=0.06, step=0.005)

# Bounds container (client explainable)
st.subheader("Optimization Bounds (Editable for Demo Explanation)")
with st.expander("Edit permitted ranges for Bayesian Optimization", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        res_min = st.number_input("Residential Rate MIN (₹/sq ft)", min_value=1000.0, value=7500.0, step=100.0)
        res_max = st.number_input("Residential Rate MAX (₹/sq ft)", min_value=1000.0, value=10500.0, step=100.0)

        vel_min = st.number_input("Sales Velocity MIN (₹/year)", min_value=1_00_00_000.0, value=12_00_00_000.0, step=50_00_000.0)
        vel_max = st.number_input("Sales Velocity MAX (₹/year)", min_value=1_00_00_000.0, value=30_00_00_000.0, step=50_00_000.0)

    with c2:
        fsi_min = st.number_input("FSI Utilization MIN", min_value=0.50, value=0.90, step=0.01)
        fsi_max = st.number_input("FSI Utilization MAX", min_value=0.50, value=1.20, step=0.01)

        constr_min = st.number_input("Construction Rate MIN (₹/sq ft)", min_value=500.0, value=2000.0, step=50.0)
        constr_max = st.number_input("Construction Rate MAX (₹/sq ft)", min_value=500.0, value=3200.0, step=50.0)

st.divider()

# Trials setup: I’m choosing a demo-friendly default and max for your specific case (4 variables, client meeting)
# Default 25 gives visible learning; max 80 keeps it fast but meaningful.
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    n_trials = st.number_input("Trials (N) – default 25, max 80", min_value=5, max_value=80, value=25, step=5)
with col_t2:
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
with col_t3:
    show_live = st.checkbox("Show live trial updates", value=True)

run_btn = st.button("Run User IRR + Optimize Best Case")

if run_btn:
    # 1) User scenario result
    user_out = simulate_demo_case(
        saleable_area_sqft=saleable_area_sqft,
        years_total=YEARS_TOTAL,
        construction_years=construction_years,
        land_cost=land_cost,
        approval_cost=approval_cost,
        res_rate=res_rate_user,
        sales_velocity_value_per_year=sales_vel_user,
        fsi_utilization_factor=fsi_factor_user,
        construction_rate=constr_rate_user,
        admin_pct_of_construction=admin_pct,
        marketing_pct_of_revenue=marketing_pct,
    )

    st.markdown("## User Scenario Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue (₹)", f"{user_out['total_revenue']:,.0f}")
    m2.metric("Total Cost (₹)", f"{user_out['total_cost']:,.0f}")
    m3.metric("Profit (₹)", f"{user_out['profit']:,.0f}")
    m4.metric("IRR", "N/A" if user_out["irr"] is None else f"{user_out['irr']*100:.2f}%")

    st.dataframe(user_out["yoy_table"], use_container_width=True)
    st.line_chart(user_out["yoy_table"].set_index("Year")[["Revenue", "Cost", "Cashflow"]])

    # 2) Optimization
    st.markdown("## Bayesian Optimization (Trial-by-Trial)")
    st.caption("Each trial proposes values inside your permitted bounds, runs the 6-year YoY simulator, computes IRR, and learns for the next trial.")

    trial_placeholder = st.empty()
    best_placeholder = st.empty()

    trials_log = []

    def objective(trial: optuna.trial.Trial):
        res_rate = trial.suggest_float("res_rate", res_min, res_max)
        sales_vel = trial.suggest_float("sales_velocity_value_per_year", vel_min, vel_max)
        fsi_factor = trial.suggest_float("fsi_utilization_factor", fsi_min, fsi_max)
        constr_rate = trial.suggest_float("construction_rate", constr_min, constr_max)

        out = simulate_demo_case(
            saleable_area_sqft=saleable_area_sqft,
            years_total=YEARS_TOTAL,
            construction_years=construction_years,
            land_cost=land_cost,
            approval_cost=approval_cost,
            res_rate=res_rate,
            sales_velocity_value_per_year=sales_vel,
            fsi_utilization_factor=fsi_factor,
            construction_rate=constr_rate,
            admin_pct_of_construction=admin_pct,
            marketing_pct_of_revenue=marketing_pct,
        )

        irr = out["irr"]
        if irr is None:
            irr = -1e18  # invalid IRR gets penalized heavily

        # Store for live display
        trials_log.append({
            "Trial #": trial.number,
            "Residential Rate": res_rate,
            "Sales Velocity (₹/year)": sales_vel,
            "FSI Utilization": fsi_factor,
            "Construction Rate": constr_rate,
            "IRR (%)": (irr * 100.0) if irr > -1e10 else np.nan,
            "Profit (₹)": out["profit"],
            "Total Revenue (₹)": out["total_revenue"],
            "Total Cost (₹)": out["total_cost"],
        })

        if show_live:
            df_live = pd.DataFrame(trials_log).sort_values("Trial #")
            # Highlight best known so far
            if df_live["IRR (%)"].notna().any():
                best_idx = df_live["IRR (%)"].idxmax()
                styled = df_live.style.apply(
                    lambda r: ["background-color: #d7f5d7; color: black"] * len(r) if r.name == best_idx else [""] * len(r),
                    axis=1
                )
                trial_placeholder.dataframe(styled, use_container_width=True)
                best_placeholder.info(f"Current Best Trial: {int(df_live.loc[best_idx, 'Trial #'])} | IRR = {df_live.loc[best_idx, 'IRR (%)']:.2f}%")
            else:
                trial_placeholder.dataframe(df_live, use_container_width=True)

        return irr

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    trials_df = pd.DataFrame(trials_log).sort_values("Trial #")

    # Mark best trial explicitly
    best_trial = study.best_trial.number
    best_row = trials_df.loc[trials_df["Trial #"] == best_trial].iloc[0]

    st.markdown("### Trial Log (Final)")
    def highlight_best(row):
        return ["background-color: #d7f5d7; color: black"] * len(row) if row["Trial #"] == best_trial else [""] * len(row)

    st.dataframe(trials_df.style.apply(highlight_best, axis=1), use_container_width=True)

    st.markdown("## Best Case Scenario (Selected)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Best Trial #", f"{best_trial}")
    b2.metric("Best IRR", f"{best_row['IRR (%)']:.2f}%")
    b3.metric("Best Profit (₹)", f"{best_row['Profit (₹)']:,.0f}")
    b4.metric("Best Revenue (₹)", f"{best_row['Total Revenue (₹)']:,.0f}")

    st.markdown("### Best Case Variable Values")
    st.dataframe(pd.DataFrame([{
        "Residential Rate": best_row["Residential Rate"],
        "Sales Velocity (₹/year)": best_row["Sales Velocity (₹/year)"],
        "FSI Utilization": best_row["FSI Utilization"],
        "Construction Rate": best_row["Construction Rate"],
    }]), use_container_width=True)

    # Re-simulate best case to show YoY tables/charts
    best_out = simulate_demo_case(
        saleable_area_sqft=saleable_area_sqft,
        years_total=YEARS_TOTAL,
        construction_years=construction_years,
        land_cost=land_cost,
        approval_cost=approval_cost,
        res_rate=float(best_row["Residential Rate"]),
        sales_velocity_value_per_year=float(best_row["Sales Velocity (₹/year)"]),
        fsi_utilization_factor=float(best_row["FSI Utilization"]),
        construction_rate=float(best_row["Construction Rate"]),
        admin_pct_of_construction=admin_pct,
        marketing_pct_of_revenue=marketing_pct,
    )

    st.markdown("### Best Case YoY Table")
    st.dataframe(best_out["yoy_table"], use_container_width=True)
    st.line_chart(best_out["yoy_table"].set_index("Year")[["Revenue", "Cost", "Cashflow"]])

    st.markdown("## Compare: User vs Best Case")
    compare_df = pd.DataFrame([
        {"Metric": "IRR (%)",
         "User": None if user_out["irr"] is None else user_out["irr"] * 100.0,
         "Best": None if best_out["irr"] is None else best_out["irr"] * 100.0,
         "Delta": None if (user_out["irr"] is None or best_out["irr"] is None) else (best_out["irr"] - user_out["irr"]) * 100.0},
        {"Metric": "Total Revenue (₹)",
         "User": user_out["total_revenue"],
         "Best": best_out["total_revenue"],
         "Delta": best_out["total_revenue"] - user_out["total_revenue"]},
        {"Metric": "Total Cost (₹)",
         "User": user_out["total_cost"],
         "Best": best_out["total_cost"],
         "Delta": best_out["total_cost"] - user_out["total_cost"]},
        {"Metric": "Profit (₹)",
         "User": user_out["profit"],
         "Best": best_out["profit"],
         "Delta": best_out["profit"] - user_out["profit"]},
    ])
    st.dataframe(compare_df, use_container_width=True)

    st.markdown("### Key Differences (What changed to improve IRR)")
    st.write(
        f"Residential Rate moved from ₹{res_rate_user:,.0f} to ₹{best_row['Residential Rate']:,.0f} per sq ft. "
        f"Sales Velocity moved from ₹{sales_vel_user:,.0f}/year to ₹{best_row['Sales Velocity (₹/year)']:,.0f}/year. "
        f"FSI Utilization moved from {fsi_factor_user:.2f} to {best_row['FSI Utilization']:.2f}. "
        f"Construction Rate moved from ₹{constr_rate_user:,.0f} to ₹{best_row['Construction Rate']:,.0f} per sq ft."
    )

st.caption("Note: This is a demo model to show optimization behavior. You can later replace the internal simulator logic with your full feasibility formulas without changing the Bayesian loop or UI.")
