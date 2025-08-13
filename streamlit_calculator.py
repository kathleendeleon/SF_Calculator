# Scenario-based Snowflake Cost Estimator for Project Managers
# - CSV inputs: credits.csv (Edition -> $/credit), warehouse_details.csv (valid sizes)
# - Provides Scenarios, Workload Templates, Assumptions, Results/Comparison, and Export

from __future__ import annotations
import io
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# ============================
# ---------- Data Layer ------
# ============================

@st.cache_data(show_spinner=False)
def load_reference_data():
    # Required CSVs (kept from the original app)
    credits_df = pd.read_csv("credits.csv")  # expects columns: Edition, Credit
    warehouse_df = pd.read_csv("warehouse_details.csv")  # expects column: SIZE

    # Validate minimal required columns
    if not set(["Edition", "Credit"]).issubset(credits_df.columns):
        st.error("credits.csv must contain columns: Edition, Credit")
    if "SIZE" not in warehouse_df.columns:
        st.error("warehouse_details.csv must contain a column: SIZE")

    # Build maps
    credit_price_by_edition = (
        credits_df.dropna(subset=["Edition", "Credit"])  # clean
        .groupby("Edition")["Credit"]
        .max()  # if multiple rows per edition, take max/most conservative
        .to_dict()
    )

    sizes = (
        warehouse_df["SIZE"].dropna().astype(str).drop_duplicates().tolist()
    )

    # Default provider/regions kept (extend as needed)
    providers_regions = [
        ["AWS", "US East (Commercial Gov-N.VA)"],
        ["AWS", "US Gov West 1"],
        ["AWS", "US Gov West 1 (Fedramp High Plus)"],
        ["Azure", "East US 2 (Virginia)"],
        ["Azure", "South Central US (Texas)"],
        ["Azure", "US Central (Iowa)"],
        ["GCP", "US Central 1 (Iowa)"],
        ["AWS", "US East (Northern Virginia)"],
        ["AWS", "US East (Ohio)"],
        ["AWS", "US East 1 Commercial Gov"],
        ["GCP", "US East 4 (N. Virginia)"],
        ["AWS", "US West (Oregon)"],
        ["Azure", "US Gov Virginia"],
        ["Azure", "West US 2 (Washington)"],
    ]
    pr_df = pd.DataFrame(providers_regions, columns=["Provider", "Region"])  # for UI

    return credit_price_by_edition, sizes, pr_df


# ============================
# ---------- Model ----------
# ============================

SIZE_TO_CREDITS = {"XS":1, "S":2, "M":4, "L":8, "XL":16, "2XL":32, "3XL":64, "4XL":128}

@dataclass
class Assumptions:
    credit_price_by_edition: Dict[str, float]
    storage_price_per_tb: float = 40.0
    capacity_price_per_tb: float = 23.0
    cloud_services_pct_default: float = 10.0
    time_travel_storage_overhead_pct: float = 0.0
    fail_safe_storage_overhead_pct: float = 0.0
    per_second_billing: bool = True
    min_seconds_per_query: int = 60

@dataclass
class Warehouse:
    name: str
    size: str  # must be present in SIZE_TO_CREDITS or provided by warehouse_details.csv
    hours_per_day: float
    days_per_week: int
    auto_suspend_minutes: int = 5
    cloud_services_pct: Optional[float] = None
    min_clusters: int = 1
    max_clusters: int = 1
    concurrency_target: Optional[int] = None
    avg_query_seconds: Optional[int] = None
    avg_idle_minutes_per_run: int = 0

    @property
    def credits_per_hour(self) -> float:
        # Try mapping from common size table; fallback to numeric if size looks numeric
        if self.size in SIZE_TO_CREDITS:
            return float(SIZE_TO_CREDITS[self.size])
        try:
            return float(self.size)
        except Exception:
            return 0.0

@dataclass
class Scenario:
    name: str
    edition: str
    provider: str
    region: str
    tb_per_month: float
    warehouses: List[Warehouse] = field(default_factory=list)
    notes: str = ""

# ============================
# ------- Pricing Engine -----
# ============================

def estimate_avg_clusters(w: Warehouse) -> float:
    if w.max_clusters <= 1:
        return 1.0
    if w.concurrency_target:
        est = max(w.min_clusters, min(w.max_clusters, math.ceil(w.concurrency_target / 8)))
        return float(est)
    return float((w.min_clusters + w.max_clusters) / 2)


def billed_hours_per_year(w: Warehouse) -> float:
    hours_week = w.hours_per_day * w.days_per_week
    hours_year = hours_week * 52
    idle_hours = (w.avg_idle_minutes_per_run / 60.0) * w.days_per_week * 52
    return max(0.0, hours_year - idle_hours)


def compute_credits_for_warehouse(w: Warehouse) -> float:
    billable_hours = billed_hours_per_year(w)
    avg_clusters = estimate_avg_clusters(w)
    return w.credits_per_hour * billable_hours * avg_clusters


def cloud_services_credits(w: Warehouse, compute_credits: float, default_pct: float) -> float:
    pct = w.cloud_services_pct if w.cloud_services_pct is not None else default_pct
    return compute_credits * (pct / 100.0)


def storage_costs(tb_per_month: float, a: Assumptions) -> Dict[str, float]:
    raw = tb_per_month * 12.0
    overhead = raw * (a.time_travel_storage_overhead_pct + a.fail_safe_storage_overhead_pct) / 100.0
    total_tb_year = raw + overhead
    return {
        "tb_year": total_tb_year,
        "ondemand_total": total_tb_year * a.storage_price_per_tb,
        "capacity_total": total_tb_year * a.capacity_price_per_tb,
    }


def scenario_totals(s: Scenario, a: Assumptions) -> Dict[str, float]:
    compute_total = 0.0
    cloud_services_total = 0.0
    for w in s.warehouses:
        comp = compute_credits_for_warehouse(w)
        cs = cloud_services_credits(w, comp, a.cloud_services_pct_default)
        compute_total += comp
        cloud_services_total += cs
    credits_total = compute_total + cloud_services_total
    credit_price = a.credit_price_by_edition.get(s.edition, 0.0)
    storage = storage_costs(s.tb_per_month, a)
    subtotal_compute_usd = credits_total * credit_price
    grand_total_ondemand = storage["ondemand_total"] + subtotal_compute_usd
    grand_total_capacity = storage["capacity_total"] + subtotal_compute_usd
    return {
        "compute_credits": compute_total,
        "cloud_services_credits": cloud_services_total,
        "credits_total": credits_total,
        "credit_unit_price": credit_price,
        "compute_usd": subtotal_compute_usd,
        "storage_tb_year": storage["tb_year"],
        "storage_usd_ondemand": storage["ondemand_total"],
        "storage_usd_capacity": storage["capacity_total"],
        "total_usd_ondemand": grand_total_ondemand,
        "total_usd_capacity": grand_total_capacity,
        "monthly_usd_ondemand": grand_total_ondemand / 12.0,
        "monthly_usd_capacity": grand_total_capacity / 12.0,
    }


# ============================
# ----------- UI -------------
# ============================

st.set_page_config(page_title="Snowflake Cost Estimator (PM)", layout="wide")
st.title("Snowflake Cost Estimator — Scenario Planner")
st.caption("For planning purposes only. Actual pricing may vary. Uses local credits.csv & warehouse_details.csv.")

credit_price_by_edition, valid_sizes, pr_df = load_reference_data()

# Initialize app state
if "assumptions" not in st.session_state:
    st.session_state.assumptions = Assumptions(credit_price_by_edition=credit_price_by_edition)
if "scenarios" not in st.session_state:
    st.session_state.scenarios: Dict[str, Scenario] = {}
if "active_scenario" not in st.session_state:
    st.session_state.active_scenario: Optional[str] = None


# ---------- Sidebar: Build/Select Scenario ----------
st.sidebar.header("Scenario Builder")
with st.sidebar:
    cols = st.columns(2)
    with cols[0]:
        s_name = st.text_input("Scenario name", value="Prod")
    with cols[1]:
        tbpm = st.number_input("TB / month", min_value=0.0, value=1.0, step=0.5)

    col2 = st.columns(2)
    with col2[0]:
        edition = st.selectbox("Edition", options=sorted(credit_price_by_edition.keys()))
    with col2[1]:
        provider = st.selectbox("Provider", options=sorted(pr_df["Provider"].unique()))

    region = st.selectbox(
        "Region",
        options=sorted(pr_df[pr_df["Provider"] == provider]["Region"].unique()),
    )

    add_or_update = st.button("Add/Update Scenario")
    if add_or_update and s_name:
        st.session_state.scenarios[s_name] = Scenario(
            name=s_name,
            edition=edition,
            provider=provider,
            region=region,
            tb_per_month=tbpm,
            warehouses=st.session_state.scenarios.get(s_name, Scenario(s_name, edition, provider, region, tbpm)).warehouses,
        )
        st.session_state.active_scenario = s_name

    if st.session_state.scenarios:
        st.session_state.active_scenario = st.selectbox(
            "Active scenario",
            options=list(st.session_state.scenarios.keys()),
            index=list(st.session_state.scenarios.keys()).index(st.session_state.active_scenario)
            if st.session_state.active_scenario in st.session_state.scenarios
            else 0,
        )

    st.divider()

    st.subheader("Add Warehouse to Scenario")
    w_cols = st.columns(2)
    with w_cols[0]:
        w_name = st.text_input("Warehouse name", value="WH_ETL")
        size = st.selectbox("Size", options=valid_sizes)
        days_per_week = st.number_input("Days/week", 1, 7, value=5)
        min_clusters = st.number_input("Min clusters", 1, 99, value=1)
        avg_idle = st.number_input("Avg idle minutes/run", 0, 240, value=0)
    with w_cols[1]:
        hours_per_day = st.number_input("Hours/day", 0.0, 24.0, value=8.0, step=0.5)
        auto_suspend = st.number_input("Auto-suspend (min)", 0, 60, value=5)
        cloud_pct = st.number_input("Cloud Services % (override)", 0, 100, value=10)
        max_clusters = st.number_input("Max clusters", 1, 99, value=1)
        concurrency = st.number_input("Concurrency target", 0, 1000, value=0)

    add_wh = st.button("Add Warehouse")
    if add_wh and st.session_state.active_scenario:
        sc = st.session_state.scenarios[st.session_state.active_scenario]
        wh = Warehouse(
            name=w_name,
            size=str(size),
            hours_per_day=float(hours_per_day),
            days_per_week=int(days_per_week),
            auto_suspend_minutes=int(auto_suspend),
            cloud_services_pct=float(cloud_pct),
            min_clusters=int(min_clusters),
            max_clusters=int(max_clusters),
            concurrency_target=int(concurrency) if concurrency > 0 else None,
            avg_idle_minutes_per_run=int(avg_idle),
        )
        # replace if same name exists
        sc.warehouses = [w for w in sc.warehouses if w.name != wh.name] + [wh]
        st.session_state.scenarios[sc.name] = sc

    if st.session_state.active_scenario and st.session_state.active_scenario in st.session_state.scenarios:
        sc = st.session_state.scenarios[st.session_state.active_scenario]
        # simple removal control
        to_remove = st.selectbox(
            "Remove warehouse",
            options=["(none)"] + [w.name for w in sc.warehouses],
            index=0,
        )
        if to_remove != "(none)" and st.button("Confirm Remove"):
            sc.warehouses = [w for w in sc.warehouses if w.name != to_remove]
            st.session_state.scenarios[sc.name] = sc

# ---------- Main Area Tabs ----------

Tabs = st.tabs(["Inputs", "Results", "Comparison", "Assumptions", "Templates", "Export", "Glossary"])

# --- Inputs Tab ---
with Tabs[0]:
    st.subheader("Scenario & Warehouses")
    if not st.session_state.scenarios:
        st.info("Create a scenario from the sidebar to begin.")
    else:
        sc = st.session_state.scenarios[st.session_state.active_scenario]
        st.markdown(f"**Active scenario:** `{sc.name}` — {sc.edition}, {sc.provider}/{sc.region}, TB/mo: {sc.tb_per_month}")
        if sc.warehouses:
            df_wh = pd.DataFrame([asdict(w) | {"credits_per_hour": w.credits_per_hour} for w in sc.warehouses])
            st.dataframe(df_wh, use_container_width=True, hide_index=True)
        else:
            st.warning("No warehouses added yet.")

# --- Results Tab ---
with Tabs[1]:
    st.subheader("Scenario Results")
    if not st.session_state.scenarios:
        st.info("Create a scenario from the sidebar to view results.")
    else:
        sc = st.session_state.scenarios[st.session_state.active_scenario]
        a = st.session_state.assumptions
        totals = scenario_totals(sc, a)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monthly On-Demand ($)", f"{totals['monthly_usd_ondemand']:,.2f}")
        m2.metric("Monthly Capacity ($)", f"{totals['monthly_usd_capacity']:,.2f}")
        m3.metric("Credits (total/yr)", f"{totals['credits_total']:,.0f}")
        m4.metric("Storage TB/yr", f"{totals['storage_tb_year']:,.2f}")

        st.divider()
        st.write("Breakdown")
        breakdown = pd.DataFrame(
            {
                "Item": [
                    "Compute credits",
                    "Cloud Services credits",
                    "Credit unit price",
                    "Compute subtotal ($)",
                    "Storage on-demand ($/yr)",
                    "Storage capacity ($/yr)",
                    "Total on-demand ($/yr)",
                    "Total capacity ($/yr)",
                ],
                "Value": [
                    totals["compute_credits"],
                    totals["cloud_services_credits"],
                    totals["credit_unit_price"],
                    totals["compute_usd"],
                    totals["storage_usd_ondemand"],
                    totals["storage_usd_capacity"],
                    totals["total_usd_ondemand"],
                    totals["total_usd_capacity"],
                ],
            }
        )
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

# --- Comparison Tab ---
with Tabs[2]:
    st.subheader("Scenario Comparison")
    if not st.session_state.scenarios:
        st.info("Add at least one scenario to compare.")
    else:
        rows = []
        for sc in st.session_state.scenarios.values():
            totals = scenario_totals(sc, st.session_state.assumptions)
            rows.append(
                {
                    "Scenario": sc.name,
                    "Edition": sc.edition,
                    "Provider": sc.provider,
                    "Region": sc.region,
                    "TB/mo": sc.tb_per_month,
                    "Monthly On-Demand ($)": totals["monthly_usd_ondemand"],
                    "Monthly Capacity ($)": totals["monthly_usd_capacity"],
                    "Credits/yr": totals["credits_total"],
                }
            )
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# --- Assumptions Tab ---
with Tabs[3]:
    st.subheader("Assumptions")
    a: Assumptions = st.session_state.assumptions

    st.markdown("**Credit price by Edition** (read from credits.csv)")
    st.dataframe(pd.DataFrame(list(a.credit_price_by_edition.items()), columns=["Edition", "$ / credit"]))

    c1, c2, c3 = st.columns(3)
    with c1:
        a.storage_price_per_tb = st.number_input("Storage price per TB ($)", 0.0, 1000.0, value=float(a.storage_price_per_tb))
        a.cloud_services_pct_default = st.number_input("Cloud Services % (default)", 0.0, 100.0, value=float(a.cloud_services_pct_default))
    with c2:
        a.capacity_price_per_tb = st.number_input("Capacity price per TB ($)", 0.0, 1000.0, value=float(a.capacity_price_per_tb))
        a.time_travel_storage_overhead_pct = st.number_input("Time Travel storage overhead %", 0.0, 100.0, value=float(a.time_travel_storage_overhead_pct))
    with c3:
        a.per_second_billing = st.checkbox("Per-second billing (min 60s)", value=a.per_second_billing)
        a.fail_safe_storage_overhead_pct = st.number_input("Fail-safe storage overhead %", 0.0, 100.0, value=float(a.fail_safe_storage_overhead_pct))

    st.session_state.assumptions = a

# --- Templates Tab ---
with Tabs[4]:
    st.subheader("Workload Templates (adds to active scenario)")
    if not st.session_state.active_scenario:
        st.info("Select an active scenario in the sidebar first.")
    else:
        sc = st.session_state.scenarios[st.session_state.active_scenario]
        t = st.selectbox("Choose template", ["(none)", "BI 9-5", "Nightly ELT", "Streaming 24/7", "Ad-hoc Analytics"])
        size_opt = st.selectbox("Default size for template", options=valid_sizes, index=valid_sizes.index("M") if "M" in valid_sizes else 0)
        if st.button("Apply Template") and t != "(none)":
            presets: List[Warehouse] = []
            if t == "BI 9-5":
                presets = [
                    Warehouse("WH_BI", size_opt, hours_per_day=8, days_per_week=5, min_clusters=1, max_clusters=2, cloud_services_pct=10),
                ]
            elif t == "Nightly ELT":
                presets = [
                    Warehouse("WH_ELT", size_opt, hours_per_day=3, days_per_week=7, min_clusters=1, max_clusters=1, cloud_services_pct=10),
                ]
            elif t == "Streaming 24/7":
                presets = [
                    Warehouse("WH_STREAM", size_opt, hours_per_day=24, days_per_week=7, min_clusters=2, max_clusters=4, cloud_services_pct=12),
                ]
            elif t == "Ad-hoc Analytics":
                presets = [
                    Warehouse("WH_ADHOC", size_opt, hours_per_day=4, days_per_week=3, min_clusters=1, max_clusters=1, cloud_services_pct=8),
                ]
            # merge/replace by name
            names = {w.name for w in sc.warehouses}
            for p in presets:
                if p.name in names:
                    sc.warehouses = [w for w in sc.warehouses if w.name != p.name] + [p]
                else:
                    sc.warehouses.append(p)
            st.session_state.scenarios[sc.name] = sc
            st.success(f"Template '{t}' applied to scenario '{sc.name}'.")

# --- Export Tab ---
with Tabs[5]:
    st.subheader("Export Results & Assumptions")
    if not st.session_state.scenarios:
        st.info("Add at least one scenario to export.")
    else:
        # Build a multi-sheet-like CSV (stacked with section markers) or provide per-scenario CSV
        export_rows = []
        for sc in st.session_state.scenarios.values():
            totals = scenario_totals(sc, st.session_state.assumptions)
            export_rows.append({"Section":"Scenario","Key":"Name","Value": sc.name})
            export_rows.append({"Section":"Scenario","Key":"Edition","Value": sc.edition})
            export_rows.append({"Section":"Scenario","Key":"Provider","Value": sc.provider})
            export_rows.append({"Section":"Scenario","Key":"Region","Value": sc.region})
            export_rows.append({"Section":"Scenario","Key":"TB/mo","Value": sc.tb_per_month})
            export_rows.append({"Section":"Totals","Key":"Monthly On-Demand ($)","Value": totals['monthly_usd_ondemand']})
            export_rows.append({"Section":"Totals","Key":"Monthly Capacity ($)","Value": totals['monthly_usd_capacity']})
            export_rows.append({"Section":"Totals","Key":"Credits/yr","Value": totals['credits_total']})
            for w in sc.warehouses:
                export_rows.append({"Section":"Warehouse","Key": w.name, "Value": json.dumps(asdict(w))})
        exp_df = pd.DataFrame(export_rows)
        csv_buf = io.StringIO()
        exp_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name="snowflake_cost_estimates.csv",
            mime="text/csv",
        )

        # Export assumptions JSON
        a_json = json.dumps(asdict(st.session_state.assumptions), indent=2)
        st.download_button(
            "Download Assumptions (JSON)",
            data=a_json,
            file_name="assumptions.json",
            mime="application/json",
        )

# --- Glossary Tab ---
with Tabs[6]:
    st.subheader("Glossary (Quick References)")
    st.markdown(
        """
        - **Credit**: Unit of compute billing; varies by warehouse size.
        - **Cloud Services %**: Overhead for metadata, optimization, compilation, etc. Defaults around 10%.
        - **Time Travel / Fail-safe**: Data retention features that can increase storage footprint.
        - **Auto-suspend**: Auto-pauses a warehouse after inactivity; reduces billable time.
        - **Min/Max Clusters**: For multi-cluster warehouses, bounds on concurrent clusters.
        - **Concurrency Target**: Expected concurrent queries; used here to estimate average clusters.
        """
    )
