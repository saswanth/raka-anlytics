from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from telemetry_analysis import SENSOR_COLUMNS, analyze_multiple_datasets

st.set_page_config(page_title="RAKA Analytics", page_icon=":bar_chart:", layout="wide")


def _build_raka_reply(
    prompt: str,
    data: pd.DataFrame,
    selected_sensor: str,
    selected_device: str,
    selected_dataset: str,
) -> str:
    p = prompt.lower().strip()
    sensor_cols = [c for c in SENSOR_COLUMNS if c in data.columns]

    if data.empty:
        return "No rows are available after your current filters. Try adjusting dataset, device, or date range."

    if any(k in p for k in ["hi", "hello", "hey", "start"]):
        return "Hi, I am RAKA, your telemetry assistant. I can generate telemetry semantics on the go."

    if "summary" in p:
        return (
            f"Summary for {selected_dataset}: rows={len(data):,}, columns={data.shape[1]}, "
            f"device_filter={selected_device}, sensor_focus={selected_sensor}."
        )

    if "missing" in p or "null" in p:
        miss = {k: int(v) for k, v in data.isna().sum().items() if int(v) > 0}
        return f"Missing-data check: {miss if miss else 'none detected in current filtered view.'}"

    if "device" in p:
        if "device" not in data.columns:
            return "No device column is available in the current dataset view."
        dist = data["device"].value_counts(normalize=True) * 100.0
        top_device = dist.index[0]
        return f"Top device in current view is {top_device} with {dist.iloc[0]:.2f}% of records."

    if any(k in p for k in ["correlation", "relationship", "related"]):
        if len(sensor_cols) < 2:
            return "At least two sensor columns are required to compute correlations."
        corr = data[sensor_cols].corr(numeric_only=True).stack()
        corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)].sort_values(ascending=False)
        if corr.empty:
            return "Could not compute correlations in the current filtered view."
        pair = corr.index[0]
        return f"Strongest positive correlation is {pair[0].upper()} vs {pair[1].upper()} with r={corr.iloc[0]:.4f}."

    if any(k in p for k in ["trend", "increase", "decrease"]):
        if selected_sensor not in sensor_cols:
            selected_sensor = sensor_cols[0] if sensor_cols else ""
        if not selected_sensor:
            return "No supported sensor columns are available for trend analysis."
        if "timestamp" not in data.columns or data["timestamp"].isna().all():
            return f"{selected_sensor.upper()} mean in current view is {data[selected_sensor].mean():.4f}."
        trend_series = (
            data.dropna(subset=["timestamp"])
            .groupby(data["timestamp"].dt.floor("D"))[selected_sensor]
            .mean()
            .sort_index()
        )
        if len(trend_series) < 2:
            return f"Not enough daily points to infer a trend for {selected_sensor.upper()}."
        first = float(trend_series.iloc[0])
        last = float(trend_series.iloc[-1])
        direction = "up" if last > first else "down" if last < first else "flat"
        return f"{selected_sensor.upper()} trend is {direction} ({first:.4f} -> {last:.4f}) over the selected period."

    if "compare" in p and "dataset" in p:
        if "dataset" not in data.columns:
            return "Only one dataset is currently loaded, so comparison is unavailable."
        counts = data["dataset"].value_counts().to_dict()
        return f"Dataset row comparison: {counts}"

    return (
        "Try one of these: summary, missing data, top device, show correlation, trend, or compare datasets."
    )


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-a: #f5f2e9;
  --bg-b: #d8ece8;
  --ink: #10251f;
  --muted: #4b6660;
  --card: rgba(255, 255, 255, 0.76);
  --line: rgba(16, 37, 31, 0.15);
  --accent: #d3542b;
  --accent-2: #0b7a75;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at 12% 12%, rgba(211, 84, 43, 0.12), transparent 35%),
    radial-gradient(circle at 88% 10%, rgba(11, 122, 117, 0.16), transparent 40%),
    linear-gradient(140deg, var(--bg-a), var(--bg-b));
}

.block-container {
  padding-top: 1.35rem;
  padding-bottom: 1.35rem;
}

.hero {
  padding: 1.15rem 1.4rem;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: linear-gradient(120deg, rgba(255,255,255,.80), rgba(255,255,255,.50));
  backdrop-filter: blur(6px);
}

.hero h1 {
  margin: 0;
  line-height: 1.05;
  font-size: clamp(1.7rem, 2.3vw, 2.7rem);
  letter-spacing: -0.02em;
}

.hero p {
  margin: 0.4rem 0 0 0;
  color: var(--muted);
}

.card {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  background: var(--card);
}

.metric-label {
  color: var(--muted);
  font-size: 0.84rem;
}

.metric-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.1rem;
  font-weight: 500;
}

ul.trends {
  margin-top: 0.3rem;
  padding-left: 1.1rem;
}

ul.trends li {
  margin-bottom: 0.35rem;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>RAKA Analytics</h1>
  <p>Multi-dataset telemetry intelligence with interactive semantics assistant and actionable controls.</p>
</div>
    """,
    unsafe_allow_html=True,
)

csv_files = sorted([p.name for p in Path(".").glob("*.csv")])
default_selection = ["iot_telemetry_data.csv"] if "iot_telemetry_data.csv" in csv_files else csv_files[:1]

with st.sidebar:
    st.header("Datasets")
    selected_files = st.multiselect("Choose one or more CSV files", options=csv_files, default=default_selection)
    custom_paths_text = st.text_area(
        "Additional paths (one per line)",
        value="",
        help="Use this for CSV files outside the workspace root if needed.",
    )
    run = st.button("Refresh Analysis", use_container_width=True)

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = True
if run:
    st.session_state.run_analysis = True

if st.session_state.run_analysis:
    try:
        extra_paths = [line.strip() for line in custom_paths_text.splitlines() if line.strip()]
        all_paths: List[str] = selected_files + extra_paths

        if not all_paths:
            st.warning("Select at least one CSV dataset to continue.")
            st.stop()

        multi = analyze_multiple_datasets(all_paths)
        combined_df = multi.combined_df.copy()

        if "raka_popup_shown" not in st.session_state:
            st.session_state.raka_popup_shown = False
        if not st.session_state.raka_popup_shown:
            st.toast("Hi, I am your telemetry assistant RAKA. I can generate telemetry semantics on the go.")
            st.session_state.raka_popup_shown = True

        dataset_options = ["All Datasets"] + list(multi.per_dataset.keys())
        with st.sidebar:
            st.header("View Controls")
            selected_dataset = st.selectbox("Dataset Scope", options=dataset_options, index=0)

        if selected_dataset == "All Datasets":
            df_scope = combined_df.copy()
            active_summary = multi.combined_summary
            active_trends = multi.combined_trends
        else:
            df_scope = multi.per_dataset[selected_dataset].cleaned_df.copy()
            df_scope["dataset"] = selected_dataset
            active_summary = multi.per_dataset[selected_dataset].summary
            active_trends = multi.per_dataset[selected_dataset].trends

        device_options = ["All"]
        if "device" in df_scope.columns:
            device_options += sorted(df_scope["device"].dropna().unique().tolist())

        sensor_cols = [c for c in SENSOR_COLUMNS if c in df_scope.columns]

        with st.sidebar:
            selected_device = st.selectbox("Device", options=device_options, index=0)
            selected_sensor = st.selectbox(
                "Sensor Focus",
                options=sensor_cols if sensor_cols else ["none"],
                index=0,
            )
            chart_style = st.selectbox("Volume Chart Type", options=["Area", "Line", "Bar"], index=0)

        df_view = df_scope.copy()
        if selected_device != "All" and "device" in df_view.columns:
            df_view = df_view[df_view["device"] == selected_device].copy()

        if "timestamp" in df_view.columns and df_view["timestamp"].notna().any():
            min_day = df_view["timestamp"].min().date()
            max_day = df_view["timestamp"].max().date()
            with st.sidebar:
                chosen_days = st.date_input(
                    "Date Range",
                    value=(min_day, max_day),
                    min_value=min_day,
                    max_value=max_day,
                )
            if isinstance(chosen_days, tuple) and len(chosen_days) == 2:
                start_day, end_day = chosen_days
                df_view = df_view[
                    (df_view["timestamp"].dt.date >= start_day)
                    & (df_view["timestamp"].dt.date <= end_day)
                ].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f"<div class='card'><div class='metric-label'>Rows (filtered)</div><div class='metric-value'>{len(df_view):,}</div></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div class='card'><div class='metric-label'>Datasets Loaded</div><div class='metric-value'>{multi.combined_summary['dataset_count']}</div></div>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<div class='card'><div class='metric-label'>Duplicates Removed</div><div class='metric-value'>{active_summary['duplicates_removed']}</div></div>",
            unsafe_allow_html=True,
        )
        c4.markdown(
            f"<div class='card'><div class='metric-label'>Devices</div><div class='metric-value'>{active_summary['devices']}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Summary")
        st.write(
            f"Scope: {selected_dataset} | Shape before: {active_summary['shape_before']} | Shape after cleaning: {active_summary['shape_after']}"
        )
        if active_summary.get("time_start") and active_summary.get("time_end"):
            st.write(f"Time range: {active_summary['time_start']} to {active_summary['time_end']}")
        if active_summary["missing_nonzero"]:
            st.warning(f"Missing data detected: {active_summary['missing_nonzero']}")
        else:
            st.success("Missing data: none")

        left, right = st.columns([1.1, 1.0])

        with left:
            daily_counts = pd.Series(dtype="int64")
            if "timestamp" in df_view.columns and df_view["timestamp"].notna().any():
                daily_counts = (
                    df_view.dropna(subset=["timestamp"])
                    .groupby(df_view["timestamp"].dt.floor("D"))
                    .size()
                    .sort_index()
                )

            if not daily_counts.empty:
                vol = daily_counts.reset_index()
                vol.columns = ["day", "records"]
                if chart_style == "Line":
                    fig_vol = px.line(vol, x="day", y="records", title="Daily Record Volume", color_discrete_sequence=["#0b7a75"])
                elif chart_style == "Bar":
                    fig_vol = px.bar(vol, x="day", y="records", title="Daily Record Volume", color_discrete_sequence=["#0b7a75"])
                else:
                    fig_vol = px.area(vol, x="day", y="records", title="Daily Record Volume", line_shape="spline", color_discrete_sequence=["#0b7a75"])
                fig_vol.update_layout(margin=dict(l=20, r=20, t=45, b=20), height=315)
                st.plotly_chart(fig_vol, use_container_width=True)

            if "dataset" in df_view.columns and df_view["dataset"].nunique() > 1:
                ds_counts = df_view["dataset"].value_counts().reset_index()
                ds_counts.columns = ["dataset", "rows"]
                fig_ds = px.bar(
                    ds_counts,
                    x="dataset",
                    y="rows",
                    title="Rows by Dataset",
                    color="dataset",
                    color_discrete_sequence=["#d3542b", "#0b7a75", "#34544f", "#8a6c1f"],
                )
                fig_ds.update_layout(showlegend=False, margin=dict(l=20, r=20, t=45, b=20), height=300)
                st.plotly_chart(fig_ds, use_container_width=True)

            if selected_sensor in df_view.columns:
                hist = px.histogram(
                    df_view,
                    x=selected_sensor,
                    nbins=45,
                    title=f"Distribution: {selected_sensor.upper()}",
                    color_discrete_sequence=["#d3542b"],
                )
                hist.update_layout(margin=dict(l=20, r=20, t=45, b=20), height=300)
                st.plotly_chart(hist, use_container_width=True)

        with right:
            st.markdown("### Trends")
            trend_html = "".join([f"<li>{t}</li>" for t in active_trends[:9]])
            st.markdown(f"<ul class='trends'>{trend_html}</ul>", unsafe_allow_html=True)

            if len(sensor_cols) > 1:
                corr = df_view[sensor_cols].corr(numeric_only=True)
                corr_long = corr.reset_index().melt(id_vars="index", var_name="sensor_b", value_name="corr")
                corr_long = corr_long.rename(columns={"index": "sensor_a"})
                fig_corr = px.density_heatmap(
                    corr_long,
                    x="sensor_a",
                    y="sensor_b",
                    z="corr",
                    text_auto=".2f",
                    title="Sensor Correlation Map",
                    color_continuous_scale="RdYlGn",
                )
                fig_corr.update_layout(margin=dict(l=20, r=20, t=45, b=20), height=330)
                st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("### RAKA Chat Assistant")
            st.caption("Use buttons for quick insights or ask your own question.")

            if "raka_messages" not in st.session_state:
                st.session_state.raka_messages = [
                    {
                        "role": "assistant",
                        "content": "Hi, I am your telemetry assistant RAKA. I can generate telemetry semantics on the go.",
                    }
                ]

            b1, b2 = st.columns(2)
            b3, b4 = st.columns(2)
            quick_prompt = ""
            if b1.button("Dataset Summary", use_container_width=True):
                quick_prompt = "summary"
            if b2.button("Missing Data", use_container_width=True):
                quick_prompt = "missing data"
            if b3.button("Top Device", use_container_width=True):
                quick_prompt = "top device"
            if b4.button("Show Correlation", use_container_width=True):
                quick_prompt = "show correlation"

            b5, b6 = st.columns(2)
            if b5.button("Sensor Trend", use_container_width=True):
                quick_prompt = "trend"
            if b6.button("Compare Datasets", use_container_width=True):
                quick_prompt = "compare datasets"

            user_prompt = st.chat_input("Message RAKA...")
            if quick_prompt and not user_prompt:
                user_prompt = quick_prompt

            for message in st.session_state.raka_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if user_prompt:
                st.session_state.raka_messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.write(user_prompt)

                reply = _build_raka_reply(
                    user_prompt,
                    df_view,
                    selected_sensor,
                    selected_device,
                    selected_dataset,
                )
                st.session_state.raka_messages.append({"role": "assistant", "content": reply})
                st.toast("RAKA generated a semantic response.")
                with st.chat_message("assistant"):
                    st.write(reply)

    except Exception as exc:
        st.error(f"Could not analyze dataset(s): {exc}")
        st.info("Verify file paths and ensure selected files are valid CSV datasets.")