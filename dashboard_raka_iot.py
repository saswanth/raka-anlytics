from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RAKA Analytics - IoT", page_icon=":bar_chart:", layout="wide")

SENSOR_COLS = ["co", "humidity", "lpg", "smoke", "temp"]


def load_iot_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": pd.NA, "None": pd.NA})

    df = df.drop_duplicates().copy()
    if "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    else:
        df["timestamp"] = pd.NaT
    return df


def iot_chat_reply(prompt: str, data: pd.DataFrame, sensor_focus: str, device_filter: str) -> str:
    p = prompt.lower().strip()

    if data.empty:
        return "No rows available for the selected filters. Try a wider date range or All devices."

    if any(k in p for k in ["hi", "hello", "hey", "start"]):
        return "Hi, I am RAKA Analytics Assistant for IoT. Ask me trends, anomalies, missing data, or top devices."

    if "summary" in p:
        return (
            f"Current view summary: rows={len(data):,}, columns={data.shape[1]}, "
            f"device_filter={device_filter}, sensor_focus={sensor_focus}."
        )

    if "missing" in p or "null" in p:
        miss = {k: int(v) for k, v in data.isna().sum().items() if int(v) > 0}
        return f"Missing-data report: {miss if miss else 'none in current IoT view.'}"

    if "top device" in p or "device" in p:
        if "device" not in data.columns:
            return "Device column is not available in this dataset."
        d = data["device"].value_counts(normalize=True) * 100
        return f"Top device is {d.index[0]} with {d.iloc[0]:.2f}% of filtered records."

    if "anomaly" in p or "outlier" in p:
        if sensor_focus not in data.columns:
            return "Choose a valid sensor focus to evaluate anomalies."
        s = data[sensor_focus].dropna()
        if s.empty:
            return f"No values available for {sensor_focus.upper()} in current filters."
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        return (
            f"Anomaly check for {sensor_focus.upper()}: {outliers} outliers based on IQR bounds "
            f"[{lower:.4f}, {upper:.4f}]."
        )

    if "trend" in p or "increase" in p or "decrease" in p:
        if sensor_focus not in data.columns:
            return "Choose a valid sensor focus to evaluate trend."
        if "timestamp" not in data.columns or data["timestamp"].isna().all():
            return f"{sensor_focus.upper()} mean is {data[sensor_focus].mean():.4f} (timestamp not available for trend)."
        t = (
            data.dropna(subset=["timestamp"])
            .groupby(data["timestamp"].dt.floor("D"))[sensor_focus]
            .mean()
            .sort_index()
        )
        if len(t) < 2:
            return f"Not enough daily points for {sensor_focus.upper()} trend detection."
        first = float(t.iloc[0])
        last = float(t.iloc[-1])
        direction = "up" if last > first else "down" if last < first else "flat"
        return f"{sensor_focus.upper()} trend is {direction} from {first:.4f} to {last:.4f}."

    if "correlation" in p:
        cols = [c for c in SENSOR_COLS if c in data.columns]
        if len(cols) < 2:
            return "Need at least two sensor columns for correlation analysis."
        corr = data[cols].corr(numeric_only=True).stack()
        corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)].sort_values(ascending=False)
        if corr.empty:
            return "Could not compute correlation in the current view."
        pair = corr.index[0]
        return f"Strongest correlation is {pair[0].upper()} vs {pair[1].upper()} with r={corr.iloc[0]:.4f}."

    return "Ask about summary, trend, anomalies, top device, missing data, or correlation."


st.markdown("## RAKA Analytics - IoT Dashboard")
st.caption("Dedicated dashboard for iot_telemetry_data.csv")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("IoT CSV path", value="iot_telemetry_data.csv")
    refresh = st.button("Refresh", use_container_width=True)

if "iot_run" not in st.session_state:
    st.session_state.iot_run = True
if refresh:
    st.session_state.iot_run = True

if st.session_state.iot_run:
    try:
        df = load_iot_data(data_path)

        if "iot_popup" not in st.session_state:
            st.session_state.iot_popup = False
        if not st.session_state.iot_popup:
            st.toast("Hi, I am your IoT RAKA assistant. I can answer questions about this dataset.")
            st.session_state.iot_popup = True

        device_options = ["All"] + (sorted(df["device"].dropna().unique().tolist()) if "device" in df.columns else [])
        sensor_options = [c for c in SENSOR_COLS if c in df.columns]

        with st.sidebar:
            st.header("Filters")
            selected_device = st.selectbox("Device", options=device_options, index=0)
            selected_sensor = st.selectbox("Sensor focus", options=sensor_options if sensor_options else ["none"], index=0)
            chart_kind = st.selectbox("Volume chart", options=["Area", "Line", "Bar"], index=0)

        view = df.copy()
        if selected_device != "All" and "device" in view.columns:
            view = view[view["device"] == selected_device].copy()

        if "timestamp" in view.columns and view["timestamp"].notna().any():
            min_day = view["timestamp"].min().date()
            max_day = view["timestamp"].max().date()
            with st.sidebar:
                day_range = st.date_input("Date range", value=(min_day, max_day), min_value=min_day, max_value=max_day)
            if isinstance(day_range, tuple) and len(day_range) == 2:
                start_day, end_day = day_range
                view = view[(view["timestamp"].dt.date >= start_day) & (view["timestamp"].dt.date <= end_day)].copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(view):,}")
        m2.metric("Columns", view.shape[1])
        m3.metric("Devices", view["device"].nunique() if "device" in view.columns else 0)
        m4.metric("Sensor Focus", selected_sensor.upper() if selected_sensor != "none" else "N/A")

        left, right = st.columns([1.2, 1.0])

        with left:
            if "timestamp" in view.columns and view["timestamp"].notna().any():
                daily = view.dropna(subset=["timestamp"]).groupby(view["timestamp"].dt.floor("D")).size().sort_index()
                if not daily.empty:
                    vol = daily.reset_index()
                    vol.columns = ["day", "records"]
                    if chart_kind == "Line":
                        fig = px.line(vol, x="day", y="records", title="Daily IoT Record Volume")
                    elif chart_kind == "Bar":
                        fig = px.bar(vol, x="day", y="records", title="Daily IoT Record Volume")
                    else:
                        fig = px.area(vol, x="day", y="records", title="Daily IoT Record Volume")
                    st.plotly_chart(fig, use_container_width=True)

            if selected_sensor in view.columns:
                hist = px.histogram(view, x=selected_sensor, nbins=40, title=f"Distribution: {selected_sensor.upper()}")
                st.plotly_chart(hist, use_container_width=True)

            if "device" in view.columns:
                dcount = view["device"].value_counts().reset_index()
                dcount.columns = ["device", "count"]
                fig_d = px.bar(dcount, x="device", y="count", title="Records by Device")
                st.plotly_chart(fig_d, use_container_width=True)

        with right:
            st.markdown("### IoT Chatbot - RAKA")
            st.caption("Click a quick action or ask any question in plain language.")

            if "iot_msgs" not in st.session_state:
                st.session_state.iot_msgs = [
                    {"role": "assistant", "content": "Hi, I am RAKA IoT assistant. Ask me about trends, devices, anomalies, or data quality."}
                ]

            q1, q2 = st.columns(2)
            q3, q4 = st.columns(2)
            quick_prompt = ""
            if q1.button("Summary", use_container_width=True):
                quick_prompt = "summary"
            if q2.button("Trend", use_container_width=True):
                quick_prompt = "trend"
            if q3.button("Top Device", use_container_width=True):
                quick_prompt = "top device"
            if q4.button("Anomalies", use_container_width=True):
                quick_prompt = "anomaly"

            for msg in st.session_state.iot_msgs:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            user_text = st.chat_input("Ask IoT RAKA...")
            if quick_prompt and not user_text:
                user_text = quick_prompt

            if user_text:
                st.session_state.iot_msgs.append({"role": "user", "content": user_text})
                with st.chat_message("user"):
                    st.write(user_text)

                reply = iot_chat_reply(user_text, view, selected_sensor, selected_device)
                st.session_state.iot_msgs.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)

    except Exception as exc:
        st.error(f"Could not load IoT dashboard: {exc}")
