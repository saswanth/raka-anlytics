from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RAKA Analytics - Weather Air", page_icon=":bar_chart:", layout="wide")


def load_weather_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": pd.NA, "None": pd.NA})

    for time_col in ["weather_datetime_utc", "air_datetime_utc", "collection_time_utc"]:
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.drop_duplicates().copy()
    return df


def weather_chat_reply(prompt: str, data: pd.DataFrame, country_filter: str, city_filter: str) -> str:
    p = prompt.lower().strip()

    if data.empty:
        return "No rows available for selected weather filters."

    if any(k in p for k in ["hi", "hello", "hey", "start"]):
        return "Hi, I am RAKA Analytics Assistant for Weather and Air Quality. Ask me AQI, PM2.5, or city-level insights."

    if "summary" in p:
        return (
            f"Current weather view summary: rows={len(data):,}, columns={data.shape[1]}, "
            f"country_filter={country_filter}, city_filter={city_filter}."
        )

    if "missing" in p or "null" in p:
        miss = {k: int(v) for k, v in data.isna().sum().items() if int(v) > 0}
        return f"Missing-data report: {miss if miss else 'none in current weather view.'}"

    if "aqi" in p or "pollution" in p:
        if "aqi" not in data.columns:
            return "AQI column is not available."
        return f"AQI stats -> min={data['aqi'].min():.2f}, mean={data['aqi'].mean():.2f}, max={data['aqi'].max():.2f}."

    if "hottest" in p or "temperature" in p:
        if not {"city_name", "temp"}.issubset(set(data.columns)):
            return "Required columns city_name and temp are unavailable."
        hottest = data.sort_values("temp", ascending=False).iloc[0]
        return f"Hottest city in current view is {hottest['city_name']} at {float(hottest['temp']):.2f}C."

    if "worst" in p or "pm2" in p:
        if not {"city_name", "pm2_5"}.issubset(set(data.columns)):
            return "PM2.5 insights are not available in this view."
        worst = data.sort_values("pm2_5", ascending=False).iloc[0]
        return f"Highest PM2.5 city is {worst['city_name']} with PM2.5={float(worst['pm2_5']):.2f}."

    if "trend" in p and "aqi" in p:
        time_col = "air_datetime_utc" if "air_datetime_utc" in data.columns else None
        if time_col is None or data[time_col].isna().all():
            return "AQI trend cannot be computed because datetime is unavailable."
        trend = data.dropna(subset=[time_col]).groupby(data[time_col].dt.floor("D"))["aqi"].mean().sort_index()
        if len(trend) < 2:
            return "Not enough daily points to infer AQI trend."
        first = float(trend.iloc[0])
        last = float(trend.iloc[-1])
        direction = "up" if last > first else "down" if last < first else "flat"
        return f"AQI trend is {direction} from {first:.2f} to {last:.2f}."

    return "Ask about summary, AQI, PM2.5, hottest city, trend AQI, or missing data."


st.markdown("## RAKA Analytics - Weather and Air Pollution Dashboard")
st.caption("Dedicated dashboard for openweather_weather_airpollution_top3cities_per_country.csv")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input(
        "Weather CSV path",
        value="openweather_weather_airpollution_top3cities_per_country.csv",
    )
    refresh = st.button("Refresh", use_container_width=True)

if "weather_run" not in st.session_state:
    st.session_state.weather_run = True
if refresh:
    st.session_state.weather_run = True

if st.session_state.weather_run:
    try:
        df = load_weather_data(data_path)

        if "weather_popup" not in st.session_state:
            st.session_state.weather_popup = False
        if not st.session_state.weather_popup:
            st.toast("Hi, I am your Weather RAKA assistant. I can answer AQI and city-weather questions.")
            st.session_state.weather_popup = True

        countries = ["All"] + (sorted(df["country_name"].dropna().unique().tolist()) if "country_name" in df.columns else [])
        with st.sidebar:
            st.header("Filters")
            selected_country = st.selectbox("Country", options=countries, index=0)

        view = df.copy()
        if selected_country != "All" and "country_name" in view.columns:
            view = view[view["country_name"] == selected_country].copy()

        cities = ["All"] + (sorted(view["city_name"].dropna().unique().tolist()) if "city_name" in view.columns else [])
        with st.sidebar:
            selected_city = st.selectbox("City", options=cities, index=0)
            measure = st.selectbox("Primary metric", options=["aqi", "pm2_5", "temp", "humidity"], index=0)

        if selected_city != "All" and "city_name" in view.columns:
            view = view[view["city_name"] == selected_city].copy()

        time_col = "air_datetime_utc" if "air_datetime_utc" in view.columns else "weather_datetime_utc"
        if time_col in view.columns and view[time_col].notna().any():
            min_day = view[time_col].min().date()
            max_day = view[time_col].max().date()
            with st.sidebar:
                day_range = st.date_input("Date range", value=(min_day, max_day), min_value=min_day, max_value=max_day)
            if isinstance(day_range, tuple) and len(day_range) == 2:
                start_day, end_day = day_range
                view = view[(view[time_col].dt.date >= start_day) & (view[time_col].dt.date <= end_day)].copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(view):,}")
        m2.metric("Countries", view["country_name"].nunique() if "country_name" in view.columns else 0)
        m3.metric("Cities", view["city_name"].nunique() if "city_name" in view.columns else 0)
        m4.metric("Avg AQI", f"{view['aqi'].mean():.2f}" if "aqi" in view.columns and not view.empty else "N/A")

        left, right = st.columns([1.25, 1.0])

        with left:
            if time_col in view.columns and measure in view.columns and view[time_col].notna().any():
                series = view.dropna(subset=[time_col]).groupby(view[time_col].dt.floor("D"))[measure].mean().sort_index()
                if not series.empty:
                    tdf = series.reset_index()
                    tdf.columns = ["day", "value"]
                    fig = px.line(tdf, x="day", y="value", title=f"Daily {measure.upper()} Trend")
                    st.plotly_chart(fig, use_container_width=True)

            if "aqi" in view.columns:
                fig_aqi = px.histogram(view, x="aqi", nbins=20, title="AQI Distribution")
                st.plotly_chart(fig_aqi, use_container_width=True)

            if {"country_name", "aqi"}.issubset(set(view.columns)):
                by_country = view.groupby("country_name", as_index=False)["aqi"].mean().sort_values("aqi", ascending=False).head(15)
                fig_c = px.bar(by_country, x="country_name", y="aqi", title="Top Countries by Average AQI")
                st.plotly_chart(fig_c, use_container_width=True)

        with right:
            st.markdown("### Weather Chatbot - RAKA")
            st.caption("Ask human-style questions about AQI, PM2.5, weather and city comparisons.")

            if "weather_msgs" not in st.session_state:
                st.session_state.weather_msgs = [
                    {"role": "assistant", "content": "Hi, I am RAKA Weather assistant. Ask me AQI trend, hottest city, PM2.5 leader, or data summary."}
                ]

            q1, q2 = st.columns(2)
            q3, q4 = st.columns(2)
            quick_prompt = ""
            if q1.button("Summary", use_container_width=True):
                quick_prompt = "summary"
            if q2.button("AQI Stats", use_container_width=True):
                quick_prompt = "aqi"
            if q3.button("Hottest City", use_container_width=True):
                quick_prompt = "hottest city"
            if q4.button("Worst PM2.5", use_container_width=True):
                quick_prompt = "pm2.5"

            for msg in st.session_state.weather_msgs:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            user_text = st.chat_input("Ask Weather RAKA...")
            if quick_prompt and not user_text:
                user_text = quick_prompt

            if user_text:
                st.session_state.weather_msgs.append({"role": "user", "content": user_text})
                with st.chat_message("user"):
                    st.write(user_text)

                reply = weather_chat_reply(user_text, view, selected_country, selected_city)
                st.session_state.weather_msgs.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)

    except Exception as exc:
        st.error(f"Could not load Weather dashboard: {exc}")
