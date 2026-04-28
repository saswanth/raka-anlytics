from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RAKA Analytics - Instagram Lifestyle", page_icon=":bar_chart:", layout="wide")


NUMERIC_CHAT_COLS = [
    "user_engagement_score",
    "daily_active_minutes_instagram",
    "average_session_length_minutes",
    "followers_count",
    "following_count",
    "notification_response_rate",
]


def load_instagram_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": pd.NA, "None": pd.NA})

    if "last_login_date" in df.columns:
        df["last_login_date"] = pd.to_datetime(df["last_login_date"], errors="coerce")

    df = df.drop_duplicates().copy()
    return df


def instagram_chat_reply(prompt: str, data: pd.DataFrame, country_filter: str, gender_filter: str) -> str:
    p = prompt.lower().strip()

    if data.empty:
        return "No rows available for current filters. Try widening country, gender, or age range."

    if any(k in p for k in ["hi", "hello", "hey", "start"]):
        return "Hi, I am RAKA Instagram assistant. I can answer lifestyle and engagement questions from this dataset."

    if "summary" in p:
        return (
            f"Current summary: rows={len(data):,}, columns={data.shape[1]}, "
            f"country_filter={country_filter}, gender_filter={gender_filter}."
        )

    if "missing" in p or "null" in p:
        miss = {k: int(v) for k, v in data.isna().sum().items() if int(v) > 0}
        return f"Missing-data report: {miss if miss else 'none in current Instagram view.'}"

    if "engagement" in p:
        if "user_engagement_score" not in data.columns:
            return "user_engagement_score column is unavailable."
        s = data["user_engagement_score"].dropna()
        return f"Engagement stats -> min={s.min():.2f}, mean={s.mean():.2f}, median={s.median():.2f}, max={s.max():.2f}."

    if "country" in p:
        if "country" not in data.columns:
            return "Country column is unavailable."
        c = data["country"].value_counts(normalize=True) * 100
        return f"Top country in current view is {c.index[0]} with {c.iloc[0]:.2f}% of users."

    if "theme" in p or "content" in p:
        if "preferred_content_theme" not in data.columns:
            return "preferred_content_theme column is unavailable."
        t = data["preferred_content_theme"].value_counts(normalize=True) * 100
        return f"Top content theme is {t.index[0]} with {t.iloc[0]:.2f}% share."

    if "followers" in p:
        if "followers_count" not in data.columns:
            return "followers_count column is unavailable."
        top = data["followers_count"].max()
        avg = data["followers_count"].mean()
        return f"Followers stats -> average={avg:.2f}, max={top:.2f}."

    if "correlation" in p or "relationship" in p:
        cols = [c for c in NUMERIC_CHAT_COLS if c in data.columns]
        if len(cols) < 2:
            return "Not enough numeric columns available for correlation."
        corr = data[cols].corr(numeric_only=True).stack()
        corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)].sort_values(ascending=False)
        if corr.empty:
            return "Could not compute correlation in current view."
        pair = corr.index[0]
        return f"Strongest positive correlation is {pair[0]} vs {pair[1]} with r={corr.iloc[0]:.4f}."

    if "trend" in p and "login" in p:
        if "last_login_date" not in data.columns or data["last_login_date"].isna().all():
            return "Login-date trend is unavailable because last_login_date is missing."
        trend = data.dropna(subset=["last_login_date"]).groupby(data["last_login_date"].dt.floor("D")).size().sort_index()
        if len(trend) < 2:
            return "Not enough date points to infer login trend."
        first = int(trend.iloc[0])
        last = int(trend.iloc[-1])
        direction = "up" if last > first else "down" if last < first else "flat"
        return f"Daily login-row volume trend is {direction} ({first} -> {last})."

    return "Ask about summary, engagement, top country, content theme, followers, correlation, login trend, or missing data."


st.markdown("## RAKA Analytics - Instagram Usage Lifestyle Dashboard")
st.caption("Dedicated dashboard for instagram_usage_lifestyle.csv")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("Instagram CSV path", value="instagram_usage_lifestyle.csv")
    refresh = st.button("Refresh", use_container_width=True)

if "ig_run" not in st.session_state:
    st.session_state.ig_run = True
if refresh:
    st.session_state.ig_run = True

if st.session_state.ig_run:
    try:
        df = load_instagram_data(data_path)

        if "ig_popup" not in st.session_state:
            st.session_state.ig_popup = False
        if not st.session_state.ig_popup:
            st.toast("Hi, I am your Instagram RAKA assistant. Ask me lifestyle and engagement questions.")
            st.session_state.ig_popup = True

        countries = ["All"] + (sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else [])
        genders = ["All"] + (sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else [])

        with st.sidebar:
            st.header("Filters")
            selected_country = st.selectbox("Country", options=countries, index=0)
            selected_gender = st.selectbox("Gender", options=genders, index=0)
            min_age = int(df["age"].min()) if "age" in df.columns else 0
            max_age = int(df["age"].max()) if "age" in df.columns else 100
            age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
            metric_focus = st.selectbox(
                "Metric Focus",
                options=["user_engagement_score", "daily_active_minutes_instagram", "average_session_length_minutes", "followers_count", "notification_response_rate"],
                index=0,
            )

        view = df.copy()
        if selected_country != "All" and "country" in view.columns:
            view = view[view["country"] == selected_country].copy()
        if selected_gender != "All" and "gender" in view.columns:
            view = view[view["gender"] == selected_gender].copy()
        if "age" in view.columns:
            view = view[(view["age"] >= age_range[0]) & (view["age"] <= age_range[1])].copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(view):,}")
        m2.metric("Countries", view["country"].nunique() if "country" in view.columns else 0)
        m3.metric("Avg Engagement", f"{view['user_engagement_score'].mean():.2f}" if "user_engagement_score" in view.columns and not view.empty else "N/A")
        m4.metric("Avg Daily Minutes", f"{view['daily_active_minutes_instagram'].mean():.2f}" if "daily_active_minutes_instagram" in view.columns and not view.empty else "N/A")

        left, right = st.columns([1.25, 1.0])

        with left:
            if metric_focus in view.columns:
                fig_metric = px.histogram(view, x=metric_focus, nbins=45, title=f"Distribution: {metric_focus}")
                st.plotly_chart(fig_metric, use_container_width=True)

            if {"country", "user_engagement_score"}.issubset(set(view.columns)):
                by_country = (
                    view.groupby("country", as_index=False)["user_engagement_score"]
                    .mean()
                    .sort_values("user_engagement_score", ascending=False)
                    .head(15)
                )
                fig_country = px.bar(by_country, x="country", y="user_engagement_score", title="Top Countries by Average Engagement")
                st.plotly_chart(fig_country, use_container_width=True)

            if {"preferred_content_theme", "user_engagement_score"}.issubset(set(view.columns)):
                by_theme = (
                    view.groupby("preferred_content_theme", as_index=False)["user_engagement_score"]
                    .mean()
                    .sort_values("user_engagement_score", ascending=False)
                    .head(12)
                )
                fig_theme = px.bar(by_theme, x="preferred_content_theme", y="user_engagement_score", title="Engagement by Content Theme")
                st.plotly_chart(fig_theme, use_container_width=True)

        with right:
            st.markdown("### Instagram Chatbot - RAKA")
            st.caption("Ask human-style questions or use quick buttons.")

            if "ig_msgs" not in st.session_state:
                st.session_state.ig_msgs = [
                    {"role": "assistant", "content": "Hi, I am RAKA Instagram assistant. Ask me engagement, country insights, themes, followers, and data quality."}
                ]

            q1, q2 = st.columns(2)
            q3, q4 = st.columns(2)
            quick_prompt = ""
            if q1.button("Summary", use_container_width=True):
                quick_prompt = "summary"
            if q2.button("Engagement", use_container_width=True):
                quick_prompt = "engagement"
            if q3.button("Top Country", use_container_width=True):
                quick_prompt = "top country"
            if q4.button("Theme", use_container_width=True):
                quick_prompt = "top theme"

            q5, q6 = st.columns(2)
            if q5.button("Followers", use_container_width=True):
                quick_prompt = "followers"
            if q6.button("Correlation", use_container_width=True):
                quick_prompt = "correlation"

            for msg in st.session_state.ig_msgs:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            user_text = st.chat_input("Ask Instagram RAKA...")
            if quick_prompt and not user_text:
                user_text = quick_prompt

            if user_text:
                st.session_state.ig_msgs.append({"role": "user", "content": user_text})
                with st.chat_message("user"):
                    st.write(user_text)

                reply = instagram_chat_reply(user_text, view, selected_country, selected_gender)
                st.session_state.ig_msgs.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)

    except Exception as exc:
        st.error(f"Could not load Instagram dashboard: {exc}")
