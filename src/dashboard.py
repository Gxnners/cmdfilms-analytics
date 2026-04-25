import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import isodate
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="CMD Films Analytics", page_icon="⚽", layout="wide")

STOPWORDS = set("""
the a an of to in on for and or but with by from at as is are was were be been
being have has had do does did i you he she it we they me him her us them
this that these those my your his its our their what which who whom whose
not no yes if then so than too very can could should would will just now
""".split())

# ---------- DATA ----------

@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "cmd_videos.csv")
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["duration_sec"] = df["duration"].apply(lambda x: isodate.parse_duration(x).total_seconds())
    df["duration_min"] = df["duration_sec"] / 60
    df["year"] = df["published_at"].dt.year
    df["day_of_week"] = df["published_at"].dt.day_name()
    df["dow_num"] = df["published_at"].dt.dayofweek
    df["hour"] = df["published_at"].dt.hour
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(0, 1)
    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["has_caps_word"] = df["title"].str.contains(r"\b[A-Z]{2,}\b", regex=True).astype(int)
    df["has_question"] = df["title"].str.contains(r"\?").astype(int)
    df["has_vs"] = df["title"].str.contains(r"\bvs\.?\b|VS|Vs", regex=True).astype(int)
    df["has_ft"] = df["title"].str.contains(r"\bft\.?\b|feat", regex=True, case=False).astype(int)
    df["has_money"] = df["title"].str.contains(r"£|\$|€").astype(int)
    df["has_number"] = df["title"].str.contains(r"\d").astype(int)
    return df

@st.cache_data
def get_keyword_performance(df_long, min_occurrences=10):
    word_views = {}
    for _, row in df_long.iterrows():
        words = re.findall(r"\b[a-zA-Z]{3,}\b", row["title"].lower())
        words = set(w for w in words if w not in STOPWORDS)
        for w in words:
            word_views.setdefault(w, []).append(row["views"])
    rows = []
    overall_median = df_long["views"].median()
    for w, views in word_views.items():
        if len(views) >= min_occurrences:
            rows.append({
                "keyword": w,
                "occurrences": len(views),
                "median_views": int(np.median(views)),
                "vs_overall_median": np.median(views) / overall_median,
            })
    return pd.DataFrame(rows).sort_values("median_views", ascending=False)

@st.cache_resource
def train_model(df_long):
    feature_cols = [
        "duration_min", "dow_num", "hour", "year",
        "title_length", "title_word_count",
        "has_caps_word", "has_question", "has_vs", "has_ft", "has_money", "has_number",
    ]
    X = df_long[feature_cols].fillna(0)
    y = np.log1p(df_long["views"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=14)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(pred_test))
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return model, feature_cols, mae, importances

df = load_data()
long_form = df[df["duration_sec"] > 60].copy()
shorts = df[df["duration_sec"] <= 60].copy()
keyword_perf = get_keyword_performance(long_form)
model, feature_cols, mae, importances = train_model(long_form)

# ---------- SIDEBAR ----------

st.sidebar.title("⚽ CMD Films Analytics")
st.sidebar.markdown("Built by **Miles McGarty** — final-year Computer Science at the University of Reading.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Sections", [
    "📊 Overview",
    "🎯 What Works",
    "🔮 View Predictor",
    "🔍 Keyword Insights",
])

# ---------- OVERVIEW ----------

if page == "📊 Overview":
    st.title("📊 CMD Films Analytics")
    st.markdown("Public analysis across **ChrisMD**, **Chris Dixon**, and **ChrisMD but shorter**.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Videos", f"{len(df):,}")
    c2.metric("Total Views", f"{df['views'].sum() / 1e9:.2f}B")
    c3.metric("Long-form", f"{len(long_form):,}")
    c4.metric("Shorts", f"{len(shorts):,}")

    st.markdown("### Per-channel summary")
    summary = df.groupby("channel").agg(
        videos=("video_id", "count"),
        total_views=("views", "sum"),
        avg_views=("views", "mean"),
        median_views=("views", "median"),
    ).round(0).astype(int)
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Long-form: views per video over time")
    yearly = long_form.groupby("year").agg(
        videos=("video_id", "count"),
        avg_views=("views", "mean"),
        median_views=("views", "median"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly["year"], y=yearly["videos"], name="Videos uploaded", yaxis="y2", opacity=0.3))
    fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["median_views"], name="Median views", mode="lines+markers", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["avg_views"], name="Mean views", mode="lines+markers", line=dict(dash="dash")))
    fig.update_layout(
        yaxis=dict(title="Views per video"),
        yaxis2=dict(title="Videos uploaded", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Watch the gap between mean and median.** When it widens (post-2020), it means a small number of hits "
        "prop up the average while most uploads underperform — the channel has become hit-driven and inconsistent."
    )

    st.markdown("### Hit-rate consistency (% of long-form videos hitting 1M+ views)")
    long_form["hit_1m"] = (long_form["views"] >= 1_000_000).astype(int)
    hit_rate = long_form.groupby("year")["hit_1m"].mean().reset_index()
    hit_rate["hit_rate_pct"] = hit_rate["hit_1m"] * 100
    fig2 = px.bar(hit_rate, x="year", y="hit_rate_pct", labels={"hit_rate_pct": "% videos hitting 1M+ views"})
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- WHAT WORKS ----------

elif page == "🎯 What Works":
    st.title("🎯 What Works")
    st.markdown("Patterns identified across long-form ChrisMD content.")

    st.markdown("### Optimal video length")
    lf = long_form.copy()
    lf["length_bucket"] = pd.cut(
        lf["duration_min"],
        bins=[0, 5, 10, 15, 20, 30, 60, 999],
        labels=["<5m","5-10m","10-15m","15-20m","20-30m","30-60m","60m+"],
    )
    length_stats = lf.groupby("length_bucket", observed=True).agg(
        videos=("video_id", "count"),
        median_views=("views", "median"),
    ).reset_index()
    fig = px.bar(length_stats, x="length_bucket", y="median_views", text="videos",
                 labels={"median_views": "Median views", "length_bucket": "Video length"})
    fig.update_traces(texttemplate="%{text} videos", textposition="outside")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**The 20–30 minute bucket is the clear sweet spot** — median views far exceed any other length. "
        "Yet under-5-minute videos make up the largest share of long-form output despite dramatically lower performance."
    )

    st.markdown("### Best day to upload")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_stats = long_form.groupby("day_of_week").agg(
        videos=("video_id", "count"),
        median_views=("views", "median"),
    ).reindex(dow_order).reset_index()
    fig = px.bar(dow_stats, x="day_of_week", y="median_views", text="videos",
                 labels={"median_views": "Median views", "day_of_week": ""})
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Title feature impact")
    feature_impact = []
    for col, label in [
        ("has_vs", "Contains 'VS'"),
        ("has_ft", "Contains 'ft' / 'feat'"),
        ("has_caps_word", "Has CAPS word"),
        ("has_question", "Contains '?'"),
        ("has_money", "Mentions money (£/$/€)"),
        ("has_number", "Contains a number"),
    ]:
        with_feat = long_form[long_form[col] == 1]["views"].median()
        without_feat = long_form[long_form[col] == 0]["views"].median()
        feature_impact.append({
            "Feature": label,
            "Median views with": int(with_feat) if not pd.isna(with_feat) else 0,
            "Median views without": int(without_feat) if not pd.isna(without_feat) else 0,
            "Lift": f"{(with_feat / without_feat - 1) * 100:+.0f}%" if without_feat and without_feat > 0 else "n/a",
        })
    st.dataframe(pd.DataFrame(feature_impact), use_container_width=True, hide_index=True)

# ---------- PREDICTOR ----------

elif page == "🔮 View Predictor":
    st.title("🔮 View Predictor")
    st.markdown(
        f"Random Forest trained on **{len(long_form):,}** long-form ChrisMD videos. "
        "Type a hypothetical video and get a predicted view range."
    )
    st.caption(f"Test-set MAE ≈ {mae:,.0f} views")

    col1, col2 = st.columns([2, 1])
    with col1:
        title_input = st.text_input("Video title", "Sidemen vs ChrisMD: The Ultimate Football Challenge")
        duration = st.slider("Duration (minutes)", 1, 60, 22)
    with col2:
        day = st.selectbox("Upload day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=6)
        hour = st.slider("Upload hour (24h)", 0, 23, 17)

    if st.button("Predict views", type="primary"):
        dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
        features = {
            "duration_min": duration,
            "dow_num": dow_map[day],
            "hour": hour,
            "year": 2026,
            "title_length": len(title_input),
            "title_word_count": len(title_input.split()),
            "has_caps_word": int(bool(re.search(r"\b[A-Z]{2,}\b", title_input))),
            "has_question": int("?" in title_input),
            "has_vs": int(bool(re.search(r"\bvs\.?\b|VS|Vs", title_input))),
            "has_ft": int(bool(re.search(r"\bft\.?\b|feat", title_input, re.IGNORECASE))),
            "has_money": int(bool(re.search(r"£|\$|€", title_input))),
            "has_number": int(bool(re.search(r"\d", title_input))),
        }
        X_pred = pd.DataFrame([features])[feature_cols]
        log_preds = np.array([t.predict(X_pred)[0] for t in model.estimators_])
        preds = np.expm1(log_preds)

        st.markdown("### Predicted views")
        c1, c2, c3 = st.columns(3)
        c1.metric("Low (10th percentile)", f"{int(np.percentile(preds, 10)):,}")
        c2.metric("Median estimate", f"{int(np.median(preds)):,}")
        c3.metric("High (90th percentile)", f"{int(np.percentile(preds, 90)):,}")

        fig = px.histogram(x=preds, nbins=30)
        fig.update_xaxes(title="Predicted views")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What drives the prediction")
    fig = px.bar(importances, x="importance", y="feature", orientation="h",
                 labels={"importance": "Feature importance", "feature": ""})
    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------- KEYWORDS ----------

elif page == "🔍 Keyword Insights":
    st.title("🔍 Keyword Insights")
    st.markdown("Which words in titles correlate with strong long-form performance?")

    overall_median = long_form["views"].median()
    st.caption(f"Overall long-form median: {overall_median:,.0f} views")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top performing keywords")
        top_kw = keyword_perf.head(15).copy()
        top_kw["lift_pct"] = ((top_kw["vs_overall_median"] - 1) * 100).round(0).astype(int)
        st.dataframe(top_kw[["keyword", "occurrences", "median_views", "lift_pct"]],
                     hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Worst performing keywords")
        bot_kw = keyword_perf.tail(15).iloc[::-1].copy()
        bot_kw["lift_pct"] = ((bot_kw["vs_overall_median"] - 1) * 100).round(0).astype(int)
        st.dataframe(bot_kw[["keyword", "occurrences", "median_views", "lift_pct"]],
                     hide_index=True, use_container_width=True)

    st.markdown("### Search any keyword")
    search = st.text_input("Keyword").lower().strip()
    if search:
        matches = long_form[long_form["title"].str.lower().str.contains(search, na=False, regex=False)]
        if len(matches) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Videos containing this", len(matches))
            c2.metric("Median views", f"{int(matches['views'].median()):,}")
            c3.metric("vs overall", f"{(matches['views'].median() / overall_median - 1) * 100:+.0f}%")
            st.dataframe(matches[["title", "views", "published_at"]].sort_values("views", ascending=False).head(10),
                         hide_index=True, use_container_width=True)
        else:
            st.warning("No videos found.")