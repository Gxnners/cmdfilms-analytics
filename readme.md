# CMD Films Analytics

An interactive analytics dashboard built around public YouTube data from CMD Films' three channels: **ChrisMD**, **Chris Dixon**, and **ChrisMD but shorter**.

The dashboard ingests every video published across the three channels (~2,800 videos), engineers performance and title-based features, and surfaces patterns that can inform content planning decisions. It also includes a trained machine learning model that predicts view counts for hypothetical videos based on length, upload timing, and title characteristics.

**Live demo:** [cmdfilms-analytics.streamlit.app](https://cmdfilms-analytics.streamlit.app)

## What it does

- **Overview** — channel-by-channel summary, view trends over time, and a hit-rate consistency tracker showing the percentage of long-form videos clearing the 1M view threshold each year
- **What Works** — optimal video length analysis, best upload day of week, and the measured impact of title features (VS, ft., CAPS, money references, questions, numbers)
- **View Predictor** — a Random Forest model trained on 1,100+ long-form videos. Enter a hypothetical title, length, upload day and hour, and get a predicted view range with a distribution plot
- **Keyword Insights** — top and bottom performing keywords by median views, with a search tool to look up any keyword's historical performance

## Tech stack

- **Python** — pandas, NumPy, scikit-learn
- **Data source** — YouTube Data API v3
- **Frontend** — Streamlit
- **Visualisation** — Plotly
- **Model** — Random Forest Regressor (300 trees, log-transformed target)

## Project structure

cmdfilms-analytics/
├── src/
│   ├── config.py         # Channel handles
│   ├── fetch_data.py     # YouTube API ingestion
│   ├── analyse.py        # CLI summary statistics
│   └── dashboard.py      # Streamlit app
├── data/
│   └── cmd_videos.csv    # Fetched dataset
├── requirements.txt
└── .env                  # YOUTUBE_API_KEY (not committed)


## Running locally

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add a `.env` file with your YouTube Data API v3 key:



Fetch fresh data, then run the dashboard:
```bash
py src/fetch_data.py
streamlit run src/dashboard.py
```

## Author

Miles McGarty — final-year BSc Computer Science, University of Reading. Built as a demonstration project alongside a final-year dissertation on predictive analysis of football player performance using LSTM networks and xG modelling