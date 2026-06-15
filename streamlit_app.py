# -*- coding: utf-8 -*-

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="F1 2025 Driver Performance Dashboard",
    page_icon="🏎️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 1) 데이터 로딩
# Streamlit은 위젯을 조작할 때마다 스크립트를 다시 실행합니다.
# @st.cache_data를 사용하면 같은 데이터프레임을 매번 다시 만들지 않아도 됩니다.

@st.cache_data
def load_f1_2025_data():
    """2025 F1 드라이버 순위 데이터 로드"""
    data = [
        {"Position": 1, "Driver": "Lando Norris", "Code": "NOR", "Nationality": "GBR", "Team": "McLaren", "Driver Points": 423},
        {"Position": 2, "Driver": "Max Verstappen", "Code": "VER", "Nationality": "NED", "Team": "Red Bull Racing", "Driver Points": 421},
        {"Position": 3, "Driver": "Oscar Piastri", "Code": "PIA", "Nationality": "AUS", "Team": "McLaren", "Driver Points": 410},
        {"Position": 4, "Driver": "George Russell", "Code": "RUS", "Nationality": "GBR", "Team": "Mercedes", "Driver Points": 319},
        {"Position": 5, "Driver": "Charles Leclerc", "Code": "LEC", "Nationality": "MON", "Team": "Ferrari", "Driver Points": 242},
        {"Position": 6, "Driver": "Lewis Hamilton", "Code": "HAM", "Nationality": "GBR", "Team": "Ferrari", "Driver Points": 156},
        {"Position": 7, "Driver": "Kimi Antonelli", "Code": "ANT", "Nationality": "ITA", "Team": "Mercedes", "Driver Points": 150},
        {"Position": 8, "Driver": "Alexander Albon", "Code": "ALB", "Nationality": "THA", "Team": "Williams", "Driver Points": 73},
        {"Position": 9, "Driver": "Carlos Sainz", "Code": "SAI", "Nationality": "ESP", "Team": "Williams", "Driver Points": 64},
        {"Position": 10, "Driver": "Fernando Alonso", "Code": "ALO", "Nationality": "ESP", "Team": "Aston Martin", "Driver Points": 56},
        {"Position": 11, "Driver": "Nico Hulkenberg", "Code": "HUL", "Nationality": "GER", "Team": "Kick Sauber", "Driver Points": 51},
        {"Position": 12, "Driver": "Isack Hadjar", "Code": "HAD", "Nationality": "FRA", "Team": "Racing Bulls", "Driver Points": 51},
        {"Position": 13, "Driver": "Oliver Bearman", "Code": "BEA", "Nationality": "GBR", "Team": "Haas F1 Team", "Driver Points": 41},
        {"Position": 14, "Driver": "Liam Lawson", "Code": "LAW", "Nationality": "NZL", "Team": "Racing Bulls", "Driver Points": 38},
        {"Position": 15, "Driver": "Esteban Ocon", "Code": "OCO", "Nationality": "FRA", "Team": "Haas F1 Team", "Driver Points": 38},
        {"Position": 16, "Driver": "Lance Stroll", "Code": "STR", "Nationality": "CAN", "Team": "Aston Martin", "Driver Points": 33},
        {"Position": 17, "Driver": "Yuki Tsunoda", "Code": "TSU", "Nationality": "JPN", "Team": "Red Bull Racing", "Driver Points": 33},
        {"Position": 18, "Driver": "Pierre Gasly", "Code": "GAS", "Nationality": "FRA", "Team": "Alpine", "Driver Points": 22},
        {"Position": 19, "Driver": "Gabriel Bortoleto", "Code": "BOR", "Nationality": "BRA", "Team": "Kick Sauber", "Driver Points": 19},
        {"Position": 20, "Driver": "Franco Colapinto", "Code": "COL", "Nationality": "ARG", "Team": "Alpine", "Driver Points": 0},
        {"Position": 21, "Driver": "Jack Doohan", "Code": "DOO", "Nationality": "AUS", "Team": "Alpine", "Driver Points": 0},
    ]
    df = pd.DataFrame(data)
    df["Season"] = 2025
    df["Team Points Sum"] = df.groupby("Team")["Driver Points"].transform("sum")
    df["Points Share in Team"] = np.where(
        df["Team Points Sum"] > 0,
        df["Driver Points"] / df["Team Points Sum"],
        0,
    )
    df["Top 10"] = (df["Position"] <= 10).astype(int)
    df["Performance Tier"] = pd.cut(
        df["Position"],
        bins=[0, 3, 7, 10, 21],
        labels=["Title Contender", "Front Runner", "Points Finisher", "Lower Midfield"],
    ).astype(str)
    df["profile_text"] = (
        df["Driver"] + " " + df["Code"] + " " + df["Team"] + " " + df["Nationality"]
    )
    return df


# -----------------------------------------------------------------------------
# 2) 모델 로딩 및 예측 함수
# 모델 객체는 무거운 리소스이므로 @st.cache_resource를 사용합니다.
# 예측 결과는 같은 입력에 대해 재사용 가능하므로 @st.cache_data를 사용합니다.

@st.cache_resource
def load_model():
    """F1 드라이버 텍스트 프로필을 기반으로 Top 10 여부를 예측하는 더미 모델"""
    train_df = load_f1_2025_data()
    model = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("clf", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )
    model.fit(train_df["profile_text"], train_df["Top 10"])
    return model


@st.cache_data
def predict_texts(texts):
    """입력 텍스트 리스트의 Top 10 가능성을 예측"""
    model = load_model()
    preds = model.predict(texts)
    probs = model.predict_proba(texts)[:, 1]
    return pd.DataFrame(
        {
            "text": texts,
            "predicted_label": preds,
            "top10_probability": probs,
            "prediction": np.where(preds == 1, "Top 10 Candidate", "Outside Top 10"),
        }
    )


# -----------------------------------------------------------------------------
# 3) 분석 보조 함수

TITLE_COL = "Driver"
TEAM_COL = "Team"
POINTS_COL = "Driver Points"
POSITION_COL = "Position"
TEXT_WIDTH = 760
CHART_HEIGHT = 460


def perform_linear_regression(dataframe, x_col, y_col, sigma_threshold):
    clean_df = dataframe[[TITLE_COL, TEAM_COL, x_col, y_col]].dropna().copy()

    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    if len(clean_df) < 2 or np.std(x) == 0:
        clean_df["Predicted"] = np.nan
        clean_df["Upper Bound"] = np.nan
        clean_df["Lower Bound"] = np.nan
        clean_df["Status"] = "In Range"
        return clean_df

    slope, intercept = np.polyfit(x, y, 1)
    predictions = slope * x + intercept
    residuals = y - predictions
    std_dev = np.std(residuals)

    clean_df["Predicted"] = predictions
    clean_df["Upper Bound"] = predictions + sigma_threshold * std_dev
    clean_df["Lower Bound"] = predictions - sigma_threshold * std_dev
    clean_df["Status"] = np.where(
        (clean_df[y_col] > clean_df["Upper Bound"])
        | (clean_df[y_col] < clean_df["Lower Bound"]),
        "Outlier",
        "In Range",
    )
    return clean_df


def draw_histogram(dataframe, metric_name):
    chart = (
        alt.Chart(dataframe)
        .mark_bar(binSpacing=0)
        .encode(
            alt.X(metric_name, type="quantitative", bin=alt.Bin(maxbins=15)),
            alt.Y("count()", title=None),
            tooltip=["count()"],
        )
        .properties(height=180)
    )
    st.altair_chart(chart, use_container_width=True)


# -----------------------------------------------------------------------------
# 4) 앱 본문

df = load_f1_2025_data()

COLUMN_CONFIG = {
    "Position": st.column_config.NumberColumn("Rank", format="%d"),
    "Driver": st.column_config.TextColumn("Driver", pinned=True),
    "Code": st.column_config.TextColumn("Code"),
    "Nationality": st.column_config.TextColumn("Nationality"),
    "Team": st.column_config.TextColumn("Team"),
    "Season": st.column_config.NumberColumn("Season", format="%d"),
    "Driver Points": st.column_config.ProgressColumn(
        "Driver Points",
        min_value=0,
        max_value=int(df["Driver Points"].max()),
        format="%d",
    ),
    "Team Points Sum": st.column_config.NumberColumn("Team Points Sum", format="%d"),
    "Points Share in Team": st.column_config.ProgressColumn(
        "Points Share in Team",
        min_value=0,
        max_value=1,
        format="%.1%",
    ),
    "Top 10": st.column_config.CheckboxColumn("Top 10"),
}

st.title("F1 2025 Driver Performance Dashboard")
st.write(
    "2025 F1 드라이버 순위 데이터를 기반으로 포인트, 팀 내 기여도, 간단한 예측 모델을 실습하는 Streamlit 앱입니다."
)

st.info(
    "모델 부분은 실전 성능 목적이 아니라, Streamlit 캐싱과 scikit-learn Pipeline 연결 방식을 보여주기 위한 더미 예측 예시입니다."
)

st.divider()

# -----------------------------------------------------------------------------
# Part I

st.header("Part I: 2025 Driver Standings")
st.write("2025 시즌 드라이버 포인트와 순위의 관계를 확인합니다.")

cols = st.columns([0.7, 0.3])

with cols[0]:
    scatter = (
        alt.Chart(df)
        .mark_point(filled=True, size=110, opacity=0.75)
        .encode(
            alt.X("Driver Points:Q", title="Driver Points"),
            alt.Y("Position:Q", title="Championship Position", sort="descending"),
            alt.Color("Team:N"),
            alt.Shape("Performance Tier:N"),
            tooltip=["Position", "Driver", "Team", "Nationality", "Driver Points", "Performance Tier"],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(scatter, use_container_width=True)

with cols[1]:
    st.subheader("Distribution")
    draw_histogram(df, "Driver Points")
    draw_histogram(df, "Position")

st.divider()

# -----------------------------------------------------------------------------
# Part II

st.header("Part II: Team Performance")
st.write("팀별 드라이버 포인트 합계와 팀 내 기여도를 확인합니다.")

team_df = (
    df.groupby("Team", as_index=False)
    .agg(
        Drivers=("Driver", "count"),
        Team_Points_Sum=("Driver Points", "sum"),
        Best_Position=("Position", "min"),
        Average_Position=("Position", "mean"),
    )
    .sort_values("Team_Points_Sum", ascending=False)
)

metric = st.selectbox(
    "팀별 비교 기준",
    ["Team_Points_Sum", "Best_Position", "Average_Position"],
)

team_chart = (
    alt.Chart(team_df)
    .mark_bar()
    .encode(
        alt.X(f"{metric}:Q", title=metric.replace("_", " ")),
        alt.Y("Team:N", sort="-x", title=None),
        tooltip=["Team", "Drivers", "Team_Points_Sum", "Best_Position", "Average_Position"],
    )
    .properties(height=360)
)
st.altair_chart(team_chart, use_container_width=True)
st.dataframe(team_df, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Part III

st.header("Part III: Linear Regression Practice")
st.write(
    "선택한 X축 지표로 Y축 지표를 단순 선형 회귀로 예측합니다. "
    "예측선에서 크게 벗어난 드라이버는 Outlier로 표시됩니다."
)

numeric_cols = [
    "Position",
    "Driver Points",
    "Team Points Sum",
    "Points Share in Team",
]

cols = st.columns(3)
with cols[0]:
    x_col = st.selectbox("X Axis", numeric_cols, index=1)
with cols[1]:
    y_col = st.selectbox("Y Axis", numeric_cols, index=0)
with cols[2]:
    sigma_val = st.slider("Confidence interval (sigma)", 0.5, 4.0, 2.0, 0.1)

model_df = perform_linear_regression(df, x_col, y_col, sigma_val)

base = alt.Chart(model_df).encode(
    alt.X(x_col, type="quantitative", scale=alt.Scale(zero=False))
)

band = base.mark_area(opacity=0.12).encode(
    alt.Y("Lower Bound:Q"),
    alt.Y2("Upper Bound:Q"),
)
line = base.mark_line(size=3).encode(alt.Y("Predicted:Q"))
points = base.mark_point(filled=True, size=90, opacity=0.75).encode(
    alt.Y(y_col, type="quantitative"),
    alt.Color("Status:N"),
    alt.Shape("Status:N"),
    tooltip=[TITLE_COL, TEAM_COL, x_col, y_col, "Status"],
)

st.altair_chart((band + points + line).properties(height=CHART_HEIGHT), use_container_width=True)

outliers = model_df[model_df["Status"] == "Outlier"][[TITLE_COL, TEAM_COL, x_col, y_col, "Status"]]
col1, col2 = st.columns([0.25, 0.75])
with col1:
    st.metric("Number of outliers", len(outliers))
with col2:
    st.dataframe(outliers, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Part IV

st.header("Part IV: Cached Model Prediction")
st.write(
    "아래 입력값은 `Driver Team Nationality` 형태의 텍스트입니다. "
    "`@st.cache_resource`로 모델을 한 번만 학습하고, `@st.cache_data`로 같은 입력의 예측 결과를 재사용합니다."
)

sample_text = "Lando Norris McLaren GBR\nMax Verstappen Red Bull Racing NED\nFranco Colapinto Alpine ARG"
user_text = st.text_area(
    "Prediction input",
    value=sample_text,
    height=120,
)
texts = [line.strip() for line in user_text.splitlines() if line.strip()]

if texts:
    prediction_df = predict_texts(texts)
    st.dataframe(
        prediction_df,
        use_container_width=True,
        column_config={
            "top10_probability": st.column_config.ProgressColumn(
                "Top 10 Probability",
                min_value=0,
                max_value=1,
                format="%.1%",
            )
        },
    )
else:
    st.warning("예측할 텍스트를 한 줄 이상 입력하세요.")

st.divider()

# -----------------------------------------------------------------------------
# Part V

st.header("Part V: Browse the Full Dataset")

team_filter = st.multiselect(
    "Team filter",
    options=sorted(df["Team"].unique()),
    default=sorted(df["Team"].unique()),
)
search_text = st.text_input("Search driver name")

filtered_df = df[df["Team"].isin(team_filter)]
if search_text:
    filtered_df = filtered_df[
        filtered_df["Driver"].str.contains(search_text, case=False, na=False)
    ]

st.dataframe(
    filtered_df.drop(columns=["profile_text"]),
    use_container_width=True,
    height=420,
    column_config=COLUMN_CONFIG,
)
