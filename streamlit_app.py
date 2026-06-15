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
# Data

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
    df["Points Share in Team"] = np.where(df["Team Points Sum"] > 0, df["Driver Points"] / df["Team Points Sum"], 0)
    df["Top 10"] = (df["Position"] <= 10).astype(int)
    df["Performance Tier"] = pd.cut(
        df["Position"],
        bins=[0, 3, 7, 10, 21],
        labels=["Title Contender", "Front Runner", "Points Finisher", "Lower Midfield"],
    ).astype(str)
    df["profile_text"] = df["Driver"] + " " + df["Code"] + " " + df["Team"] + " " + df["Nationality"]
    return df


@st.cache_data
def load_f1_2025_race_results():
    """그랑프리별 포디움/폴/패스티스트 랩 데이터. 실습용 내장 데이터입니다."""
    race_data = [
        {"Round": 1, "Grand Prix": "Australian GP", "Circuit": "Albert Park", "Winner": "Lando Norris", "Second": "Max Verstappen", "Third": "George Russell", "Winning Team": "McLaren", "Pole Sitter": "Lando Norris", "Fastest Lap Driver": "Lando Norris"},
        {"Round": 2, "Grand Prix": "Chinese GP", "Circuit": "Shanghai", "Winner": "Oscar Piastri", "Second": "Lando Norris", "Third": "George Russell", "Winning Team": "McLaren", "Pole Sitter": "Oscar Piastri", "Fastest Lap Driver": "Lewis Hamilton"},
        {"Round": 3, "Grand Prix": "Japanese GP", "Circuit": "Suzuka", "Winner": "Max Verstappen", "Second": "Lando Norris", "Third": "Oscar Piastri", "Winning Team": "Red Bull Racing", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Kimi Antonelli"},
        {"Round": 4, "Grand Prix": "Bahrain GP", "Circuit": "Sakhir", "Winner": "Oscar Piastri", "Second": "George Russell", "Third": "Lando Norris", "Winning Team": "McLaren", "Pole Sitter": "Oscar Piastri", "Fastest Lap Driver": "Oscar Piastri"},
        {"Round": 5, "Grand Prix": "Saudi Arabian GP", "Circuit": "Jeddah", "Winner": "Oscar Piastri", "Second": "Max Verstappen", "Third": "Charles Leclerc", "Winning Team": "McLaren", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Lando Norris"},
        {"Round": 6, "Grand Prix": "Miami GP", "Circuit": "Miami", "Winner": "Oscar Piastri", "Second": "Lando Norris", "Third": "George Russell", "Winning Team": "McLaren", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Oscar Piastri"},
        {"Round": 7, "Grand Prix": "Emilia Romagna GP", "Circuit": "Imola", "Winner": "Max Verstappen", "Second": "Lando Norris", "Third": "Oscar Piastri", "Winning Team": "Red Bull Racing", "Pole Sitter": "Oscar Piastri", "Fastest Lap Driver": "Max Verstappen"},
        {"Round": 8, "Grand Prix": "Monaco GP", "Circuit": "Monaco", "Winner": "Lando Norris", "Second": "Charles Leclerc", "Third": "Oscar Piastri", "Winning Team": "McLaren", "Pole Sitter": "Lando Norris", "Fastest Lap Driver": "Lewis Hamilton"},
        {"Round": 9, "Grand Prix": "Spanish GP", "Circuit": "Barcelona", "Winner": "Oscar Piastri", "Second": "Lando Norris", "Third": "Charles Leclerc", "Winning Team": "McLaren", "Pole Sitter": "Oscar Piastri", "Fastest Lap Driver": "Lando Norris"},
        {"Round": 10, "Grand Prix": "Canadian GP", "Circuit": "Montreal", "Winner": "George Russell", "Second": "Max Verstappen", "Third": "Kimi Antonelli", "Winning Team": "Mercedes", "Pole Sitter": "George Russell", "Fastest Lap Driver": "George Russell"},
        {"Round": 11, "Grand Prix": "Austrian GP", "Circuit": "Red Bull Ring", "Winner": "Lando Norris", "Second": "Oscar Piastri", "Third": "Charles Leclerc", "Winning Team": "McLaren", "Pole Sitter": "Lando Norris", "Fastest Lap Driver": "Oscar Piastri"},
        {"Round": 12, "Grand Prix": "British GP", "Circuit": "Silverstone", "Winner": "Lando Norris", "Second": "Oscar Piastri", "Third": "Nico Hulkenberg", "Winning Team": "McLaren", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Lando Norris"},
        {"Round": 13, "Grand Prix": "Belgian GP", "Circuit": "Spa-Francorchamps", "Winner": "Oscar Piastri", "Second": "Lando Norris", "Third": "Charles Leclerc", "Winning Team": "McLaren", "Pole Sitter": "Lando Norris", "Fastest Lap Driver": "Oscar Piastri"},
        {"Round": 14, "Grand Prix": "Hungarian GP", "Circuit": "Hungaroring", "Winner": "Lando Norris", "Second": "Oscar Piastri", "Third": "George Russell", "Winning Team": "McLaren", "Pole Sitter": "Charles Leclerc", "Fastest Lap Driver": "George Russell"},
        {"Round": 15, "Grand Prix": "Dutch GP", "Circuit": "Zandvoort", "Winner": "Oscar Piastri", "Second": "Max Verstappen", "Third": "Isack Hadjar", "Winning Team": "McLaren", "Pole Sitter": "Oscar Piastri", "Fastest Lap Driver": "Lando Norris"},
        {"Round": 16, "Grand Prix": "Italian GP", "Circuit": "Monza", "Winner": "Max Verstappen", "Second": "Lando Norris", "Third": "Oscar Piastri", "Winning Team": "Red Bull Racing", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Lando Norris"},
        {"Round": 17, "Grand Prix": "Azerbaijan GP", "Circuit": "Baku", "Winner": "Max Verstappen", "Second": "George Russell", "Third": "Carlos Sainz", "Winning Team": "Red Bull Racing", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Max Verstappen"},
        {"Round": 18, "Grand Prix": "Singapore GP", "Circuit": "Marina Bay", "Winner": "George Russell", "Second": "Max Verstappen", "Third": "Lando Norris", "Winning Team": "Mercedes", "Pole Sitter": "George Russell", "Fastest Lap Driver": "Fernando Alonso"},
        {"Round": 19, "Grand Prix": "United States GP", "Circuit": "COTA", "Winner": "Max Verstappen", "Second": "Lando Norris", "Third": "Charles Leclerc", "Winning Team": "Red Bull Racing", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Max Verstappen"},
        {"Round": 20, "Grand Prix": "Mexico City GP", "Circuit": "Mexico City", "Winner": "Lando Norris", "Second": "Charles Leclerc", "Third": "Max Verstappen", "Winning Team": "McLaren", "Pole Sitter": "Lando Norris", "Fastest Lap Driver": "Lewis Hamilton"},
        {"Round": 21, "Grand Prix": "São Paulo GP", "Circuit": "Interlagos", "Winner": "Lando Norris", "Second": "Kimi Antonelli", "Third": "Max Verstappen", "Winning Team": "McLaren", "Pole Sitter": "Lando Norris", "Fastest Lap Driver": "Max Verstappen"},
        {"Round": 22, "Grand Prix": "Las Vegas GP", "Circuit": "Las Vegas", "Winner": "Max Verstappen", "Second": "George Russell", "Third": "Kimi Antonelli", "Winning Team": "Red Bull Racing", "Pole Sitter": "Lando Norris", "Fastest Lap Driver": "Kimi Antonelli"},
        {"Round": 23, "Grand Prix": "Qatar GP", "Circuit": "Lusail", "Winner": "Max Verstappen", "Second": "Oscar Piastri", "Third": "Carlos Sainz", "Winning Team": "Red Bull Racing", "Pole Sitter": "Oscar Piastri", "Fastest Lap Driver": "Max Verstappen"},
        {"Round": 24, "Grand Prix": "Abu Dhabi GP", "Circuit": "Yas Marina", "Winner": "Max Verstappen", "Second": "Oscar Piastri", "Third": "Lando Norris", "Winning Team": "Red Bull Racing", "Pole Sitter": "Max Verstappen", "Fastest Lap Driver": "Max Verstappen"},
    ]
    return pd.DataFrame(race_data)


@st.cache_data
def build_full_race_classification(driver_df, race_df):
    """각 그랑프리별 전체 선수 순위 생성.

    1~3위는 race_df의 포디움 데이터를 고정하고,
    4위 이하는 드라이버 챔피언십 순위와 라운드 번호를 이용해 회전 배치한 실습용 데이터입니다.
    실제 공식 전체 분류표를 쓰려면 이 함수만 CSV/API 로딩으로 교체하면 됩니다.
    """
    point_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    drivers = driver_df[["Driver", "Code", "Team", "Nationality", "Position"]].copy()
    rows = []

    for _, race in race_df.iterrows():
        podium = [race["Winner"], race["Second"], race["Third"]]
        others = drivers[~drivers["Driver"].isin(podium)].sort_values("Position").reset_index(drop=True)

        shift = int(race["Round"] - 1) % len(others)
        others = pd.concat([others.iloc[shift:], others.iloc[:shift]], ignore_index=True)

        ordered_names = podium + others["Driver"].tolist()
        for finish_pos, driver_name in enumerate(ordered_names, start=1):
            driver_row = drivers[drivers["Driver"] == driver_name].iloc[0]
            rows.append(
                {
                    "Round": race["Round"],
                    "Grand Prix": race["Grand Prix"],
                    "Circuit": race["Circuit"],
                    "Finish Position": finish_pos,
                    "Driver": driver_row["Driver"],
                    "Code": driver_row["Code"],
                    "Team": driver_row["Team"],
                    "Nationality": driver_row["Nationality"],
                    "Race Points": point_map.get(finish_pos, 0),
                    "Status": "Classified",
                    "Podium": finish_pos <= 3,
                    "Points Finish": finish_pos <= 10,
                }
            )

    return pd.DataFrame(rows)


@st.cache_data
def build_driver_race_summary(classification_df, race_df):
    """전체 순위표를 드라이버 단위 요약으로 변환"""
    base = (
        classification_df.groupby("Driver", as_index=False)
        .agg(
            Starts=("Round", "count"),
            Average_Finish=("Finish Position", "mean"),
            Best_Finish=("Finish Position", "min"),
            Race_Points=("Race Points", "sum"),
            Podiums=("Podium", "sum"),
            Points_Finishes=("Points Finish", "sum"),
        )
    )
    wins = race_df["Winner"].value_counts().rename_axis("Driver").reset_index(name="Win")
    poles = race_df["Pole Sitter"].value_counts().rename_axis("Driver").reset_index(name="Pole")
    fastest = race_df["Fastest Lap Driver"].value_counts().rename_axis("Driver").reset_index(name="Fastest Lap")

    summary = base.merge(wins, on="Driver", how="left").merge(poles, on="Driver", how="left").merge(fastest, on="Driver", how="left")
    for col in ["Win", "Pole", "Fastest Lap"]:
        summary[col] = summary[col].fillna(0).astype(int)
    return summary.sort_values(["Race_Points", "Win", "Podiums"], ascending=False)


# -----------------------------------------------------------------------------
# Cached model

@st.cache_resource
def load_model():
    train_df = load_f1_2025_data()
    model = Pipeline([
        ("vec", CountVectorizer()),
        ("clf", LogisticRegression(max_iter=300, random_state=42)),
    ])
    model.fit(train_df["profile_text"], train_df["Top 10"])
    return model


@st.cache_data
def predict_texts(texts):
    model = load_model()
    preds = model.predict(texts)
    probs = model.predict_proba(texts)[:, 1]
    return pd.DataFrame({
        "text": texts,
        "predicted_label": preds,
        "top10_probability": probs,
        "prediction": np.where(preds == 1, "Top 10 Candidate", "Outside Top 10"),
    })


# -----------------------------------------------------------------------------
# Helper functions

TITLE_COL = "Driver"
TEAM_COL = "Team"
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
        (clean_df[y_col] > clean_df["Upper Bound"]) | (clean_df[y_col] < clean_df["Lower Bound"]),
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
# App body

df = load_f1_2025_data()
race_df = load_f1_2025_race_results()
classification_df = build_full_race_classification(df, race_df)
driver_race_summary_df = build_driver_race_summary(classification_df, race_df)

COLUMN_CONFIG = {
    "Position": st.column_config.NumberColumn("Rank", format="%d"),
    "Driver": st.column_config.TextColumn("Driver", pinned=True),
    "Code": st.column_config.TextColumn("Code"),
    "Nationality": st.column_config.TextColumn("Nationality"),
    "Team": st.column_config.TextColumn("Team"),
    "Season": st.column_config.NumberColumn("Season", format="%d"),
    "Driver Points": st.column_config.ProgressColumn("Driver Points", min_value=0, max_value=int(df["Driver Points"].max()), format="%d"),
    "Team Points Sum": st.column_config.NumberColumn("Team Points Sum", format="%d"),
    "Points Share in Team": st.column_config.ProgressColumn("Points Share in Team", min_value=0, max_value=1, format="%.1%"),
    "Top 10": st.column_config.CheckboxColumn("Top 10"),
}

st.title("F1 2025 Driver Performance Dashboard")
st.write("2025 F1 드라이버 순위, 그랑프리별 결과, 서킷별 전체 선수 순위를 확인하는 Streamlit 실습 앱입니다.")
st.info("그랑프리 포디움 데이터와 전체 순위 데이터는 실습용 내장 데이터입니다. 공식 전체 분류표가 필요하면 CSV/API 데이터로 교체하는 것을 권장합니다.")

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Total races", len(race_df))
with metric_cols[1]:
    st.metric("Different winners", race_df["Winner"].nunique())
with metric_cols[2]:
    st.metric("Most wins", race_df["Winner"].value_counts().index[0])
with metric_cols[3]:
    st.metric("Total classifications", len(classification_df))

st.divider()

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

# Part II
st.header("Part II: Team Performance")
st.write("팀별 드라이버 포인트 합계와 팀 내 기여도를 확인합니다.")

team_df = (
    df.groupby("Team", as_index=False)
    .agg(Drivers=("Driver", "count"), Team_Points_Sum=("Driver Points", "sum"), Best_Position=("Position", "min"), Average_Position=("Position", "mean"))
    .sort_values("Team_Points_Sum", ascending=False)
)
metric = st.selectbox("팀별 비교 기준", ["Team_Points_Sum", "Best_Position", "Average_Position"])
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

# Part III
st.header("Part III: Linear Regression Practice")
st.write("선택한 X축 지표로 Y축 지표를 단순 선형 회귀로 예측합니다.")

numeric_cols = ["Position", "Driver Points", "Team Points Sum", "Points Share in Team"]
cols = st.columns(3)
with cols[0]:
    x_col = st.selectbox("X Axis", numeric_cols, index=1)
with cols[1]:
    y_col = st.selectbox("Y Axis", numeric_cols, index=0)
with cols[2]:
    sigma_val = st.slider("Confidence interval (sigma)", 0.5, 4.0, 2.0, 0.1)

model_df = perform_linear_regression(df, x_col, y_col, sigma_val)
base = alt.Chart(model_df).encode(alt.X(x_col, type="quantitative", scale=alt.Scale(zero=False)))
band = base.mark_area(opacity=0.12).encode(alt.Y("Lower Bound:Q"), alt.Y2("Upper Bound:Q"))
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

# Part IV
st.header("Part IV: Cached Model Prediction")
st.write("`@st.cache_resource`로 모델을 한 번만 학습하고, `@st.cache_data`로 같은 입력의 예측 결과를 재사용합니다.")
sample_text = "Lando Norris McLaren GBR\nMax Verstappen Red Bull Racing NED\nFranco Colapinto Alpine ARG"
user_text = st.text_area("Prediction input", value=sample_text, height=120)
texts = [line.strip() for line in user_text.splitlines() if line.strip()]
if texts:
    prediction_df = predict_texts(texts)
    st.dataframe(
        prediction_df,
        use_container_width=True,
        column_config={"top10_probability": st.column_config.ProgressColumn("Top 10 Probability", min_value=0, max_value=1, format="%.1%")},
    )
else:
    st.warning("예측할 텍스트를 한 줄 이상 입력하세요.")

st.divider()

# Part V
st.header("Part V: Grand Prix Results")
st.write("그랑프리별 우승자, 포디움, 폴시터, 패스티스트 랩을 확인합니다.")
selected_gp = st.selectbox("Grand Prix summary", options=race_df["Grand Prix"].tolist())
selected_race = race_df[race_df["Grand Prix"] == selected_gp].iloc[0]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Winner", selected_race["Winner"])
with col2:
    st.metric("P2", selected_race["Second"])
with col3:
    st.metric("P3", selected_race["Third"])
with col4:
    st.metric("Pole", selected_race["Pole Sitter"])

race_chart = (
    alt.Chart(race_df)
    .mark_circle(size=160)
    .encode(
        alt.X("Round:O", title="Round"),
        alt.Y("Winner:N", title="Winner"),
        alt.Color("Winning Team:N"),
        tooltip=["Round", "Grand Prix", "Circuit", "Winner", "Second", "Third", "Winning Team", "Pole Sitter", "Fastest Lap Driver"],
    )
    .properties(height=420)
)
st.altair_chart(race_chart, use_container_width=True)
st.dataframe(race_df, use_container_width=True, height=380)

st.divider()

# Part VI - Full circuit classification
st.header("Part VI: Circuit Full Classification")
st.write("서킷별 모든 선수의 결승 순위를 확인합니다.")

circuit_options = (race_df["Round"].astype(str) + ". " + race_df["Grand Prix"] + " - " + race_df["Circuit"]).tolist()
selected_circuit_label = st.selectbox("Circuit / Grand Prix", circuit_options)
selected_round = int(selected_circuit_label.split(".")[0])
selected_classification = classification_df[classification_df["Round"] == selected_round].sort_values("Finish Position")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Classified drivers", len(selected_classification))
with m2:
    st.metric("Points finishers", int(selected_classification["Points Finish"].sum()))
with m3:
    st.metric("Winner", selected_classification.iloc[0]["Driver"])
with m4:
    st.metric("Winning team", selected_classification.iloc[0]["Team"])

classification_chart = (
    alt.Chart(selected_classification)
    .mark_bar()
    .encode(
        alt.X("Finish Position:Q", title="Finish Position", scale=alt.Scale(reverse=True)),
        alt.Y("Driver:N", sort="x", title=None),
        alt.Color("Team:N"),
        tooltip=["Finish Position", "Driver", "Team", "Race Points", "Status"],
    )
    .properties(height=560)
)
st.altair_chart(classification_chart, use_container_width=True)

st.dataframe(
    selected_classification[["Finish Position", "Driver", "Code", "Team", "Nationality", "Race Points", "Status", "Podium", "Points Finish"]],
    use_container_width=True,
    height=560,
    column_config={
        "Finish Position": st.column_config.NumberColumn("Pos", format="%d"),
        "Race Points": st.column_config.NumberColumn("Pts", format="%d"),
        "Podium": st.column_config.CheckboxColumn("Podium"),
        "Points Finish": st.column_config.CheckboxColumn("Points"),
    },
)

st.divider()

# Part VII
st.header("Part VII: Driver Race Summary")
st.write("전체 순위표를 드라이버 단위로 집계한 요약입니다.")
summary_metric = st.selectbox("Driver summary metric", ["Race_Points", "Win", "Podiums", "Points_Finishes", "Average_Finish", "Pole", "Fastest Lap"], index=0)
summary_sort = "-x" if summary_metric != "Average_Finish" else "x"
summary_chart = (
    alt.Chart(driver_race_summary_df)
    .mark_bar()
    .encode(
        alt.X(f"{summary_metric}:Q", title=summary_metric.replace("_", " ")),
        alt.Y("Driver:N", sort=summary_sort, title=None),
        tooltip=["Driver", "Starts", "Race_Points", "Win", "Podiums", "Points_Finishes", "Average_Finish", "Best_Finish", "Pole", "Fastest Lap"],
    )
    .properties(height=520)
)
st.altair_chart(summary_chart, use_container_width=True)
st.dataframe(driver_race_summary_df, use_container_width=True, height=420)

st.divider()

# Part VIII
st.header("Part VIII: Browse the Driver Dataset")
team_filter = st.multiselect("Team filter", options=sorted(df["Team"].unique()), default=sorted(df["Team"].unique()))
search_text = st.text_input("Search driver name")
filtered_df = df[df["Team"].isin(team_filter)]
if search_text:
    filtered_df = filtered_df[filtered_df["Driver"].str.contains(search_text, case=False, na=False)]

st.dataframe(filtered_df.drop(columns=["profile_text"]), use_container_width=True, height=420, column_config=COLUMN_CONFIG)
