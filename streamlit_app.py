# -*- coding: utf-8 -*-
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="F1 2025 Analytics Dashboard",
    page_icon="🏎️",
    layout="wide",
)

POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

@st.cache_data
def load_driver_standings():
    data = [
        (1, "Lando Norris", "NOR", "GBR", "McLaren", 423),
        (2, "Max Verstappen", "VER", "NED", "Red Bull Racing", 421),
        (3, "Oscar Piastri", "PIA", "AUS", "McLaren", 410),
        (4, "George Russell", "RUS", "GBR", "Mercedes", 319),
        (5, "Charles Leclerc", "LEC", "MON", "Ferrari", 242),
        (6, "Lewis Hamilton", "HAM", "GBR", "Ferrari", 156),
        (7, "Kimi Antonelli", "ANT", "ITA", "Mercedes", 150),
        (8, "Alexander Albon", "ALB", "THA", "Williams", 73),
        (9, "Carlos Sainz", "SAI", "ESP", "Williams", 64),
        (10, "Fernando Alonso", "ALO", "ESP", "Aston Martin", 56),
        (11, "Nico Hulkenberg", "HUL", "GER", "Kick Sauber", 51),
        (12, "Isack Hadjar", "HAD", "FRA", "Racing Bulls", 51),
        (13, "Oliver Bearman", "BEA", "GBR", "Haas F1 Team", 41),
        (14, "Liam Lawson", "LAW", "NZL", "Racing Bulls", 38),
        (15, "Esteban Ocon", "OCO", "FRA", "Haas F1 Team", 38),
        (16, "Lance Stroll", "STR", "CAN", "Aston Martin", 33),
        (17, "Yuki Tsunoda", "TSU", "JPN", "Red Bull Racing", 33),
        (18, "Pierre Gasly", "GAS", "FRA", "Alpine", 22),
        (19, "Gabriel Bortoleto", "BOR", "BRA", "Kick Sauber", 19),
        (20, "Franco Colapinto", "COL", "ARG", "Alpine", 0),
        (21, "Jack Doohan", "DOO", "AUS", "Alpine", 0),
    ]
    df = pd.DataFrame(data, columns=["Position", "Driver", "Code", "Nationality", "Team", "Driver Points"])
    df["Season"] = 2025
    df["Team Points"] = df.groupby("Team")["Driver Points"].transform("sum")
    df["Team Share"] = np.where(df["Team Points"] > 0, df["Driver Points"] / df["Team Points"], 0)
    df["Top 10"] = (df["Position"] <= 10).astype(int)
    df["profile_text"] = df["Driver"] + " " + df["Code"] + " " + df["Team"] + " " + df["Nationality"]
    return df

@st.cache_data
def load_gp_summary():
    races = [
        (1, "Australian GP", "Albert Park", "Lando Norris", "Max Verstappen", "George Russell", "McLaren", "Lando Norris", "Lando Norris"),
        (2, "Chinese GP", "Shanghai", "Oscar Piastri", "Lando Norris", "George Russell", "McLaren", "Oscar Piastri", "Lewis Hamilton"),
        (3, "Japanese GP", "Suzuka", "Max Verstappen", "Lando Norris", "Oscar Piastri", "Red Bull Racing", "Max Verstappen", "Andrea Kimi Antonelli"),
        (4, "Bahrain GP", "Sakhir", "Oscar Piastri", "George Russell", "Lando Norris", "McLaren", "Oscar Piastri", "Oscar Piastri"),
        (5, "Saudi Arabian GP", "Jeddah", "Oscar Piastri", "Max Verstappen", "Charles Leclerc", "McLaren", "Max Verstappen", "Lando Norris"),
        (6, "Miami GP", "Miami", "Oscar Piastri", "Lando Norris", "George Russell", "McLaren", "Max Verstappen", "Oscar Piastri"),
        (7, "Emilia Romagna GP", "Imola", "Max Verstappen", "Lando Norris", "Oscar Piastri", "Red Bull Racing", "Oscar Piastri", "Max Verstappen"),
        (8, "Monaco GP", "Monaco", "Lando Norris", "Charles Leclerc", "Oscar Piastri", "McLaren", "Lando Norris", "Lando Norris"),
        (9, "Spanish GP", "Barcelona", "Oscar Piastri", "Lando Norris", "Charles Leclerc", "McLaren", "Oscar Piastri", "Oscar Piastri"),
        (10, "Canadian GP", "Montreal", "George Russell", "Max Verstappen", "Kimi Antonelli", "Mercedes", "George Russell", "George Russell"),
        (11, "Austrian GP", "Red Bull Ring", "Lando Norris", "Oscar Piastri", "Charles Leclerc", "McLaren", "Lando Norris", "Lando Norris"),
        (12, "British GP", "Silverstone", "Lando Norris", "Oscar Piastri", "Nico Hulkenberg", "McLaren", "Max Verstappen", "Lando Norris"),
        (13, "Belgian GP", "Spa-Francorchamps", "Oscar Piastri", "Lando Norris", "Charles Leclerc", "McLaren", "Lando Norris", "Oscar Piastri"),
        (14, "Hungarian GP", "Hungaroring", "Lando Norris", "Oscar Piastri", "George Russell", "McLaren", "Charles Leclerc", "George Russell"),
        (15, "Dutch GP", "Zandvoort", "Oscar Piastri", "Max Verstappen", "Isack Hadjar", "McLaren", "Oscar Piastri", "Oscar Piastri"),
        (16, "Italian GP", "Monza", "Max Verstappen", "Lando Norris", "Oscar Piastri", "Red Bull Racing", "Max Verstappen", "Max Verstappen"),
        (17, "Azerbaijan GP", "Baku", "Max Verstappen", "George Russell", "Carlos Sainz", "Red Bull Racing", "Max Verstappen", "Lando Norris"),
        (18, "Singapore GP", "Marina Bay", "George Russell", "Max Verstappen", "Lando Norris", "Mercedes", "George Russell", "George Russell"),
        (19, "United States GP", "COTA", "Max Verstappen", "Lando Norris", "Charles Leclerc", "Red Bull Racing", "Max Verstappen", "Max Verstappen"),
        (20, "Mexico City GP", "Mexico City", "Lando Norris", "Charles Leclerc", "Max Verstappen", "McLaren", "Lando Norris", "Max Verstappen"),
        (21, "São Paulo GP", "Interlagos", "Lando Norris", "Kimi Antonelli", "Max Verstappen", "McLaren", "Lando Norris", "Max Verstappen"),
        (22, "Las Vegas GP", "Las Vegas", "Max Verstappen", "George Russell", "Kimi Antonelli", "Red Bull Racing", "Lando Norris", "Max Verstappen"),
        (23, "Qatar GP", "Lusail", "Max Verstappen", "Oscar Piastri", "Carlos Sainz", "Red Bull Racing", "Oscar Piastri", "Max Verstappen"),
        (24, "Abu Dhabi GP", "Yas Marina", "Lando Norris", "Max Verstappen", "Oscar Piastri", "McLaren", "Lando Norris", "Lando Norris"),
    ]
    return pd.DataFrame(races, columns=["Round", "Grand Prix", "Circuit", "Winner", "Second", "Third", "Winning Team", "Pole Sitter", "Fastest Lap"])

@st.cache_data
def build_full_classification(drivers, races):
    base_order = drivers.sort_values("Position")["Driver"].tolist()
    rows = []
    for _, race in races.iterrows():
        podium = [race["Winner"], race["Second"], race["Third"]]
        rest = [d for d in base_order if d not in podium]
        seed = int(race["Round"]) * 2025
        rng = np.random.default_rng(seed)
        top_bias = rest[:8].copy()
        lower = rest[8:].copy()
        rng.shuffle(top_bias)
        rng.shuffle(lower)
        order = podium + top_bias[:7] + lower + top_bias[7:]
        for pos, driver in enumerate(order, start=1):
            info = drivers.loc[drivers["Driver"] == driver].iloc[0]
            rows.append({
                "Round": race["Round"],
                "Grand Prix": race["Grand Prix"],
                "Circuit": race["Circuit"],
                "Race Position": pos,
                "Driver": driver,
                "Code": info["Code"],
                "Team": info["Team"],
                "Nationality": info["Nationality"],
                "Race Points": POINTS_TABLE.get(pos, 0),
                "Podium": pos <= 3,
                "Points Finish": pos <= 10,
                "Pole Sitter": race["Pole Sitter"],
                "Fastest Lap": race["Fastest Lap"],
            })
    return pd.DataFrame(rows)

@st.cache_resource
def load_model():
    df = load_driver_standings()
    model = Pipeline([
        ("vec", CountVectorizer()),
        ("clf", LogisticRegression(max_iter=300, random_state=42)),
    ])
    model.fit(df["profile_text"], df["Top 10"])
    return model

@st.cache_data
def predict_texts(texts):
    model = load_model()
    preds = model.predict(texts)
    probs = model.predict_proba(texts)[:, 1]
    return pd.DataFrame({
        "text": texts,
        "prediction": np.where(preds == 1, "Top 10 Candidate", "Outside Top 10"),
        "top10_probability": probs,
    })

def bar_chart(data, x, y, color=None, height=420, sort="-x"):
    enc = {
        "x": alt.X(f"{x}:Q", title=x.replace("_", " ")),
        "y": alt.Y(f"{y}:N", sort=sort, title=None),
        "tooltip": list(data.columns),
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N")
    return alt.Chart(data).mark_bar().encode(**enc).properties(height=height)

def make_metric_cards(drivers, races, full):
    leader = drivers.iloc[0]
    constructor = drivers.groupby("Team", as_index=False)["Driver Points"].sum().sort_values("Driver Points", ascending=False).iloc[0]
    win_counts = races["Winner"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Drivers Leader", leader["Driver"], f'{leader["Driver Points"]} pts')
    col2.metric("Constructors Leader", constructor["Team"], f'{constructor["Driver Points"]} pts')
    col3.metric("Most Wins", win_counts.index[0], f'{win_counts.iloc[0]} wins')
    col4.metric("Total Races", len(races), "2025 season")

# data
standings_df = load_driver_standings()
gp_df = load_gp_summary()
classification_df = build_full_classification(standings_df, gp_df)

race_stats = classification_df.groupby("Driver", as_index=False).agg(
    Wins=("Race Position", lambda s: int((s == 1).sum())),
    Podiums=("Race Position", lambda s: int((s <= 3).sum())),
    Points_Finishes=("Race Position", lambda s: int((s <= 10).sum())),
    Avg_Finish=("Race Position", "mean"),
    Best_Finish=("Race Position", "min"),
    Race_Points=("Race Points", "sum"),
)
standings_plus = standings_df.merge(race_stats, on="Driver", how="left")
constructor_df = standings_df.groupby("Team", as_index=False).agg(
    Drivers=("Driver", "count"),
    Constructor_Points=("Driver Points", "sum"),
    Best_Driver_Rank=("Position", "min"),
).sort_values("Constructor_Points", ascending=False)

st.title("🏎️ F1 2025 Analytics Dashboard")
st.caption("드라이버 순위, 컨스트럭터 순위, 그랑프리별 전체 결과, 드라이버 상세 분석, 간단한 ML 예측을 한 화면에서 확인합니다.")
st.info("주의: 드라이버 최종 순위는 공식 순위 기준이며, 서킷별 전체 1~21위 표는 Streamlit 실습용으로 생성한 샘플 classification 데이터입니다.")
make_metric_cards(standings_df, gp_df, classification_df)
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Standings", "Grand Prix", "Drivers", "Teams", "Statistics", "Prediction", "Data"
])

with tab1:
    st.header("Championship Standings")
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.subheader("Drivers Championship")
        chart_df = standings_df.sort_values("Driver Points", ascending=True)
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Driver Points:Q", title="Points"),
            y=alt.Y("Driver:N", sort="x", title=None),
            color=alt.Color("Team:N"),
            tooltip=["Position", "Driver", "Team", "Driver Points"],
        ).properties(height=560)
        st.altair_chart(chart, use_container_width=True)
    with c2:
        st.subheader("Constructors Championship")
        st.altair_chart(
            bar_chart(constructor_df, "Constructor_Points", "Team", height=360),
            use_container_width=True,
        )
        st.dataframe(constructor_df, use_container_width=True, hide_index=True)
    st.dataframe(
        standings_plus.drop(columns=["profile_text"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Driver Points": st.column_config.ProgressColumn("Driver Points", min_value=0, max_value=int(standings_df["Driver Points"].max()), format="%d"),
            "Team Share": st.column_config.ProgressColumn("Team Share", min_value=0, max_value=1, format="%.1%"),
        },
    )

with tab2:
    st.header("Grand Prix Results")
    gp_name = st.selectbox("Grand Prix", gp_df["Grand Prix"].tolist(), index=4)
    gp = gp_df.loc[gp_df["Grand Prix"] == gp_name].iloc[0]
    gp_class = classification_df[classification_df["Grand Prix"] == gp_name].sort_values("Race Position")

    st.subheader(f'{gp["Round"]}. {gp["Grand Prix"]} · {gp["Circuit"]}')
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🥇 Winner", gp["Winner"])
    p2.metric("🥈 P2", gp["Second"])
    p3.metric("🥉 P3", gp["Third"])
    p4.metric("Pole", gp["Pole Sitter"])

    left, right = st.columns([0.52, 0.48])
    with left:
        st.subheader("Full classification")
        st.dataframe(
            gp_class[["Race Position", "Driver", "Team", "Race Points", "Podium", "Points Finish", "Pole Sitter", "Fastest Lap"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Race Points": st.column_config.ProgressColumn("Race Points", min_value=0, max_value=25, format="%d"),
                "Podium": st.column_config.CheckboxColumn("Podium"),
                "Points Finish": st.column_config.CheckboxColumn("Points Finish"),
            },
        )
    with right:
        st.subheader("Race points by driver")
        pts = gp_class.sort_values("Race Points", ascending=True)
        chart = alt.Chart(pts).mark_bar().encode(
            x=alt.X("Race Points:Q", title="Race Points"),
            y=alt.Y("Driver:N", sort="x", title=None),
            color=alt.Color("Team:N"),
            tooltip=["Race Position", "Driver", "Team", "Race Points"],
        ).properties(height=560)
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Race calendar summary")
    st.dataframe(gp_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Driver Detail")
    driver = st.selectbox("Driver", standings_df["Driver"].tolist())
    driver_row = standings_plus[standings_plus["Driver"] == driver].iloc[0]
    driver_results = classification_df[classification_df["Driver"] == driver].sort_values("Round")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rank", int(driver_row["Position"]))
    c2.metric("Points", int(driver_row["Driver Points"]))
    c3.metric("Wins", int(driver_row["Wins"]))
    c4.metric("Podiums", int(driver_row["Podiums"]))
    c5.metric("Avg Finish", f'{driver_row["Avg_Finish"]:.1f}')

    st.subheader("Race-by-race finish")
    line = alt.Chart(driver_results).mark_line(point=True).encode(
        x=alt.X("Round:O", title="Round"),
        y=alt.Y("Race Position:Q", sort="descending", title="Finish Position"),
        tooltip=["Round", "Grand Prix", "Circuit", "Race Position", "Race Points"],
    ).properties(height=360)
    st.altair_chart(line, use_container_width=True)
    st.dataframe(
        driver_results[["Round", "Grand Prix", "Circuit", "Race Position", "Race Points", "Podium", "Points Finish"]],
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.header("Team Analysis")
    team = st.selectbox("Team", constructor_df["Team"].tolist())
    team_drivers = standings_plus[standings_plus["Team"] == team].sort_values("Position")
    team_results = classification_df[classification_df["Team"] == team]
    team_gp = team_results.groupby("Grand Prix", as_index=False).agg(
        Round=("Round", "first"),
        Race_Points=("Race Points", "sum"),
        Best_Finish=("Race Position", "min"),
    ).sort_values("Round")

    a, b, c = st.columns(3)
    a.metric("Constructor Points", int(team_drivers["Driver Points"].sum()))
    b.metric("Drivers", len(team_drivers))
    c.metric("Best Finish", int(team_results["Race Position"].min()))

    st.subheader("Team drivers")
    st.dataframe(team_drivers.drop(columns=["profile_text"]), use_container_width=True, hide_index=True)
    st.subheader("Race points trend")
    chart = alt.Chart(team_gp).mark_line(point=True).encode(
        x=alt.X("Round:O"),
        y=alt.Y("Race_Points:Q", title="Race Points"),
        tooltip=["Round", "Grand Prix", "Race_Points", "Best_Finish"],
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)

with tab5:
    st.header("Season Statistics")
    wins = gp_df["Winner"].value_counts().rename_axis("Driver").reset_index(name="Wins")
    podium_rows = pd.concat([
        gp_df[["Winner"]].rename(columns={"Winner": "Driver"}),
        gp_df[["Second"]].rename(columns={"Second": "Driver"}),
        gp_df[["Third"]].rename(columns={"Third": "Driver"}),
    ])
    podiums = podium_rows["Driver"].value_counts().rename_axis("Driver").reset_index(name="Podiums")
    poles = gp_df["Pole Sitter"].value_counts().rename_axis("Driver").reset_index(name="Poles")
    fastest = gp_df["Fastest Lap"].value_counts().rename_axis("Driver").reset_index(name="Fastest Laps")

    stat_choice = st.radio("Statistic", ["Wins", "Podiums", "Poles", "Fastest Laps"], horizontal=True)
    stat_map = {"Wins": wins, "Podiums": podiums, "Poles": poles, "Fastest Laps": fastest}
    stat_df = stat_map[stat_choice].sort_values(stat_choice, ascending=True)
    chart = alt.Chart(stat_df).mark_bar().encode(
        x=alt.X(f"{stat_choice}:Q", title=stat_choice),
        y=alt.Y("Driver:N", sort="x", title=None),
        tooltip=["Driver", stat_choice],
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

    combined = standings_df[["Driver", "Team", "Driver Points"]].merge(wins, on="Driver", how="left").merge(podiums, on="Driver", how="left").merge(poles, on="Driver", how="left").merge(fastest, on="Driver", how="left").fillna(0)
    st.dataframe(combined, use_container_width=True, hide_index=True)

with tab6:
    st.header("Cached Model Prediction")
    st.write("`@st.cache_resource`로 모델을 한 번만 학습하고, `@st.cache_data`로 같은 입력의 예측 결과를 재사용합니다.")
    sample = "Lando Norris McLaren GBR\nMax Verstappen Red Bull Racing NED\nFranco Colapinto Alpine ARG"
    text = st.text_area("Prediction input", value=sample, height=140)
    texts = [line.strip() for line in text.splitlines() if line.strip()]
    if texts:
        pred = predict_texts(texts)
        st.dataframe(
            pred,
            use_container_width=True,
            hide_index=True,
            column_config={
                "top10_probability": st.column_config.ProgressColumn("Top 10 Probability", min_value=0, max_value=1, format="%.1%")
            },
        )

with tab7:
    st.header("Data Browser")
    dataset = st.radio("Dataset", ["Driver standings", "Grand Prix summary", "Full race classification"], horizontal=True)
    if dataset == "Driver standings":
        show_df = standings_df.drop(columns=["profile_text"])
    elif dataset == "Grand Prix summary":
        show_df = gp_df
    else:
        gp_filter = st.multiselect("Grand Prix filter", gp_df["Grand Prix"].tolist(), default=gp_df["Grand Prix"].tolist()[:3])
        show_df = classification_df[classification_df["Grand Prix"].isin(gp_filter)]
    search = st.text_input("Search")
    if search:
        mask = show_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        show_df = show_df[mask]
    st.dataframe(show_df, use_container_width=True, hide_index=True, height=560)
