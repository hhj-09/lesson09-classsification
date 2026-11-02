
import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/hhj-09/data/refs/heads/main/stroke.csv"


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_columns = [
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    return data


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# 필요한 열 확인
required_columns = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke",
    "smoking_status"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()


# ==========================================
# 3. 페이지 제목
# ==========================================

st.title("🔎 데이터 탐색")

st.write(
    "뇌졸중 데이터의 분포와 여러 특성에 따른 "
    "뇌졸중 발생 비율을 살펴봅니다."
)

st.divider()


# ==========================================
# 4. 나이와 평균 혈당의 분포
# ==========================================

st.header("1. 나이와 평균 혈당의 분포")

st.write(
    "전체 데이터에서 나이와 평균 혈당이 "
    "어떤 분포를 나타내는지 확인합니다."
)

col1, col2 = st.columns(2)

with col1:
    age_data = df.dropna(subset=["age"])

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "인원수"
        }
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="인원수",
        showlegend=False
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:
    glucose_data = df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "인원수"
        }
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="인원수",
        showlegend=False
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# ==========================================
# 5. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# ==========================================

st.header("2. 뇌졸중 발생 여부에 따른 특성 비교")

st.write(
    "뇌졸중을 겪은 사람과 겪지 않은 사람의 "
    "나이 및 평균 혈당 분포를 비교합니다."
)

df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 미발생",
    1: "뇌졸중 발생"
})

col1, col2 = st.columns(2)

with col1:
    age_box_data = df.dropna(
        subset=["stroke_group", "age"]
    )

    fig_box_age = px.box(
        age_box_data,
        x="stroke_group",
        y="age",
        color="stroke_group",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이"
        },
        category_orders={
            "stroke_group": [
                "뇌졸중 미발생",
                "뇌졸중 발생"
            ]
        }
    )

    fig_box_age.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이"
    )

    st.plotly_chart(
        fig_box_age,
        use_container_width=True
    )


with col2:
    glucose_box_data = df.dropna(
        subset=["stroke_group", "avg_glucose_level"]
    )

    fig_box_glucose = px.box(
        glucose_box_data,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        category_orders={
            "stroke_group": [
                "뇌졸중 미발생",
                "뇌졸중 발생"
            ]
        }
    )

    fig_box_glucose.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당"
    )

    st.plotly_chart(
        fig_box_glucose,
        use_container_width=True
    )


# 그룹별 평균값 계산
mean_data = df.dropna(
    subset=["stroke_group"]
)

mean_table = (
    mean_data
    .groupby("stroke_group")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex([
        "뇌졸중 미발생",
        "뇌졸중 발생"
    ])
    .reset_index()
)

mean_table.columns = [
    "뇌졸중 여부",
    "평균 나이",
    "평균 혈당"
]

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.subheader("뇌졸중 여부에 따른 평균값")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 6. 고혈압과 심장병에 따른 뇌졸중 발생 비율
# ==========================================

st.header("3. 질환 여부에 따른 뇌졸중 발생 비율")

st.write(
    "고혈압과 심장병이 있는 집단과 없는 집단의 "
    "뇌졸중 발생 비율을 비교합니다."
)


def make_rate_table(data, column, labels):
    valid_data = data.dropna(
        subset=[column, "stroke"]
    ).copy()

    valid_data["구분"] = valid_data[column].map(labels)

    valid_data = valid_data.dropna(
        subset=["구분"]
    )

    result = (
        valid_data
        .groupby("구분", sort=False)["stroke"]
        .agg(["count", "sum"])
        .reset_index()
    )

    result.columns = [
        "구분",
        "전체 인원",
        "뇌졸중 발생 인원"
    ]

    result["뇌졸중 발생 비율"] = (
        result["뇌졸중 발생 인원"]
        / result["전체 인원"]
        * 100
    )

    return result


# 고혈압 데이터
hypertension_labels = {
    0: "고혈압 없음",
    1: "고혈압 있음"
}

hypertension_table = make_rate_table(
    df,
    "hypertension",
    hypertension_labels
)


# 심장병 데이터
heart_labels = {
    0: "심장병 없음",
    1: "심장병 있음"
}

heart_table = make_rate_table(
    df,
    "heart_disease",
    heart_labels
)


col1, col2 = st.columns(2)

with col1:
    st.subheader("고혈압 여부에 따른 뇌졸중 비율")

    fig_hypertension = px.bar(
        hypertension_table,
        x="구분",
        y="뇌졸중 발생 비율",
        text="뇌졸중 발생 비율",
        title="고혈압과 뇌졸중 발생 비율",
        labels={
            "구분": "고혈압 여부",
            "뇌졸중 발생 비율": "뇌졸중 발생 비율(%)"
        }
    )

    fig_hypertension.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    max_value = hypertension_table[
        "뇌졸중 발생 비율"
    ].max()

    fig_hypertension.update_layout(
        xaxis_title="고혈압 여부",
        yaxis_title="뇌졸중 발생 비율(%)",
        yaxis_range=[0, max(1, max_value * 1.2)]
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True
    )

    st.dataframe(
        hypertension_table.round(2),
        use_container_width=True,
        hide_index=True
    )


with col2:
    st.subheader("심장병 여부에 따른 뇌졸중 비율")

    fig_heart = px.bar(
        heart_table,
        x="구분",
        y="뇌졸중 발생 비율",
        text="뇌졸중 발생 비율",
        title="심장병과 뇌졸중 발생 비율",
        labels={
            "구분": "심장병 여부",
            "뇌졸중 발생 비율": "뇌졸중 발생 비율(%)"
        }
    )

    fig_heart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    max_value = heart_table[
        "뇌졸중 발생 비율"
    ].max()

    fig_heart.update_layout(
        xaxis_title="심장병 여부",
        yaxis_title="뇌졸중 발생 비율(%)",
        yaxis_range=[0, max(1, max_value * 1.2)]
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True
    )

    st.dataframe(
        heart_table.round(2),
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ==========================================
# 7. BMI 결측자와 전체 데이터의 뇌졸중 비율
# ==========================================

st.header("4. BMI 결측자와 전체 데이터 비교")

st.write(
    "BMI가 비어 있는 사람들만 따로 구분하여 "
    "전체 데이터와 뇌졸중 발생 비율을 비교합니다."
)

# BMI가 비어 있는 사람
missing_bmi = df[df["bmi"].isna()].copy()

# 뇌졸중 여부가 확인되는 전체 사람
valid_stroke = df.dropna(
    subset=["stroke"]
)

# BMI 결측자 중 뇌졸중 여부가 확인되는 사람
missing_bmi_valid = missing_bmi.dropna(
    subset=["stroke"]
)

# BMI 결측자의 뇌졸중 발생 인원
missing_bmi_stroke_count = int(
    (missing_bmi_valid["stroke"] == 1).sum()
)

# 전체 데이터의 뇌졸중 발생 인원
total_stroke_count = int(
    (valid_stroke["stroke"] == 1).sum()
)

# BMI 결측자의 발생 비율
if len(missing_bmi_valid) > 0:
    missing_bmi_rate = (
        missing_bmi_stroke_count
        / len(missing_bmi_valid)
        * 100
    )
else:
    missing_bmi_rate = 0


# 전체 데이터의 발생 비율
if len(valid_stroke) > 0:
    total_stroke_rate = (
        total_stroke_count
        / len(valid_stroke)
        * 100
    )
else:
    total_stroke_rate = 0


bmi_table = pd.DataFrame({
    "구분": [
        "BMI 결측자",
        "전체 데이터"
    ],
    "전체 인원": [
        len(missing_bmi_valid),
        len(valid_stroke)
    ],
    "뇌졸중 발생 인원": [
        missing_bmi_stroke_count,
        total_stroke_count
    ],
    "뇌졸중 발생 비율(%)": [
        round(missing_bmi_rate, 2),
        round(total_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "BMI 결측 인원: "
    + str(len(missing_bmi))
    + "명"
)


st.divider()


# ==========================================
# 8. 흡연 상태별 인원수
# ==========================================

st.header("5. 흡연 상태별 인원수")

st.write(
    "흡연 상태의 각 값에 해당하는 사람 수를 확인합니다. "
    "결측값도 별도로 표시합니다."
)

smoking_counts = (
    df["smoking_status"]
    .fillna("결측값")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="인원수")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 9. 하단 안내
# ==========================================

st.divider()

st.caption(
    "※ 이 페이지는 데이터의 분포와 집단별 차이를 "
    "탐색하기 위한 것입니다. "
    "집단 간 비율 차이만으로 특정 요인이 "
    "뇌졸중의 원인이라고 판단할 수는 없습니다."
)
