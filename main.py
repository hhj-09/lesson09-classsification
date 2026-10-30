
import streamlit as st
import pandas as pd
import json
import os

# =====================================
# 1. 기본 설정
# =====================================
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "뇌졸중 관련 데이터를 살펴보고 데이터의 구성과 특징을 "
    "직접 정리하는 실습 공간입니다."
)

# =====================================
# 2. 데이터 주소 및 기록 파일 경로
# =====================================
DATA_URL = (
    "https://raw.githubusercontent.com/hhj-09/data/"
    "refs/heads/main/stroke.csv"
)

RECORD_FOLDER = "내_기록"
RECORD_FILE = os.path.join(
    RECORD_FOLDER,
    "stroke_notes.json"
)

EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]


# =====================================
# 3. 기존 기록 불러오기
# =====================================
def load_records():
    """저장된 우리말 뜻과 데이터 출처를 불러옵니다."""

    if not os.path.exists(RECORD_FILE):
        return {
            "meanings": {},
            "source": ""
        }

    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)

        return {
            "meanings": records.get("meanings", {}),
            "source": records.get("source", "")
        }

    except (json.JSONDecodeError, OSError):
        return {
            "meanings": {},
            "source": ""
        }


# =====================================
# 4. 기록 저장하기
# =====================================
def save_records(meanings, source):
    """입력한 내용을 JSON 파일로 저장합니다."""

    os.makedirs(RECORD_FOLDER, exist_ok=True)

    records = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )


# =====================================
# 5. 데이터 불러오기
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    # 요청한 열 순서대로 정리
    df = df[EXPECTED_COLUMNS]

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 문제가 발생했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# =====================================
# 6. 데이터 기본 정보 계산
# =====================================
total_people = len(df)
total_columns = len(df.columns)

# stroke가 1인 사람 수
stroke_count = (
    pd.to_numeric(df["stroke"], errors="coerce") == 1
).sum()

# 뇌졸중 발생 비율
stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)


# =====================================
# 7. 숫자 카드 네 개
# =====================================
st.subheader("📊 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="전체 사람 수",
    value=f"{total_people:,}명"
)

col2.metric(
    label="전체 열 개수",
    value=f"{total_columns}개"
)

col3.metric(
    label="stroke가 1인 사람 수",
    value=f"{stroke_count:,}명"
)

col4.metric(
    label="stroke가 1인 사람 비율",
    value=f"{stroke_ratio:.2f}%"
)

st.divider()


# =====================================
# 8. 열의 값 종류 확인
# =====================================
def get_value_type(column):
    """각 열의 값 종류를 구분합니다."""

    if column == "id":
        return "식별 번호"

    # 0과 1로 이루어진 열은 범주형으로 표시
    if column in [
        "hypertension",
        "heart_disease",
        "stroke"
    ]:
        return "범주형 (0, 1)"

    if pd.api.types.is_numeric_dtype(df[column]):
        return "수치형"

    unique_count = df[column].nunique(dropna=True)

    return f"범주형 ({unique_count}종류)"


# =====================================
# 9. 기존 기록 불러오기
# =====================================
saved_records = load_records()
saved_meanings = saved_records["meanings"]


# =====================================
# 10. 열 이름과 데이터 특성 표
# =====================================
st.subheader("📋 데이터 열의 의미와 종류")

st.write(
    "교재를 참고하여 각 열의 우리말 뜻을 직접 입력하세요. "
    "입력한 내용은 자동으로 저장됩니다."
)

column_info = []

for column in EXPECTED_COLUMNS:
    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": get_value_type(column),
        "빈 값 개수": int(df[column].isna().sum())
    })

info_df = pd.DataFrame(column_info)

edited_info = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            width="medium"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 직접 입력하세요.",
            width="large"
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            width="medium"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d개",
            width="small"
        )
    },
    key="meaning_editor"
)


# =====================================
# 11. 데이터 처음 다섯 행
# =====================================
st.divider()

st.subheader("🔍 데이터 미리 보기")

st.write(
    "실제 데이터의 처음 다섯 행입니다. "
    "빈 값은 원본 그대로 표시합니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# =====================================
# 12. 데이터 출처 입력
# =====================================
st.divider()

st.subheader("📚 데이터 출처")

st.write(
    "교재에 제시된 데이터 출처를 확인하여 입력하세요."
)

source_text = st.text_area(
    "데이터 출처",
    value=saved_records["source"],
    placeholder="교재에 나온 데이터 출처를 입력하세요.",
    height=100,
    key="source_editor"
)


# =====================================
# 13. 우리말 뜻과 출처 저장
# =====================================
current_meanings = {}

for _, row in edited_info.iterrows():
    current_meanings[row["열 이름"]] = row["우리말 뜻"]

try:
    save_records(
        current_meanings,
        source_text
    )

    st.success(
        "입력한 우리말 뜻과 데이터 출처가 자동 저장되었습니다."
    )

except OSError as e:
    st.error(f"기록을 저장하지 못했습니다: {e}")


# =====================================
# 14. 데이터 출처 안내
# =====================================
st.divider()

st.caption(
    "데이터 파일: "
    "https://github.com/hhj-09/data/blob/main/stroke.csv"
)

st.caption(
    "입력한 기록은 내_기록/stroke_notes.json 파일에 저장됩니다."
)
