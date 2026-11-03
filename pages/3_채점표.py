
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/hhj-09/data/refs/heads/main/stroke.csv"
RANDOM_STATE = 42

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(KOREAN_NAMES.keys())


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "id", "age", "avg_glucose_level", "bmi",
        "hypertension", "heart_disease", "stroke"
    ]

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"필요한 열이 없습니다: {missing_columns}")

    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["id", "stroke"]).copy()
    data = data.sort_values("id").reset_index(drop=True)
    data["stroke"] = data["stroke"].astype(int)

    return data


try:
    data = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 제목 및 입력 속성 선택
# --------------------------------------------------

st.title("📋 채점표")
st.write(
    "테스트 데이터로 분류 모델의 예측 결과를 채점하고, "
    "실제로 놓친 환자와 잘못 예측한 사람을 확인합니다."
)

st.info(
    "stroke가 1이면 양성(뇌졸중), 0이면 음성(뇌졸중 아님)입니다."
)

st.subheader("1. 입력 속성 선택")

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: KOREAN_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 학습용 / 테스트용 데이터 분리
# --------------------------------------------------

data = data.sort_values("id").reset_index(drop=True)

positions = np.arange(len(data))
test_mask = positions % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].copy()
y_test = test_data["stroke"].copy()


# --------------------------------------------------
# BMI 결측값 처리
# --------------------------------------------------

bmi_median = None

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

if X_train.isnull().any().any() or X_test.isnull().any().any():
    st.error(
        "선택한 입력 속성에 결측값이 남아 있습니다. "
        "BMI 이외의 결측값도 확인해 주세요."
    )
    st.stop()


# --------------------------------------------------
# 데이터 분리 정보
# --------------------------------------------------

positive_test_count = int((y_test == 1).sum())

st.subheader("2. 데이터 분리")

st.write(
    f"테스트용 사람 수: **{len(test_data):,}명** "
    f"/ 그중 실제 뇌졸중인 사람: **{positive_test_count:,}명**"
)

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(data):,}명")
col2.metric("훈련 데이터", f"{len(train_data):,}명")
col3.metric("테스트 데이터", f"{len(test_data):,}명")

st.caption(
    "번호 순으로 정렬한 뒤 10명씩 묶어 각 묶음의 앞 3명을 "
    "테스트용으로 고정했습니다. 나머지 7명은 훈련용입니다."
)

if bmi_median is not None:
    st.caption(
        f"BMI 결측값은 훈련 데이터 중앙값({bmi_median:.2f})으로 채웠습니다."
    )


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

logistic_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(X_train, y_train)


# --------------------------------------------------
# 예측
# --------------------------------------------------

logistic_pred = logistic_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# 훈련용에서 더 많은 쪽으로만 답하는 기준 모델
majority_class = int(y_train.mode().iloc[0])

baseline_pred = np.full(
    len(y_test),
    majority_class,
    dtype=int
)


# --------------------------------------------------
# 혼동행렬과 평가 지표 계산
# --------------------------------------------------

def calculate_scores(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0

    actual_positive = tp + fn
    predicted_positive = tp + fp

    recall = tp / actual_positive if actual_positive > 0 else None
    precision = tp / predicted_positive if predicted_positive > 0 else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


logistic_scores = calculate_scores(y_test, logistic_pred)
tree_scores = calculate_scores(y_test, tree_pred)
baseline_scores = calculate_scores(y_test, baseline_pred)


# --------------------------------------------------
# 혼동행렬 표시
# --------------------------------------------------

def show_confusion_matrix(scores, model_title):
    st.markdown(f"### {model_title}")

    tn = scores["TN"]
    fp = scores["FP"]
    fn = scores["FN"]
    tp = scores["TP"]

    matrix = pd.DataFrame(
        [
            [
                f"TN\n{tn:,}명",
                f"FP\n{fp:,}명",
                f"{tn + fp:,}명"
            ],
            [
                f"FN\n{fn:,}명",
                f"TP\n{tp:,}명",
                f"{fn + tp:,}명"
            ],
            [
                f"{tn + fn:,}명",
                f"{fp + tp:,}명",
                f"{tn + fp + fn + tp:,}명"
            ]
        ],
        index=[
            "실제 뇌졸중 아님",
            "실제 뇌졸중",
            "열 합계"
        ],
        columns=[
            "예측: 뇌졸중 아님",
            "예측: 뇌졸중",
            "행 합계"
        ]
    )

    st.dataframe(
        matrix,
        use_container_width=True
    )

    st.caption(
        "TP: 실제 뇌졸중을 뇌졸중으로 예측 / "
        "TN: 실제 아님을 아님으로 예측 / "
        "FP: 실제 아님을 뇌졸중으로 예측 / "
        "FN: 실제 뇌졸중을 아님으로 예측"
    )


st.subheader("3. 테스트 데이터 혼동행렬")

matrix_col1, matrix_col2 = st.columns(2)

with matrix_col1:
    show_confusion_matrix(
        logistic_scores,
        "로지스틱 회귀(확률로 답하는 모델)"
    )

with matrix_col2:
    show_confusion_matrix(
        tree_scores,
        "의사결정트리(질문으로 답하는 모델)"
    )


# --------------------------------------------------
# 평가 지표 비교표
# --------------------------------------------------

st.subheader("4. 네 가지 평가 지표 비교")

def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value * 100:.2f}%"


metric_rows = []

for name, scores in [
    ("로지스틱 회귀(확률로 답하는 모델)", logistic_scores),
    ("의사결정트리(질문으로 답하는 모델)", tree_scores),
    ("훈련용에서 많은 쪽으로만 답하는 모델", baseline_scores)
]:
    metric_rows.append({
        "모델": name,
        "정확도": format_metric(scores["정확도"]),
        "재현율": format_metric(scores["재현율"]),
        "정밀도": format_metric(scores["정밀도"]),
        "F1": format_metric(scores["F1"])
    })

metric_table = pd.DataFrame(metric_rows)

st.dataframe(
    metric_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정확도 = (TP + TN) / 전체 인원, "
    "재현율 = TP / (TP + FN), "
    "정밀도 = TP / (TP + FP), "
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)


# --------------------------------------------------
# 오류 예측 목록을 위한 모델 선택
# --------------------------------------------------

st.subheader("5. 잘못 예측한 사람 확인")

model_choice = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)"
    ]
)

if model_choice.startswith("로지스틱"):
    selected_pred = logistic_pred
    selected_scores = logistic_scores
else:
    selected_pred = tree_pred
    selected_scores = tree_scores


# --------------------------------------------------
# 오류 목록 만들기
# --------------------------------------------------

error_data = test_data.copy()
error_data["실제"] = y_test.to_numpy()
error_data["예측"] = selected_pred

# 번호와 필요한 속성만 표시
display_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

missed = error_data[
    (error_data["실제"] == 1) &
    (error_data["예측"] == 0)
].copy()

false_alarm = error_data[
    (error_data["실제"] == 0) &
    (error_data["예측"] == 1)
].copy()

# 나이가 많은 순서
missed = missed.sort_values(
    "age",
    ascending=False
)

false_alarm = false_alarm.sort_values(
    "age",
    ascending=False
)


def prepare_error_table(frame):
    result = frame[display_columns].copy()

    result = result.rename(columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부"
    })

    result["고혈압 여부"] = result["고혈압 여부"].map({
        0: "없음",
        1: "있음"
    })

    result["심장병 여부"] = result["심장병 여부"].map({
        0: "없음",
        1: "있음"
    })

    result["나이"] = result["나이"].round(1)
    result["평균 혈당"] = result["평균 혈당"].round(2)

    return result


missed_table = prepare_error_table(missed)
false_alarm_table = prepare_error_table(false_alarm)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown(
        f"### 놓친 환자 ({len(missed):,}명)"
    )
    st.caption(
        "실제 뇌졸중이지만 모델이 뇌졸중이 아니라고 예측한 사람"
    )

    st.dataframe(
        missed_table.head(20),
        use_container_width=True,
        hide_index=True
    )

    if len(missed_table) > 20:
        st.caption(
            f"전체 {len(missed_table):,}명 중 나이가 많은 "
            "앞 20명만 표시했습니다."
        )

with right_col:
    st.markdown(
        f"### 헛짚은 사람 ({len(false_alarm):,}명)"
    )
    st.caption(
        "실제 뇌졸중이 아니지만 모델이 뇌졸중이라고 예측한 사람"
    )

    st.dataframe(
        false_alarm_table.head(20),
        use_container_width=True,
        hide_index=True
    )

    if len(false_alarm_table) > 20:
        st.caption(
            f"전체 {len(false_alarm_table):,}명 중 나이가 많은 "
            "앞 20명만 표시했습니다."
        )


# --------------------------------------------------
# 혼동행렬과 목록 인원 비교
# --------------------------------------------------

st.info(
    f"목록 인원 확인: 놓친 환자 {len(missed):,}명 = "
    f"혼동행렬 FN {selected_scores['FN']:,}명 / "
    f"헛짚은 사람 {len(false_alarm):,}명 = "
    f"혼동행렬 FP {selected_scores['FP']:,}명"
)


# --------------------------------------------------
# 마무리
# --------------------------------------------------

st.divider()

st.warning(
    "이 페이지는 교육용 데이터로 분류 모델을 평가하는 실습입니다. "
    "실제 개인의 뇌졸중 진단이나 의료적 판단에 사용할 수 없습니다."
)
