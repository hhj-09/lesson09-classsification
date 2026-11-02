
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/hhj-09/data/refs/heads/main/stroke.csv"

RANDOM_STATE = 42

# 열 이름과 화면에 표시할 우리말 이름
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
        "id",
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"데이터에 필요한 열이 없습니다: {missing_columns}"
        )

    # 숫자형으로 변환
    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # 번호 순으로 정렬
    data = data.sort_values(
        "id",
        ascending=True
    ).reset_index(drop=True)

    # 정답이 없는 행은 제외
    data = data.dropna(subset=["stroke"]).copy()

    # 뇌졸중 여부를 정수형으로 변환
    data["stroke"] = data["stroke"].astype(int)

    return data


try:
    data = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 제목
# --------------------------------------------------

st.title("🤖 분류 모델")
st.write(
    "선택한 건강 정보를 바탕으로 뇌졸중 발생 여부를 분류하는 "
    "두 가지 머신러닝 모델을 비교합니다."
)

st.info(
    "뇌졸중(stroke)이 1이면 양성(뇌졸중), "
    "0이면 음성(뇌졸중 아님)으로 설정했습니다."
)


# --------------------------------------------------
# 입력 속성 선택
# --------------------------------------------------

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
    st.warning("입력 속성을 두 개 이상 선택해야 모델을 만들 수 있습니다.")
    st.stop()

selected_names = [
    KOREAN_NAMES[column]
    for column in selected_features
]

st.write(
    "선택한 속성: " + ", ".join(selected_names)
)


# --------------------------------------------------
# 학습용 / 테스트용 데이터 분리
# --------------------------------------------------

# 번호 순으로 정렬된 데이터에서
# 10명씩 묶었을 때 앞의 3명을 테스트용으로 지정한다.
data = data.sort_values(
    "id",
    ascending=True
).reset_index(drop=True)

position = np.arange(len(data))

test_mask = position % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

# 입력 데이터와 정답 데이터 분리
X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].copy()
y_test = test_data["stroke"].copy()


# --------------------------------------------------
# BMI 결측값 처리
# --------------------------------------------------

bmi_median = None

if "bmi" in selected_features:

    # 학습용 데이터의 BMI 중앙값만 사용
    bmi_median = X_train["bmi"].median()

    # 학습용과 테스트용 모두 학습용 중앙값으로 채움
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 선택한 입력 속성에 결측값이 남아 있는지 확인
if X_train.isnull().any().any() or X_test.isnull().any().any():
    st.error(
        "선택한 입력 속성에 BMI 이외의 결측값이 있습니다. "
        "해당 데이터의 결측값을 확인해야 합니다."
    )
    st.stop()


# --------------------------------------------------
# 데이터 분리 정보
# --------------------------------------------------

st.subheader("2. 데이터 분리")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터",
        f"{len(data):,}명"
    )

with col2:
    st.metric(
        "학습 데이터",
        f"{len(train_data):,}명"
    )

with col3:
    st.metric(
        "테스트 데이터",
        f"{len(test_data):,}명"
    )

st.caption(
    "번호 순으로 정렬한 뒤 10명씩 묶고, 각 묶음의 앞 3명을 "
    "테스트용으로 고정했습니다. 나머지 7명만 학습에 사용합니다."
)

if len(test_data) == 1533:
    st.success("테스트 데이터가 1,533명으로 분리되었습니다.")
else:
    st.warning(
        f"현재 데이터의 테스트 인원은 {len(test_data):,}명입니다. "
        "데이터의 행 수에 따라 테스트 인원은 달라질 수 있습니다."
    )

if bmi_median is not None:
    st.write(
        f"체질량지수 결측값 대체 기준: "
        f"학습 데이터의 중앙값 **{bmi_median:.2f}**"
    )
else:
    st.write("체질량지수는 선택하지 않았으므로 결측값 대체를 하지 않았습니다.")


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------

st.subheader("3. 분류 모델 학습")

# 로지스틱 회귀
# 표준화는 학습 데이터에 대해서만 기준을 계산한다.
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

logistic_model.fit(X_train, y_train)


# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(X_train, y_train)


# --------------------------------------------------
# 기준 모델
# --------------------------------------------------

# 입력 속성을 하나도 보지 않고
# 학습 데이터에서 더 많은 쪽으로만 예측한다.
majority_class = int(y_train.mode().iloc[0])

baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)


# --------------------------------------------------
# 정확도 계산
# --------------------------------------------------

logistic_train_accuracy = logistic_model.score(
    X_train,
    y_train
)

logistic_test_accuracy = logistic_model.score(
    X_test,
    y_test
)

tree_train_accuracy = tree_model.score(
    X_train,
    y_train
)

tree_test_accuracy = tree_model.score(
    X_test,
    y_test
)

baseline_train_accuracy = np.mean(
    baseline_train_pred == y_train.to_numpy()
)

baseline_test_accuracy = np.mean(
    baseline_test_pred == y_test.to_numpy()
)


# --------------------------------------------------
# 정확도 카드
# --------------------------------------------------

st.subheader("4. 모델 정확도 비교")

def accuracy_card(title, train_accuracy, test_accuracy):
    st.markdown(f"### {title}")

    st.metric(
        label="테스트 데이터 정확도",
        value=f"{test_accuracy * 100:.2f}%"
    )

    small_col1, small_col2 = st.columns(2)

    with small_col1:
        st.caption("훈련 데이터 정확도")
        st.markdown(
            f"**{train_accuracy * 100:.2f}%**"
        )

    with small_col2:
        st.caption("테스트 데이터 정확도")
        st.markdown(
            f"**{test_accuracy * 100:.2f}%**"
        )


card1, card2, card3 = st.columns(3)

with card1:
    accuracy_card(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_train_accuracy,
        logistic_test_accuracy
    )

with card2:
    accuracy_card(
        "의사결정트리(질문으로 답하는 모델)",
        tree_train_accuracy,
        tree_test_accuracy
    )

with card3:
    accuracy_card(
        "다수 집단 기준 모델",
        baseline_train_accuracy,
        baseline_test_accuracy
    )

st.caption(
    "다수 집단 기준 모델은 입력 속성을 전혀 사용하지 않고, "
    "훈련 데이터에서 더 많은 쪽으로만 예측합니다."
)


# --------------------------------------------------
# 산점도에 사용할 두 속성 선택
# --------------------------------------------------

st.subheader("5. 분류 결과 시각화")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x]
    )

with axis_col2:
    y_options = [
        feature for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x]
    )


# --------------------------------------------------
# 테스트 데이터의 중앙값으로 나머지 속성 고정
# --------------------------------------------------

# 그림을 그릴 때 가로축과 세로축 이외의 속성은
# 테스트 데이터의 중앙값으로 고정한다.
fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(
            X_test[feature].median()
        )

if fixed_values:
    fixed_text = ", ".join(
        [
            f"{KOREAN_NAMES[feature]}: {value:.2f}"
            for feature, value in fixed_values.items()
        ]
    )

    st.write(
        "**나머지 속성의 고정값:** " + fixed_text
    )
else:
    st.write("선택한 두 속성 외에 고정할 속성은 없습니다.")


# --------------------------------------------------
# 결정 영역 계산
# --------------------------------------------------

# 테스트 데이터의 범위를 기준으로 격자 생성
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

x_grid = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    100
)

y_grid = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    100
)

xx, yy = np.meshgrid(x_grid, y_grid)


# 격자 데이터를 모델 입력 형태로 변환
grid_data = pd.DataFrame()

for feature in selected_features:

    if feature == x_feature:
        grid_data[feature] = xx.ravel()

    elif feature == y_feature:
        grid_data[feature] = yy.ravel()

    else:
        grid_data[feature] = fixed_values[feature]


# 로지스틱 회귀의 뇌졸중 확률
logistic_probability = logistic_model.predict_proba(
    grid_data[selected_features]
)[:, 1]

probability_grid = logistic_probability.reshape(
    xx.shape
)


# 의사결정트리의 예측 결과
tree_prediction = tree_model.predict(
    grid_data[selected_features]
)

tree_grid = tree_prediction.reshape(
    xx.shape
)


# --------------------------------------------------
# Plotly 산점도
# --------------------------------------------------

fig = go.Figure()


# 의사결정트리 결정 영역
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=tree_grid,
        colorscale=[
            [0.0, "rgba(80, 140, 220, 0.16)"],
            [0.5, "rgba(80, 140, 220, 0.16)"],
            [0.5, "rgba(230, 100, 100, 0.16)"],
            [1.0, "rgba(230, 100, 100, 0.16)"]
        ],
        showscale=False,
        contours=dict(
            coloring="fill",
            showlines=False
        ),
        hoverinfo="skip",
        name="의사결정트리 결정 영역"
    )
)


# 로지스틱 회귀의 0.5 확률 경계선
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=probability_grid,
        contours=dict(
            start=0.5,
            end=0.5,
            size=0.01,
            coloring="lines",
            showlines=True
        ),
        line=dict(
            color="black",
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계선"
    )
)


# 실제 뇌졸중 여부에 따라 테스트 데이터를 표시
test_plot = X_test.copy()
test_plot["stroke"] = y_test.to_numpy()

# 뇌졸중이 아닌 사람
negative = test_plot[
    test_plot["stroke"] == 0
]

fig.add_trace(
    go.Scatter(
        x=negative[x_feature],
        y=negative[y_feature],
        mode="markers",
        name="실제 뇌졸중 아님",
        marker=dict(
            color="#2878B5",
            size=7,
            opacity=0.7
        )
    )
)


# 뇌졸중인 사람
positive = test_plot[
    test_plot["stroke"] == 1
]

fig.add_trace(
    go.Scatter(
        x=positive[x_feature],
        y=positive[y_feature],
        mode="markers",
        name="실제 뇌졸중",
        marker=dict(
            color="#D94B4B",
            size=8,
            opacity=0.8
        )
    )
)


fig.update_layout(
    title="테스트 데이터와 두 모델의 분류 영역",
    xaxis_title=KOREAN_NAMES[x_feature],
    yaxis_title=KOREAN_NAMES[y_feature],
    legend_title="범례",
    height=650,
    template="plotly_white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 로지스틱 회귀 경계선이 그림 안에 있는지 확인
# --------------------------------------------------

# 격자 안에서 0.5에 가장 가까운 확률 확인
minimum_difference = np.min(
    np.abs(probability_grid - 0.5)
)

if minimum_difference > 0.02:
    st.warning(
        "로지스틱 회귀의 0.5 경계선이 현재 그림의 범위 안에 "
        "나타나지 않습니다. 경계선이 그림 밖에 있거나 "
        "현재 범위에서 0.5를 통과하지 않을 수 있습니다."
    )
else:
    st.success(
        "로지스틱 회귀의 0.5 경계선이 그림의 범위 안에서 "
        "확인됩니다."
    )

st.caption(
    "검은 선은 로지스틱 회귀 모델이 뇌졸중 확률 0.5를 기준으로 "
    "분류하는 경계입니다. 배경의 옅은 색은 의사결정트리 모델의 "
    "분류 영역입니다."
)


# --------------------------------------------------
# 의사결정트리 가지 그림 만들기
# --------------------------------------------------

st.subheader("6. 의사결정트리의 질문 구조")

tree = tree_model.tree_

# 훈련 데이터가 각 마디를 몇 번 통과했는지 계산
decision_path = tree_model.decision_path(
    X_train[selected_features]
)

node_counts = np.asarray(
    decision_path.sum(axis=0)
).ravel()

# 각 마디에 도착한 훈련 데이터 중 실제 뇌졸중 인원 계산
positive_counts = np.zeros(
    tree.node_count,
    dtype=int
)

for node_index in range(tree.node_count):

    row_indices = decision_path[:, node_index].nonzero()[0]

    positive_counts[node_index] = int(
        y_train.iloc[row_indices].sum()
    )


# DOT 문자열에 들어갈 특수문자 처리
def dot_escape(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


dot_lines = [
    "digraph Tree {",
    'graph [rankdir=TB, nodesep=0.35, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic", fontsize=10];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]


# 답을 내는 마디 수 계산
leaf_nodes = []

# 트리가 실제로 사용한 속성
used_features = set()


def make_tree_dot(node_index):

    sample_count = int(
        node_counts[node_index]
    )

    positive_count = int(
        positive_counts[node_index]
    )

    positive_rate = (
        positive_count / sample_count * 100
        if sample_count > 0
        else 0
    )

    left_child = tree.children_left[node_index]
    right_child = tree.children_right[node_index]

    # 답을 내는 마디
    if left_child == right_child:

        leaf_nodes.append(node_index)

        predicted_class = int(
            tree_model.classes_[
                np.argmax(tree.value[node_index][0])
            ]
        )

        if predicted_class == 1:
            answer = "뇌졸중"
            color = "#F5B5B5"
        else:
            answer = "아님"
            color = "#B8D5F0"

        label = (
            f"답: {answer}\\n"
            f"훈련 인원: {sample_count}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"실제 비율: {positive_rate:.1f}%"
        )

        dot_lines.append(
            f'{node_index} [label="{dot_escape(label)}", fillcolor="{color}"];'
        )

        return

    # 질문을 하는 마디
    feature_index = tree.feature[node_index]
    feature_name = FEATURES[0]

    # 모델 입력 순서에 맞게 실제 속성 이름 찾기
    feature_name = selected_features[feature_index]
    used_features.add(feature_name)

    threshold = tree.threshold[node_index]

    question = (
        f"{KOREAN_NAMES[feature_name]} "
        f"≤ {threshold:.2f} ?"
    )

    label = (
        f"{question}\\n"
        f"훈련 인원: {sample_count}명\\n"
        f"실제 뇌졸중: {positive_count}명\\n"
        f"실제 비율: {positive_rate:.1f}%"
    )

    dot_lines.append(
        f'{node_index} [label="{dot_escape(label)}", fillcolor="#EEEEEE"];'
    )

    # 왼쪽 가지: 조건을 만족하는 경우
    # 오른쪽 가지: 조건을 만족하지 않는 경우
    dot_lines.append(
        f'{node_index} -> {left_child} [label="예"];'
    )

    dot_lines.append(
        f'{node_index} -> {right_child} [label="아니요"];'
    )

    make_tree_dot(left_child)
    make_tree_dot(right_child)


make_tree_dot(0)

dot_lines.append("}")

tree_dot = "\n".join(dot_lines)

st.graphviz_chart(
    tree_dot,
    use_container_width=True
)


# --------------------------------------------------
# 의사결정트리 구조 요약
# --------------------------------------------------

st.subheader("7. 의사결정트리 분석")

total_leaf_count = len(leaf_nodes)

# 예측 결과가 아님인 잎 마디 수
negative_leaf_count = 0

for node_index in leaf_nodes:

    predicted_class = int(
        tree_model.classes_[
            np.argmax(tree.value[node_index][0])
        ]
    )

    if predicted_class == 0:
        negative_leaf_count += 1


st.write(
    f"**답을 내는 마디:** {total_leaf_count}칸"
)

st.write(
    f"**그중 뇌졸중이 아니라고 답하는 마디:** "
    f"{negative_leaf_count}칸"
)

if used_features:
    used_names = [
        KOREAN_NAMES[feature]
        for feature in selected_features
        if feature in used_features
    ]

    st.write(
        "**의사결정트리가 실제로 질문한 속성:** "
        + ", ".join(used_names)
    )
else:
    st.write(
        "**의사결정트리가 실제로 질문한 속성:** 없음"
    )

st.caption(
    "의사결정트리는 최대 세 단계까지만 질문하도록 설정했습니다. "
    "각 마디에 도착한 훈련 인원이 5명 미만인 경우에는 "
    "더 나누지 않습니다. 난수 기준값은 42로 고정했습니다."
)


# --------------------------------------------------
# 마무리 안내
# --------------------------------------------------

st.divider()

st.subheader("모델 비교 시 주의할 점")

st.write(
    "정확도는 전체 예측 중 실제 정답과 일치한 비율입니다. "
    "다만 뇌졸중 데이터처럼 뇌졸중이 아닌 사람이 훨씬 많은 자료에서는 "
    "정확도만으로 모델의 성능을 판단하기 어렵습니다."
)

st.write(
    "특히 뇌졸중을 양성으로 설정했으므로, "
    "실제 뇌졸중인 사람을 얼마나 잘 찾아내는지도 함께 살펴보는 것이 중요합니다."
)

st.warning(
    "이 페이지의 모델은 학습용 데이터로 만든 교육용 분류 모델입니다. "
    "실제 개인의 뇌졸중 위험을 진단하거나 의료적 판단을 대신할 수 없습니다."
)
