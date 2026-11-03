
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
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="⚖️",
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

st.title("⚖️ 성능 높이기")

st.write(
    "뇌졸중을 겪은 사람이 상대적으로 적은 데이터에서 "
    "클래스 가중치를 조정하면 분류 결과가 어떻게 달라지는지 비교합니다."
)

st.info(
    "stroke가 1이면 양성(뇌졸중), 0이면 음성(뇌졸중 아님)입니다. "
    "가중치 조정 전후 모두 같은 테스트 데이터를 사용합니다."
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

st.subheader("2. 데이터 분리")

positive_test_count = int((y_test == 1).sum())

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
# 모델 생성 함수
# --------------------------------------------------

def make_logistic_model(class_weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight=class_weight
        ))
    ])


def make_tree_model(class_weight):
    return DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight=class_weight
    )


# --------------------------------------------------
# 가중치 조정 전 모델
# --------------------------------------------------

logistic_before = make_logistic_model(None)
tree_before = make_tree_model(None)

logistic_before.fit(X_train, y_train)
tree_before.fit(X_train, y_train)


# --------------------------------------------------
# 가중치 조정 후 모델
# --------------------------------------------------

logistic_after = make_logistic_model("balanced")
tree_after = make_tree_model("balanced")

logistic_after.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


# --------------------------------------------------
# 평가 함수
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

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    precision = (
        tp / predicted_positive
        if predicted_positive > 0
        else None
    )

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


# --------------------------------------------------
# 테스트 데이터 예측
# --------------------------------------------------

predictions = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "전": logistic_before.predict(X_test),
        "후": logistic_after.predict(X_test)
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "전": tree_before.predict(X_test),
        "후": tree_after.predict(X_test)
    }
}

scores = {}

for model_name, result in predictions.items():
    scores[model_name] = {
        "전": calculate_scores(y_test, result["전"]),
        "후": calculate_scores(y_test, result["후"])
    }


# --------------------------------------------------
# 혼동행렬 및 지표 비교표
# --------------------------------------------------

st.subheader("3. 가중치 조정 전후 성능 비교")

st.write(
    "TN: 실제 아님을 아님으로 예측 / "
    "FP: 실제 아님을 뇌졸중으로 예측 / "
    "FN: 실제 뇌졸중을 아님으로 예측 / "
    "TP: 실제 뇌졸중을 뇌졸중으로 예측"
)


def format_number(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value * 100:.2f}%"


comparison_rows = []

for model_name in predictions:
    for stage in ["전", "후"]:
        result = scores[model_name][stage]

        comparison_rows.append({
            "모델": model_name,
            "가중치": "조정 전" if stage == "전" else "조정 후",
            "TN": result["TN"],
            "FP": result["FP"],
            "FN": result["FN"],
            "TP": result["TP"],
            "정확도": format_number(result["정확도"]),
            "재현율": format_number(result["재현율"]),
            "정밀도": format_number(result["정밀도"]),
            "F1": format_number(result["F1"])
        })

comparison_table = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison_table,
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
# 환자 발견 및 오류 인원 변화
# --------------------------------------------------

st.subheader("4. 환자 발견 및 오류 인원 변화")

for model_name in predictions:
    before = scores[model_name]["전"]
    after = scores[model_name]["후"]

    st.markdown(f"**{model_name}**")

    st.write(
        f"찾아낸 환자(TP): {before['TP']:,}명 → {after['TP']:,}명 / "
        f"놓친 환자(FN): {before['FN']:,}명 → {after['FN']:,}명 / "
        f"헛짚은 사람(FP): {before['FP']:,}명 → {after['FP']:,}명"
    )


# --------------------------------------------------
# 시각화에 사용할 축 선택
# --------------------------------------------------

st.subheader("5. 결정 경계 시각화")

if len(selected_features) == 2:

    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x]
    )

    y_feature = st.selectbox(
        "세로축 속성",
        options=[
            feature for feature in selected_features
            if feature != x_feature
        ],
        format_func=lambda x: KOREAN_NAMES[x]
    )

    z_feature = None

else:

    st.write("선택한 속성 가운데 세 가지를 3차원 축으로 지정하세요.")

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축 속성",
            options=selected_features,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x]
        )

    remaining_y = [
        feature for feature in selected_features
        if feature != x_feature
    ]

    with axis_col2:
        y_feature = st.selectbox(
            "세로축 속성",
            options=remaining_y,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x]
        )

    remaining_z = [
        feature for feature in selected_features
        if feature not in [x_feature, y_feature]
    ]

    with axis_col3:
        z_feature = st.selectbox(
            "깊이축 속성",
            options=remaining_z,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x]
        )


# --------------------------------------------------
# 나머지 속성 고정값
# --------------------------------------------------

axis_features = [x_feature, y_feature]

if z_feature is not None:
    axis_features.append(z_feature)

fixed_values = {}

for feature in selected_features:
    if feature not in axis_features:
        fixed_values[feature] = float(X_test[feature].median())

if fixed_values:
    fixed_text = ", ".join(
        [
            f"{KOREAN_NAMES[feature]} {value:.2f}"
            for feature, value in fixed_values.items()
        ]
    )

    st.write(
        "**그림에서 고정한 나머지 속성의 값:** " + fixed_text
    )
else:
    st.write("축으로 선택하지 않은 속성이 없습니다.")


# --------------------------------------------------
# 그림 범위 설정
# --------------------------------------------------

def make_axis_range(series):
    minimum = float(series.min())
    maximum = float(series.max())

    padding = (maximum - minimum) * 0.08

    if padding == 0:
        padding = 1.0

    return minimum - padding, maximum + padding


x_min, x_max = make_axis_range(X_test[x_feature])
y_min, y_max = make_axis_range(X_test[y_feature])


# --------------------------------------------------
# 모델별 확률 계산 함수
# --------------------------------------------------

models = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "전": logistic_before,
        "후": logistic_after
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "전": tree_before,
        "후": tree_after
    }
}


def make_prediction_frame(x_values, y_values, z_values=None):
    """
    축으로 지정한 값과 고정값을 이용해
    모델에 넣을 데이터프레임을 만든다.
    """

    frame = pd.DataFrame(index=np.arange(len(x_values)))

    for feature in selected_features:

        if feature == x_feature:
            frame[feature] = x_values

        elif feature == y_feature:
            frame[feature] = y_values

        elif z_feature is not None and feature == z_feature:
            frame[feature] = z_values

        else:
            frame[feature] = fixed_values[feature]

    return frame[selected_features]


def get_probability(model, frame):
    return model.predict_proba(frame)[:, 1]


# --------------------------------------------------
# 2차원 산점도와 결정 경계
# --------------------------------------------------

if z_feature is None:

    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)

    xx, yy = np.meshgrid(x_grid, y_grid)

    grid_frame = make_prediction_frame(
        xx.ravel(),
        yy.ravel()
    )

    colors = {
        ("로지스틱 회귀(확률로 답하는 모델)", "전"): "#E45756",
        ("로지스틱 회귀(확률로 답하는 모델)", "후"): "#2CA02C",
        ("의사결정트리(질문으로 답하는 모델)", "전"): "#F28E2B",
        ("의사결정트리(질문으로 답하는 모델)", "후"): "#9467BD"
    }

    fig = go.Figure()

    # 네 가지 모델 상태의 0.5 경계선
    boundary_status = []

    for model_name in models:

        for stage in ["전", "후"]:

            model = models[model_name][stage]

            probability = get_probability(
                model,
                grid_frame
            ).reshape(xx.shape)

            color = colors[(model_name, stage)]

            # 확률이 0.5를 통과하는 경우에만 경계선 표시
            crosses = (
                probability.min() <= 0.5 <= probability.max()
            )

            if crosses:
                fig.add_trace(
                    go.Contour(
                        x=x_grid,
                        y=y_grid,
                        z=probability,
                        contours=dict(
                            start=0.5,
                            end=0.5,
                            size=0.01,
                            coloring="lines",
                            showlines=True
                        ),
                        line=dict(
                            color=color,
                            width=3
                        ),
                        showscale=False,
                        hoverinfo="skip",
                        name=f"{model_name} {stage}"
                    )
                )

                boundary_status.append(
                    f"{model_name} {stage}: 그림 안에 경계선이 있습니다."
                )

            else:
                boundary_status.append(
                    f"{model_name} {stage}: 0.5 경계선이 그림 범위 밖에 있습니다."
                )

    # 실제 테스트 데이터 표시
    plot_data = X_test.copy()
    plot_data["실제"] = y_test.to_numpy()

    negative = plot_data[plot_data["실제"] == 0]
    positive = plot_data[plot_data["실제"] == 1]

    fig.add_trace(
        go.Scatter(
            x=negative[x_feature],
            y=negative[y_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#2878B5",
                size=7,
                opacity=0.65
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=positive[x_feature],
            y=positive[y_feature],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#D62728",
                size=8,
                opacity=0.8
            )
        )
    )

    fig.update_layout(
        title="가중치 조정 전후 0.5 결정 경계",
        xaxis_title=KOREAN_NAMES[x_feature],
        yaxis_title=KOREAN_NAMES[y_feature],
        height=650,
        template="plotly_white",
        legend_title="범례"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    for message in boundary_status:
        st.caption(message)

    st.caption(
        "실제 테스트 데이터는 색으로 구분했습니다. "
        "경계선은 각 모델이 뇌졸중일 확률 0.5를 기준으로 "
        "분류하는 위치를 나타냅니다."
    )


# --------------------------------------------------
# 3차원 산점도와 결정 평면
# --------------------------------------------------

else:

    z_min, z_max = make_axis_range(X_test[z_feature])

    x_grid = np.linspace(x_min, x_max, 35)
    y_grid = np.linspace(y_min, y_max, 35)
    z_grid = np.linspace(z_min, z_max, 45)

    xx, yy = np.meshgrid(x_grid, y_grid)

    # 각 x-y 위치에서 확률이 0.5가 되는 z값을 찾는다.
    colors = {
        ("로지스틱 회귀(확률로 답하는 모델)", "전"): "#E45756",
        ("로지스틱 회귀(확률로 답하는 모델)", "후"): "#2CA02C",
        ("의사결정트리(질문으로 답하는 모델)", "전"): "#F28E2B",
        ("의사결정트리(질문으로 답하는 모델)", "후"): "#9467BD"
    }

    fig = go.Figure()

    plane_status = []

    for model_name in models:

        for stage in ["전", "후"]:

            model = models[model_name][stage]

            # x, y, z의 모든 조합 생성
            mesh_x = np.repeat(xx.ravel(), len(z_grid))
            mesh_y = np.repeat(yy.ravel(), len(z_grid))
            mesh_z = np.tile(z_grid, xx.size)

            mesh_frame = make_prediction_frame(
                mesh_x,
                mesh_y,
                mesh_z
            )

            probabilities = get_probability(
                model,
                mesh_frame
            ).reshape(xx.size, len(z_grid))

            # 각 x-y 위치에서 0.5와 가장 가까운 z값 탐색
            surface_z = np.full(xx.size, np.nan)
            has_crossing = False

            for index in range(xx.size):

                values = probabilities[index]

                # z축 범위에서 0.5를 통과하는지 확인
                if values.min() <= 0.5 <= values.max():

                    has_crossing = True

                    differences = values - 0.5
                    crossing_indices = np.where(
                        differences[:-1] * differences[1:] <= 0
                    )[0]

                    if len(crossing_indices) > 0:

                        z_index = crossing_indices[0]

                        z1 = z_grid[z_index]
                        z2 = z_grid[z_index + 1]

                        p1 = values[z_index]
                        p2 = values[z_index + 1]

                        if p2 != p1:
                            ratio = (0.5 - p1) / (p2 - p1)
                            surface_z[index] = z1 + ratio * (z2 - z1)
                        else:
                            surface_z[index] = z1

            if has_crossing and not np.all(np.isnan(surface_z)):

                color = colors[(model_name, stage)]

                fig.add_trace(
                    go.Surface(
                        x=xx,
                        y=yy,
                        z=surface_z.reshape(xx.shape),
                        opacity=0.35,
                        showscale=False,
                        colorscale=[
                            [0, color],
                            [1, color]
                        ],
                        name=f"{model_name} {stage}",
                        hovertemplate=(
                            f"{KOREAN_NAMES[x_feature]}: %{{x:.2f}}<br>"
                            f"{KOREAN_NAMES[y_feature]}: %{{y:.2f}}<br>"
                            f"{KOREAN_NAMES[z_feature]}: %{{z:.2f}}"
                            "<extra></extra>"
                        )
                    )
                )

                plane_status.append(
                    f"{model_name} {stage}: 0.5 결정 평면을 표시했습니다."
                )

            else:
                plane_status.append(
                    f"{model_name} {stage}: 현재 그림 범위에서 "
                    "0.5 결정 평면을 찾을 수 없습니다."
                )

    # 실제 테스트 데이터
    plot_data = X_test.copy()
    plot_data["실제"] = y_test.to_numpy()

    negative = plot_data[plot_data["실제"] == 0]
    positive = plot_data[plot_data["실제"] == 1]

    fig.add_trace(
        go.Scatter3d(
            x=negative[x_feature],
            y=negative[y_feature],
            z=negative[z_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#2878B5",
                size=4,
                opacity=0.65
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=positive[x_feature],
            y=positive[y_feature],
            z=positive[z_feature],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#D62728",
                size=5,
                opacity=0.85
            )
        )
    )

    fig.update_layout(
        title="3차원 가중치 조정 전후 0.5 결정 평면",
        scene=dict(
            xaxis_title=KOREAN_NAMES[x_feature],
            yaxis_title=KOREAN_NAMES[y_feature],
            zaxis_title=KOREAN_NAMES[z_feature]
        ),
        height=750,
        template="plotly_white",
        legend_title="범례"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    for message in plane_status:
        st.caption(message)

    st.caption(
        "마우스로 드래그하면 3차원 그림을 회전할 수 있습니다. "
        "나머지 입력 속성은 테스트 데이터의 중앙값으로 고정했습니다."
    )


# --------------------------------------------------
# 결과 해석 안내
# --------------------------------------------------

st.divider()

st.subheader("6. 결과를 해석할 때 주의할 점")

st.write(
    "클래스 가중치 조정은 훈련 과정에서 소수 집단인 뇌졸중 사례의 "
    "오분류에 더 큰 비중을 주는 방법입니다. "
    "따라서 뇌졸중을 찾아내는 재현율이 높아질 수 있지만, "
    "뇌졸중이 아닌 사람을 뇌졸중이라고 예측하는 FP도 늘어날 수 있습니다."
)

st.write(
    "정확도만 비교하기보다는 재현율, 정밀도, F1과 함께 "
    "놓친 환자(FN)와 헛짚은 사람(FP)이 어떻게 바뀌었는지 살펴보세요."
)

st.warning(
    "이 페이지는 교육용 분류 실습입니다. "
    "실제 개인의 뇌졸중 위험을 진단하거나 의료적 판단을 대신할 수 없습니다."
)
