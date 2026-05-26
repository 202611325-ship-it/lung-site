import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------
# 페이지 설정
# -----------------------------------
st.set_page_config(
    page_title="폐암 위험도 분석 시스템",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------------
# CSS 스타일
# -----------------------------------
st.markdown("""
<style>

.main {
    background-color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #0F172A;
}

.subtitle {
    text-align: center;
    color: #64748B;
    font-size: 18px;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0F172A, #1E293B);
    color: white;
    border: none;
    border-radius: 14px;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1E293B, #334155);
    color: white;
}

.result-box {
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
}

.safe-box {
    background-color: #ECFDF5;
    border: 3px solid #10B981;
}

.warning-box {
    background-color: #FEF3C7;
    border: 3px solid #F59E0B;
}

.danger-box {
    background-color: #FEF2F2;
    border: 3px solid #EF4444;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# 제목
# -----------------------------------
st.markdown("""
<div class='title'>
🫁 폐암 위험도 분석 시스템
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
AI 기반 건강 데이터를 활용하여 폐암 위험도를 분석합니다.
</div>
""", unsafe_allow_html=True)

st.write("---")

# -----------------------------------
# 샘플 데이터 생성
# -----------------------------------
np.random.seed(42)

data_size = 1500

df = pd.DataFrame({
    "나이": np.random.randint(30, 81, data_size),
    "흡연": np.random.randint(1, 3, data_size),
    "음주": np.random.randint(1, 3, data_size),
    "기침": np.random.randint(1, 3, data_size),
    "호흡곤란": np.random.randint(1, 3, data_size),
    "가슴통증": np.random.randint(1, 3, data_size),
})

# -----------------------------------
# 위험 점수 생성
# -----------------------------------
risk_score = (
    df["흡연"] * 3
    + df["음주"] * 2
    + df["기침"] * 2
    + df["호흡곤란"] * 3
    + df["가슴통증"] * 3
    + (df["나이"] >= 60).astype(int) * 2
)

# 클래스 균형 맞추기
df["폐암위험"] = np.where(risk_score >= 10, 1, 0)

# -----------------------------------
# 모델 학습
# -----------------------------------
X = df.drop("폐암위험", axis=1)
y = df["폐암위험"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train)

# -----------------------------------
# 사용자 입력
# -----------------------------------
st.subheader("📋 환자 건강 정보 입력")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "👤 나이",
        min_value=30,
        max_value=100,
        value=50
    )

    smoking = st.selectbox(
        "🚬 흡연 여부",
        ["비흡연", "흡연"]
    )

    alcohol = st.selectbox(
        "🍺 음주 여부",
        ["적음", "많음"]
    )

with col2:

    cough = st.selectbox(
        "🤧 기침 여부",
        ["없음", "있음"]
    )

    breath = st.selectbox(
        "😮‍💨 호흡곤란 여부",
        ["없음", "있음"]
    )

    chest = st.selectbox(
        "💢 가슴통증 여부",
        ["없음", "있음"]
    )

# -----------------------------------
# 입력값 변환
# -----------------------------------
smoking_val = 2 if smoking == "흡연" else 1
alcohol_val = 2 if alcohol == "많음" else 1
cough_val = 2 if cough == "있음" else 1
breath_val = 2 if breath == "있음" else 1
chest_val = 2 if chest == "있음" else 1

# -----------------------------------
# 분석 버튼
# -----------------------------------
st.write("")

if st.button("🔍 위험도 분석하기"):

    input_data = pd.DataFrame({
        "나이": [age],
        "흡연": [smoking_val],
        "음주": [alcohol_val],
        "기침": [cough_val],
        "호흡곤란": [breath_val],
        "가슴통증": [chest_val],
    })

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    prediction = model.predict(input_scaled)[0]

    # 확률 안전 처리
    proba = model.predict_proba(input_scaled)

    if proba.shape[1] > 1:
        probability = proba[0][1]
    else:
        probability = 0.0

    st.markdown("## 📊 분석 결과")

    # 저위험군
    if probability < 0.35:

        st.markdown(f"""
        <div class="result-box safe-box">

            <h1 style="color:#059669;">
                ✅ 저위험군
            </h1>

            <h2 style="color:#065F46;">
                위험 확률 : {probability*100:.1f}%
            </h2>

            <p style="font-size:18px; color:#065F46;">
                현재 건강 상태는 비교적 안정적입니다.
            </p>

        </div>
        """, unsafe_allow_html=True)

    # 중위험군
    elif probability < 0.7:

        st.markdown(f"""
        <div class="result-box warning-box">

            <h1 style="color:#D97706;">
                ⚠️ 중위험군
            </h1>

            <h2 style="color:#92400E;">
                위험 확률 : {probability*100:.1f}%
            </h2>

            <p style="font-size:18px; color:#92400E;">
                생활 습관 개선이 필요합니다.
            </p>

        </div>
        """, unsafe_allow_html=True)

    # 고위험군
    else:

        st.markdown(f"""
        <div class="result-box danger-box">

            <h1 style="color:#DC2626;">
                🚨 고위험군
            </h1>

            <h2 style="color:#991B1B;">
                위험 확률 : {probability*100:.1f}%
            </h2>

            <p style="font-size:18px; color:#991B1B;">
                병원 정밀 검진을 권장합니다.
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 진행바
    st.progress(int(probability * 100))