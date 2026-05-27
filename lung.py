import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.graph_objects as go

# =========================================================
# 한글 깨짐 방지
# =========================================================

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="폐암 환자 군집 분석 시스템",
    page_icon="🫁",
    layout="wide"
)

# =========================================================
# 제목
# =========================================================

st.title("🫁 폐암 환자 군집 분석 시스템")
st.caption("생활 습관 데이터를 기반으로 환자 군집을 분석합니다.")

# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    return df

df = load_data()

# =========================================================
# 컬럼명 변경
# =========================================================

df = df.rename(columns={
    'GENDER': '성별',
    'AGE': '나이',
    'SMOKING': '흡연',
    'YELLOW_FINGERS': '손가락변색',
    'ANXIETY': '불안',
    'PEER_PRESSURE': '또래영향',
    'CHRONIC_DISEASE': '만성질환',
    'FATIGUE': '피로',
    'ALLERGY': '알레르기',
    'WHEEZING': '쌕쌕거림',
    'ALCOHOL_CONSUMING': '음주',
    'COUGHING': '기침',
    'SHORTNESS_OF_BREATH': '호흡곤란',
    'SWALLOWING_DIFFICULTY': '삼킴곤란',
    'CHEST_PAIN': '가슴통증',
    'LUNG_CANCER': '폐암여부'
})

# =========================================================
# 문자 → 숫자 변환
# =========================================================

df['성별'] = df['성별'].map({'M': 1, 'F': 0})
df['폐암여부'] = df['폐암여부'].map({'YES': 1, 'NO': 0})

# =========================================================
# 특성 공학
# =========================================================

df['증상점수'] = df[['기침', '호흡곤란', '쌕쌕거림', '가슴통증', '삼킴곤란']].sum(axis=1)

df['생활위험'] = (
    df['흡연'] +
    df['음주'] +
    df['또래영향']
)

df['흡연기침'] = df['흡연'] * df['기침']

df['호흡문제'] = (
    df['호흡곤란'] +
    df['쌕쌕거림']
)

df['건강취약'] = (
    df['만성질환'] +
    df['피로']
)

df['고령'] = (df['나이'] >= 60).astype(int)

# =========================================================
# 군집화(KMeans)
# =========================================================

cluster_features = ['흡연', '음주', '나이']

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df['군집'] = kmeans.fit_predict(df[cluster_features])

# =========================================================
# 군집 설명
# =========================================================

cluster_info = {
    0: "매우 건강한 생활 패턴",
    1: "평균적인 위험도",
    2: "비교적 건강한 그룹",
    3: "높은 위험도를 가진 그룹"
}

# =========================================================
# KNN 모델 학습
# =========================================================

X = df.drop(['폐암여부', '군집'], axis=1)
y = df['폐암여부']

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

pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, pred)

# =========================================================
# 사이드바
# =========================================================

st.sidebar.header("환자 정보 입력")

흡연입력 = st.sidebar.slider(
    "🚬 흡연 정도",
    0,
    35,
    3
)

음주입력 = st.sidebar.slider(
    "🍺 음주 정도",
    0,
    10,
    2
)

나이입력 = st.sidebar.slider(
    "🧑 나이",
    30,
    80,
    50
)

성별입력 = st.sidebar.selectbox(
    "성별 선택",
    ["남성", "여성"]
)

# =========================================================
# 새 환자 데이터
# =========================================================

성별값 = 1 if 성별입력 == "남성" else 0

new_patient = pd.DataFrame([{
    '성별': 성별값,
    '나이': 나이입력,
    '흡연': 2 if 흡연입력 >= 10 else 1,
    '손가락변색': 1,
    '불안': 1,
    '또래영향': 1,
    '만성질환': 1,
    '피로': 1,
    '알레르기': 1,
    '쌕쌕거림': 1,
    '음주': 2 if 음주입력 >= 5 else 1,
    '기침': 1,
    '호흡곤란': 1,
    '삼킴곤란': 1,
    '가슴통증': 1
}])

# =========================================================
# 특성 공학 동일 적용
# =========================================================

new_patient['증상점수'] = new_patient[
    ['기침', '호흡곤란', '쌕쌕거림', '가슴통증', '삼킴곤란']
].sum(axis=1)

new_patient['생활위험'] = (
    new_patient['흡연'] +
    new_patient['음주'] +
    new_patient['또래영향']
)

new_patient['흡연기침'] = (
    new_patient['흡연'] *
    new_patient['기침']
)

new_patient['호흡문제'] = (
    new_patient['호흡곤란'] +
    new_patient['쌕쌕거림']
)

new_patient['건강취약'] = (
    new_patient['만성질환'] +
    new_patient['피로']
)

new_patient['고령'] = (
    new_patient['나이'] >= 60
).astype(int)

# =========================================================
# 폐암 예측
# =========================================================

new_scaled = scaler.transform(new_patient)

prediction = model.predict(new_scaled)[0]

probability = model.predict_proba(new_scaled)[0][1]

# =========================================================
# 군집 예측
# =========================================================

cluster_result = kmeans.predict(
    [[흡연입력, 음주입력, 나이입력]]
)[0]

# =========================================================
# 건강 점수 계산
# =========================================================

health_score = 100 - (
    흡연입력 * 1.7 +
    음주입력 * 2 +
    (나이입력 - 30) * 0.5
)

health_score = max(0, min(100, int(health_score)))

risk_score = 100 - health_score

# =========================================================
# 분석 결과
# =========================================================

st.divider()

st.header("📊 분석 결과")

st.subheader(f"예측 군집 : {cluster_result}번 군집")

if cluster_result == 0:
    st.success(f"🟢 {cluster_info[0]}")

elif cluster_result == 1:
    st.warning(f"🟡 {cluster_info[1]}")

elif cluster_result == 2:
    st.info(f"🔵 {cluster_info[2]}")

else:
    st.error(f"🔴 {cluster_info[3]}")

# =========================================================
# 건강도 분석
# =========================================================

st.divider()

st.header("💚 건강도 분석")

col1, col2 = st.columns(2)

with col1:

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "건강 점수"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 70], 'color': "#fff4cc"},
                {'range': [70, 100], 'color': "#d4ffd4"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

with col2:

    st.metric("폐 건강 위험도", f"{risk_score}%")
    st.metric("건강 점수", f"{health_score}점")

    if risk_score < 30:
        st.success("위험도가 낮은 상태입니다.")

    elif risk_score < 60:
        st.warning("생활 습관 개선이 필요합니다.")

    else:
        st.error("폐 건강 관리가 필요합니다.")

# =========================================================
# 군집 시각화
# =========================================================

st.divider()

st.header("📈 군집 시각화")

fig, ax = plt.subplots(figsize=(9, 6))

colors = ['green', 'orange', 'blue', 'red']

for i in range(4):

    cluster_data = df[df['군집'] == i]

    ax.scatter(
        cluster_data['흡연'],
        cluster_data['음주'],
        color=colors[i],
        alpha=0.5,
        s=70,
        label=f"{i}번 군집"
    )

# 새 환자 표시
ax.scatter(
    흡연입력,
    음주입력,
    color='black',
    marker='X',
    s=350,
    label='새 환자'
)

ax.set_xlabel("흡연")
ax.set_ylabel("음주")
ax.set_title("폐암 환자 군집")

ax.legend()

st.pyplot(fig)

# =========================================================
# 군집 특징
# =========================================================

st.divider()

st.header("📌 군집 특징")

st.info("🟢 0번 군집 : 매우 건강한 생활 패턴")
st.info("🟡 1번 군집 : 평균적인 위험도")
st.info("🔵 2번 군집 : 비교적 건강한 그룹")
st.info("🔴 3번 군집 : 높은 위험도를 가진 그룹")

# =========================================================
# AI 건강 조언
# =========================================================

st.divider()

st.header("🧑‍⚕️ AI 건강 조언")

if cluster_result == 0:

    st.success("""
    현재 매우 건강한 상태입니다.

    ✔ 규칙적인 운동 유지  
    ✔ 충분한 수면 유지  
    ✔ 건강검진 꾸준히 받기
    """)

elif cluster_result == 1:

    st.warning("""
    평균적인 위험도를 가진 상태입니다.

    ✔ 흡연량 줄이기  
    ✔ 음주 습관 개선  
    ✔ 유산소 운동 추천
    """)

elif cluster_result == 2:

    st.info("""
    비교적 건강한 상태입니다.

    ✔ 현재 생활 습관 유지  
    ✔ 스트레스 관리 필요  
    ✔ 정기적인 폐 건강 검사 권장
    """)

else:

    st.error("""
    높은 위험도를 가진 상태입니다.

    ✔ 금연 필수  
    ✔ 음주 조절 필요  
    ✔ 병원 검진 권장  
    ✔ 폐 건강 관리 필요
    """)

# =========================================================
# 입력 정보
# =========================================================

st.divider()

st.header("🧾 입력된 환자 정보")

st.write(f"🚬 흡연 정도 : {흡연입력}")
st.write(f"🍺 음주 정도 : {음주입력}")
st.write(f"🧑 나이 : {나이입력}")
st.write(f"👨 성별 : {성별입력}")

# =========================================================
# 모델 성능
# =========================================================

st.divider()

st.header("🤖 모델 성능")

st.write(f"KNN 모델 정확도 : {round(accuracy * 100, 2)}%")

# =========================================================
# 혼돈행렬
# =========================================================

cm = confusion_matrix(y_test, pred)

fig2, ax2 = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax2
)

ax2.set_xlabel("예측값")
ax2.set_ylabel("실제값")
ax2.set_title("혼돈 행렬")

st.pyplot(fig2)