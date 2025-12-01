import streamlit as st
import pandas as pd
import numpy as np

# --- 1. 데이터 생성 함수 ---
# st.cache_data를 사용하여 데이터를 한 번만 로드하고 캐시합니다.
@st.cache_data
def load_data():
    """랜덤 샘플 데이터를 생성하고 인위적으로 상관관계를 부여합니다."""
    # 50개의 행과 5개의 속성을 가진 데이터 생성
    data = {
        'A_Score': np.random.randint(60, 100, 50),
        'B_StudyHours': np.random.randint(2, 10, 50),
        'C_SleepHours': np.random.uniform(5, 9, 50),
        'D_ActivityLevel': np.random.randint(1, 6, 50),
        'E_StressLevel': np.random.randint(1, 10, 50)
    }
    df = pd.DataFrame(data)
    
    # 인위적으로 상관관계를 부여:
    # B_StudyHours (공부 시간)가 길수록 A_Score가 높아지도록 (양의 상관)
    df['A_Score'] = df['A_Score'] + (df['B_StudyHours'] * 3)
    # E_StressLevel (스트레스)가 높을수록 A_Score가 낮아지도록 (음의 상관)
    df['A_Score'] = df['A_Score'] - (df['E_StressLevel'] * 2)

    return df.round(2)

# --- 2. 상관관계 분석 함수 ---
def find_highest_correlation(corr_df, positive=True):
    """
    상관 행렬에서 가장 높거나(양의 상관) 가장 낮은(음의 상관) 관계를 찾습니다.
    (자기 자신과의 상관관계(1.0)는 제외하고 중복 쌍도 한 번만 계산)
    """
    # 상삼각 행렬을 만들고 대각선 (k=1)을 제외
    corr_array = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    
    if positive:
        # 가장 높은 양의 상관관계 (1.0 제외)
        max_corr = corr_array.stack().max()
        result = corr_array.stack().idxmax()
        return result, max_corr
    else:
        # 가장 낮은 값 (가장 높은 음의 상관관계)
        min_corr = corr_array.stack().min()
        result = corr_array.stack().idxmin()
        return result, min_corr

# --- Streamlit 앱 시작 ---
st.title("데이터 속성 간 상관관계 분석 앱 📊")
st.markdown("이 앱은 샘플 데이터셋을 사용하여 속성 간의 최고 양/음의 상관관계를 분석합니다.")
st.markdown("---")

df = load_data()
corr_matrix = df.corr()

# 1. 데이터 미리보기
st.header("1. 샘플 데이터")
st.dataframe(df)

# 2. 상관관계 행렬 시각화
st.header("2. 상관관계 행렬 (Correlation Matrix)")
# 배경색 그라데이션으로 상관관계를 시각적으로 표현
st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format(precision=2))

st.markdown("---")
st.header("3. 최고 상관관계 분석 결과")

col1, col2 = st.columns(2)

with col1:
    # --- 최고 양의 상관관계 버튼 ---
    if st.button("⬆️ 최고 양의 상관관계 분석", use_container_width=True):
        st.subheader("최고 양의 상관관계 결과:")
        
        (attr1, attr2), max_corr = find_highest_correlation(corr_matrix, positive=True)
        
        st.success(f"**{attr1}**과 **{attr2}**")
        st.metric("상관계수 (R)", f"{max_corr:.4f}")
        st.write("이 두 속성은 값이 **함께 증가하거나 감소**하는 경향이 가장 강합니다.")
        
        # 산점도 시각화
        st.caption(f"**{attr1}** vs **{attr2}** 산점도")
        st.scatter_chart(df, x=attr1, y=attr2)

with col2:
    # --- 최고 음의 상관관계 버튼 ---
    if st.button("⬇️ 최고 음의 상관관계 분석", use_container_width=True):
        st.subheader("최고 음의 상관관계 결과:")
        
        (attr1, attr2), min_corr = find_highest_correlation(corr_matrix, positive=False)
        
        st.error(f"**{attr1}**과 **{attr2}**")
        st.metric("상관계수 (R)", f"{min_corr:.4f}")
        st.write("이 두 속성은 한 값이 **증가할 때 다른 값이 감소**하는 경향이 가장 강합니다.")
        
        # 산점도 시각화
        st.caption(f"**{attr1}** vs **{attr2}** 산점도")
        st.scatter_chart(df, x=attr1, y=attr2)
