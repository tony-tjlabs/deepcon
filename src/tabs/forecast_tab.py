"""
DeepCon Forecast 탭 - 평시 위험도 패턴 분석
==========================================

주중/주말 평균 위험도 히트맵 표시
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from src.cached_data_loader import CachedDataLoader


def render_forecast_tab(loader: CachedDataLoader):
    """DeepCon Forecast 탭 렌더링"""
    
    st.markdown('<div class="main-header"><h1>🔮 DeepCon Forecast</h1><p>평시 위험도 패턴 분석 (주중/주말 평균)</p></div>', unsafe_allow_html=True)
    
    # 설명
    st.markdown("""
    ### 📊 평시 위험도 패턴이란?
    
    **DeepCon Forecast**는 과거 7일간의 데이터를 분석하여 **평시(정상) 상황의 위험도 패턴**을 보여줍니다.
    
    - **주중 평균**: 수~금, 월~화 (5일) 데이터의 평균
    - **주말 평균**: 토~일 (2일) 데이터의 평균
    
    이 패턴은 **DeepCon Simulator**에서 실시간 모니터링 시 비정상 상황을 감지하는 기준으로 사용됩니다.
    
    ---
    """)
    
    # 위험도 계산 로직 설명
    with st.expander("📝 위험도 계산 로직 설명", expanded=False):
        st.markdown("""
        ### 위험도 계산 5가지 핵심 요소
        
        DeepCon 위험도는 다음 5가지 요소를 종합하여 계산됩니다:
        
        #### 1️⃣ **면적 대비 인원 밀집도** (가중치: 25%)
        - 단순 인원수가 아닌 **면적 당 작업자 수** (명/m²) 기준
        - 밀집도 기준:
          - 0.05명/m² 미만: 낮음 (위험도 0.1)
          - 0.10명/m² 미만: 중간 (위험도 0.3)
          - 0.15명/m² 미만: 높음 (위험도 0.6)
          - 0.15명/m² 이상: 매우 높음 (위험도 0.9)
        
        #### 2️⃣ **구역 고유 위험도** (가중치: 20%)
        - 작업 공간 특성에 따른 기본 위험도:
          - **밀폐공간** (TB02_밀폐, TB17_밀폐 등): 0.40
          - **작업공간** (WWT, FAB, CUB): 0.30
          - **휴게실/흡연실**: 0.05
          - **기타**: 0.10
        
        #### 3️⃣ **작업자-장비 혼재** (가중치: 15%)
        - 작업자와 장비가 동시에 있는 경우 위험도 상승
        - 장비 3대 이상 + 작업자 5명 이상: 고위험 (0.7)
        - 장비 2대 이상 + 작업자 3명 이상: 중위험 (0.5)
        
        #### 4️⃣ **과거 패턴 대비 편차** ⭐ (가중치: 35% - 가장 중요!)
        - 평시 패턴 대비 현재 상황의 비정상 정도
        - 표준편차(σ) 배수로 평가:
          - 1σ 미만: 정상 (위험도 0.1)
          - 2σ 미만: 약간 비정상 (위험도 0.4)
          - 3σ 미만: 비정상 (위험도 0.7)
          - 3σ 이상: **매우 비정상** (위험도 0.95) 🚨
        
        #### 5️⃣ **밀폐공간 예외 처리** (가중치: 5%)
        - 밀폐공간은 기본 위험도가 높지만, 작업자가 없으면 실제 위험도 감소
        - 작업자 0명: 위험도 -0.5
        - 작업자 1~2명: 위험도 -0.2
        
        ---
        
        ### 🔧 스케일링 팩터 (곱셈 적용)
        
        기본 위험도에 추가로 다음 팩터들이 곱해집니다:
        
        #### 📍 **체류시간 팩터** (1.0~1.3배)
        - **10분 미만**: 1.0배 (정상)
        - **10~30분**: 1.0~1.1배 (약간 증가)
        - **30~60분**: 1.1~1.2배 (증가)
        - **60분 이상**: 1.3배 (큰 증가)
        
        > 💡 한 구역에 오래 머무를수록 위험도 증가
        
        #### 🕐 **시간대 팩터** (1.0~1.25배)
        - **새벽 (00:00~05:00)**: 1.25배 ⚠️ 가장 위험
        - **야간 (22:00~23:59)**: 1.15배 ⚠️
        - **이른 아침 (05:00~07:00)**: 1.08배
        - **정상 근무 (07:00~22:00)**: 1.0배
        
        > 💡 새벽/야간 작업 시 위험도 증가
        
        **최종 위험도 = 기본 위험도 × 체류시간 팩터 × 시간대 팩터**
        
        ---
        
        ### 최종 위험도 등급
        - **Safe** (0~0.3): 🟢 정상
        - **Caution** (0.3~0.6): 🟡 주의
        - **Critical** (0.6~1.0): 🔴 위험
        
        ---
        
        💡 **평시 데이터는 사고가 없는 정상 상황이므로, Critical 케이스는 거의 없습니다.**
        """)
    
    # 주중/주말 선택
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        period_type = st.radio(
            "📅 기간 선택",
            ['주중', '주말'],
            index=0,
            help="주중: 수~금+월~화 (5일 평균) | 주말: 토~일 (2일 평균)"
        )
        
        st.info(f"""
        **{period_type} 데이터**
        
        {'• 12/10(수), 12/11(목), 12/12(금)' if period_type == '주중' else '• 12/13(토), 12/14(일)'}
        {'• 12/15(월), 12/16(화)' if period_type == '주중' else ''}
        
        총 {'5일' if period_type == '주중' else '2일'} 평균
        """)
    
    with col2:
        # 데이터 로드
        cache_file = Path('Cache') / f'forecast_{"weekday" if period_type == "주중" else "weekend"}_avg.parquet'
        
        if not cache_file.exists():
            st.error(f"❌ {period_type} 평균 데이터가 없습니다. `python src/precompute_forecast.py`를 먼저 실행하세요.")
            return
        
        df = pd.read_parquet(cache_file)
        
        st.markdown(f"### 🌡️ {period_type} 평균 위험도 히트맵")
        
        # 통계 요약
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("평균 위험도", f"{df['avg_risk'].mean():.3f}")
        with col_b:
            st.metric("최대 위험도", f"{df['avg_risk'].max():.3f}")
        with col_c:
            safe_pct = len(df[df['risk_level'] == 'Safe']) / len(df) * 100
            st.metric("Safe 비율", f"{safe_pct:.1f}%")
        with col_d:
            critical_cnt = len(df[df['risk_level'] == 'Critical'])
            st.metric("Critical 건수", f"{critical_cnt}")
        
        # Zone 리스트 추출 및 이름순 정렬 (관련 Zone이 함께 보이도록)
        zone_names = sorted(df['spot_name'].unique())
        
        # 시간 레이블 생성 (288개)
        time_labels = []
        for t in range(288):
            minutes = t * 5
            hh = (minutes // 60) % 24
            mm = minutes % 60
            time_labels.append(f"{hh:02d}:{mm:02d}")
        
        # Pivot 테이블 생성 (zone x time)
        pivot = df.pivot_table(
            index='spot_name',
            columns='time_index',
            values='avg_risk',
            fill_value=0
        )
        
        # Zone 순서 맞추기
        pivot = pivot.reindex(zone_names)
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[time_labels[i-1] for i in pivot.columns if 1 <= i <= 288],
            y=pivot.index,
            colorscale=[
                [0.0, '#E8F4F8'],    # 매우 낮음
                [0.2, '#B3E5FC'],    # 낮음
                [0.3, '#4FC3F7'],    # 중하
                [0.5, '#FFD54F'],    # 중간
                [0.7, '#FF9800'],    # 중상
                [1.0, '#D32F2F']     # 높음
            ],
            colorbar=dict(
                title="위험도",
                tickmode="linear",
                tick0=0,
                dtick=0.1,
                tickfont=dict(color="#000000", size=10),
                len=0.7
            ),
            hovertemplate='<b>%{y}</b><br>시간: %{x}<br>위험도: %{z:.3f}<extra></extra>',
            zmin=0,
            zmax=1.0
        ))
        
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#000000",
            height=max(600, len(zone_names) * 20),
            margin=dict(l=10, r=100, t=30, b=50),
            xaxis=dict(
                title="시간",
                tickfont=dict(color="#000000", size=8),
                showgrid=True,
                gridcolor='#E0E0E0',
                side='top',
                tickmode='array',
                tickvals=[time_labels[0], time_labels[72], time_labels[144], time_labels[216], time_labels[287]],
                ticktext=['00:00', '06:00', '12:00', '18:00', '23:55']
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color="#000000", size=10),
                showgrid=True,
                gridcolor='#E0E0E0',
                autorange="reversed"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 **평시 패턴**: 정상 운영 시의 위험도 분포 | 🟢 Safe (<0.3) | 🟡 Caution (0.3~0.6) | 🔴 Critical (>0.6)")
        
        # 고위험 구역 TOP 10
        st.markdown("### ⚠️ 주요 관심 구역 TOP 10")
        
        top_zones = df.groupby('spot_name').agg({
            'avg_risk': 'mean',
            'avg_worker': 'mean',
            'max_worker': 'max',
            'zone_type': 'first'
        }).sort_values('avg_risk', ascending=False).head(10)
        
        top_zones = top_zones.reset_index()
        top_zones.columns = ['구역명', '평균 위험도', '평균 작업자', '최대 작업자', '구역 타입']
        top_zones['평균 위험도'] = top_zones['평균 위험도'].apply(lambda x: f"{x:.3f}")
        top_zones['평균 작업자'] = top_zones['평균 작업자'].apply(lambda x: f"{x:.1f}명")
        top_zones['최대 작업자'] = top_zones['최대 작업자'].apply(lambda x: f"{int(x)}명")
        
        st.dataframe(top_zones, use_container_width=True, hide_index=True)
