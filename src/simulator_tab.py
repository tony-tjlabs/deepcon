"""
DeepCon Simulator 탭
====================

실시간 위험도 모니터링 및 30분 후 예측

Features:
- 날짜/시간 선택
- 현재 위험도 히트맵
- 30분 후 예측 히트맵
- 위험 요인 상세 분석
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta


# 사용 가능한 날짜
AVAILABLE_DATES = {
    '20251210': '2025-12-10 (화)',
    '20251211': '2025-12-11 (수)',
    '20251212': '2025-12-12 (목)',
    '20251213': '2025-12-13 (금)',
    '20251214': '2025-12-14 (토)'
}


def format_time_index(time_idx: int) -> str:
    """시간 인덱스를 HH:MM 형식으로 변환"""
    minutes = (time_idx - 1) * 5
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def get_time_index_from_time(hour: int, minute: int) -> int:
    """시간(HH:MM)을 time_index로 변환"""
    total_minutes = hour * 60 + minute
    return (total_minutes // 5) + 1


def load_simulator_data(date_str: str) -> pd.DataFrame:
    """시뮬레이터 데이터 로드"""
    cache_file = Path('Cache') / f'simulator_{date_str}.parquet'
    
    if not cache_file.exists():
        st.error(f"❌ 데이터 파일이 없습니다: {cache_file}")
        return pd.DataFrame()
    
    return pd.read_parquet(cache_file)


def create_comparison_heatmap(df: pd.DataFrame, time_idx: int, use_transformer: bool = False):
    """현재 vs 30분 후 비교 히트맵 생성
    
    Args:
        df: 데이터프레임
        time_idx: 시간 인덱스
        use_transformer: True면 Transformer 예측 사용, False면 통계 방법 사용
    """
    # 해당 시간대 데이터 필터링
    df_time = df[df['time_index'] == time_idx].copy()
    
    if df_time.empty:
        st.warning(f"⚠️ {format_time_index(time_idx)} 시점에 데이터가 없습니다.")
        return
    
    # 예측 컬럼 선택
    pred_col = 'transformer_pred_30min' if use_transformer else 'predicted_risk_30min'
    
    # Transformer 예측이 없는 경우 처리
    if use_transformer and pred_col not in df_time.columns:
        st.error("❌ Transformer 예측 데이터가 없습니다. 먼저 `python src/precompute_transformer_predictions.py`를 실행하세요.")
        return
    
    # NaN 제거
    if use_transformer:
        df_time = df_time[df_time[pred_col].notna()].copy()
    
    # spot_name으로 정렬 (관련 Zone이 함께 보이도록)
    df_time = df_time.sort_values(['spot_name', 'current_risk'], ascending=[True, False])
    
    # 상위 30개만 표시
    df_time = df_time.head(30)
    
    # 색상 매핑 함수
    def get_color(risk):
        if risk >= 0.5:
            return '#d32f2f'  # Critical
        elif risk >= 0.3:
            return '#f57c00'  # Caution
        else:
            return '#388e3c'  # Safe
    
    current_colors = df_time['current_risk'].apply(get_color)
    predicted_colors = df_time[pred_col].apply(get_color)
    
    # 변화량 계산
    risk_changes = df_time[pred_col] - df_time['current_risk']
    change_icons = risk_changes.apply(lambda x: 
        '📈' if x > 0.05 else '📉' if x < -0.05 else '➡️'
    )
    
    # 예측 방법 표시
    method_name = "🤖 Transformer" if use_transformer else "📊 통계 방법"
    
    # 2개 컬럼 레이아웃
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🔴 현재 위험도 ({format_time_index(time_idx)})")
        fig1 = go.Figure(data=[
            go.Bar(
                x=df_time['spot_name'],
                y=df_time['current_risk'],
                marker=dict(
                    color=current_colors,
                    line=dict(color='white', width=0.5)
                ),
                text=[f"{v:.3f}" for v in df_time['current_risk']],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>' +
                    '현재 위험도: %{y:.3f}<br>' +
                    '<extra></extra>'
                )
            )
        ])
        
        fig1.update_layout(
            xaxis=dict(
                title="Zone",
                tickangle=-45,
                tickfont=dict(size=9)
            ),
            yaxis=dict(
                title="Risk Score",
                range=[0, 0.7]
            ),
            height=450,
            margin=dict(b=120, l=50, r=20, t=30),
            plot_bgcolor='#f8f9fa',
            hovermode='x'
        )
        
        fig1.add_hline(y=0.5, line_dash="dash", line_color="red", line_width=1)
        fig1.add_hline(y=0.3, line_dash="dash", line_color="orange", line_width=1)
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        future_time_idx = min(time_idx + 6, 288)
        st.markdown(f"### 🔮 30분 후 예측 ({format_time_index(future_time_idx)}) - {method_name}")
        fig2 = go.Figure(data=[
            go.Bar(
                x=df_time['spot_name'],
                y=df_time[pred_col],
                marker=dict(
                    color=predicted_colors,
                    line=dict(color='white', width=0.5)
                ),
                text=[f"{v:.3f} {icon}" for v, icon in zip(df_time[pred_col], change_icons)],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>' +
                    '예측 위험도: %{y:.3f}<br>' +
                    '<extra></extra>'
                )
            )
        ])
        
        fig2.update_layout(
            xaxis=dict(
                title="Zone",
                tickangle=-45,
                tickfont=dict(size=9)
            ),
            yaxis=dict(
                title="Risk Score",
                range=[0, 0.7]
            ),
            height=450,
            margin=dict(b=120, l=20, r=50, t=30),
            plot_bgcolor='#f8f9fa',
            hovermode='x'
        )
        
        fig2.add_hline(y=0.5, line_dash="dash", line_color="red", line_width=1)
        fig2.add_hline(y=0.3, line_dash="dash", line_color="orange", line_width=1)
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # 변화 요약 테이블
    st.markdown("### 📊 위험도 변화 상세")
    
    # 변화가 큰 순서로 정렬 (절댓값 기준)
    df_time_sorted = df_time.copy()
    df_time_sorted['abs_change'] = df_time_sorted['risk_change'].abs()
    df_time_sorted = df_time_sorted.sort_values('abs_change', ascending=False).head(20)
    
    # 표 데이터 생성
    change_data = []
    for _, row in df_time_sorted.iterrows():
        change_pct = (row['risk_change'] / row['current_risk'] * 100) if row['current_risk'] > 0 else 0
        
        # 현재 위험도 산출 데이터
        current_reasons = []
        current_reasons.append(f"작업자 {row['current_worker']:.0f}명")
        if row['current_equipment'] > 0:
            current_reasons.append(f"장비 {row['current_equipment']:.0f}개")
        current_reasons.append(f"밀도 {row['density_risk']:.3f}")
        if row['z_score_worker'] > 2.0:
            current_reasons.append(f"패턴이상(Z={row['z_score_worker']:.1f})")
        
        # 30분 후 예측 원인
        future_reasons = []
        worker_change = row['predicted_worker_30min'] - row['current_worker']
        if abs(worker_change) >= 1:
            future_reasons.append(f"작업자 {worker_change:+.0f}명 예상")
        else:
            future_reasons.append(f"작업자 유지({row['predicted_worker_30min']:.0f}명)")
        
        density_change = row['future_density_risk'] - row['density_risk']
        if abs(density_change) > 0.02:
            future_reasons.append(f"밀도 {density_change:+.3f}")
        
        pattern_change = row['future_pattern_deviation_risk'] - row['pattern_deviation_risk']
        if abs(pattern_change) > 0.02:
            future_reasons.append(f"패턴편차 {pattern_change:+.3f}")
        
        time_factor_change = row['future_time_of_day_factor'] - row['time_of_day_factor']
        if abs(time_factor_change) > 0.05:
            future_reasons.append(f"시간대 ×{row['future_time_of_day_factor']:.2f}")
        
        current_reason_text = ", ".join(current_reasons)
        future_reason_text = ", ".join(future_reasons) if future_reasons else "변화 미미"
        
        change_data.append({
            'Zone': row['spot_name'],
            '현재위험': row['current_risk'],
            '예측위험': row['predicted_risk_30min'],
            '변화': row['risk_change'],
            '변화율': change_pct,
            '현재 위험도 산출': current_reason_text,
            '30분 예측 원인': future_reason_text
        })
    
    change_df = pd.DataFrame(change_data)
    
    # 스타일 적용 함수 - 30분 후 위험도 기준으로 색상 결정
    def style_risk_row(row):
        predicted_risk = row['예측위험']
        
        # 위험도에 따른 배경색 (세련된 파스텔톤)
        if predicted_risk >= 0.5:
            bg_color = '#ef5350'  # 빨강 (Material Red 400)
            text_color = '#ffffff'
        elif predicted_risk >= 0.3:
            bg_color = '#ff9800'  # 주황 (Material Orange 500)
            text_color = '#ffffff'
        else:
            bg_color = '#66bb6a'  # 초록 (Material Green 400)
            text_color = '#ffffff'
        
        return [f'background-color: {bg_color}; color: {text_color}; font-weight: 500' for _ in row]
    
    styled_df = change_df.style.apply(style_risk_row, axis=1).format({
        '현재위험': '{:.3f}',
        '예측위험': '{:.3f}',
        '변화': '{:+.3f}',
        '변화율': '{:+.1f}%'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)


def display_risk_breakdown(df: pd.DataFrame, time_idx: int):
    """위험 요인 분석 표시"""
    df_time = df[df['time_index'] == time_idx].copy()
    
    if df_time.empty:
        return
    
    # 위험도 상위 10개 Zone
    top_risks = df_time.nlargest(10, 'current_risk')
    
    st.markdown("### 📊 위험 요인 분석 (Top 10)")
    
    for idx, row in top_risks.iterrows():
        with st.expander(f"🔴 {row['spot_name']} - 위험도: {row['current_risk']:.3f} ({row['risk_level']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**현재 상태**")
                st.write(f"- 작업자: {row['current_worker']}명")
                st.write(f"- 장비: {row['current_equipment']}개")
                st.write(f"- 구역 유형: {row['zone_type']}")
                st.write(f"- 면적: {row['area']:.0f}㎡")
                
                st.markdown("**기준 패턴 대비**")
                st.write(f"- 평균 작업자: {row['avg_worker']:.1f}명")
                st.write(f"- 편차 (Z-score): {row['z_score_worker']:.2f}")
                
                if abs(row['z_score_worker']) > 2.0:
                    st.warning(f"⚠️ 패턴 이상 감지! (Z > 2.0)")
            
            with col2:
                st.markdown("**위험 구성 요소 (현재)**")
                
                # 5가지 요소 기여도
                total_base = row['base_risk_before_scaling']
                st.write(f"1️⃣ 밀도: {row['density_risk']:.3f} ({row['density_risk']/total_base*100:.1f}%)")
                st.write(f"2️⃣ 구역 위험도: {row['zone_base_risk']:.3f} ({row['zone_base_risk']/total_base*100:.1f}%)")
                st.write(f"3️⃣ 작업자-장비: {row['coexistence_risk']:.3f} ({row['coexistence_risk']/total_base*100:.1f}%)")
                st.write(f"4️⃣ 패턴 편차: {row['pattern_deviation_risk']:.3f} ({row['pattern_deviation_risk']/total_base*100:.1f}%)")
                st.write(f"5️⃣ 밀폐공간: {row['confined_adjustment']:.3f} ({row['confined_adjustment']/total_base*100:.1f}%)")
                
                st.markdown("**조정 계수 (현재)**")
                st.write(f"- 체류시간: ×{row['dwell_time_factor']:.2f}")
                st.write(f"- 시간대: ×{row['time_of_day_factor']:.2f}")
            
            # 30분 후 예측 분석 (전체 너비)
            st.markdown("---")
            st.markdown("### 🔮 30분 후 예측 분석")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**예측 상태**")
                worker_change = row['predicted_worker_30min'] - row['current_worker']
                st.write(f"- 예상 작업자: {row['predicted_worker_30min']:.0f}명 ({worker_change:+.0f})")
                st.write(f"- 예상 장비: {row['predicted_equipment_30min']:.0f}개")
                st.write(f"- 예상 밀도: {row['future_density_risk']:.3f}")
                st.write(f"- 예상 패턴편차: {row['future_pattern_deviation_risk']:.3f} (Z={row['future_z_score_worker']:.2f})")
                
                risk_change = row['risk_change']
                if risk_change > 0.05:
                    st.error(f"📈 **위험도 증가**: {row['current_risk']:.3f} → {row['predicted_risk_30min']:.3f} (+{risk_change:.3f})")
                elif risk_change < -0.05:
                    st.success(f"📉 **위험도 감소**: {row['current_risk']:.3f} → {row['predicted_risk_30min']:.3f} ({risk_change:.3f})")
                else:
                    st.info(f"➡️ **위험도 유지**: {row['predicted_risk_30min']:.3f} ({risk_change:+.3f})")
            
            with col4:
                st.markdown("**변화 원인 분석**")
                
                # 각 요소별 변화량
                density_change = row['future_density_risk'] - row['density_risk']
                pattern_change = row['future_pattern_deviation_risk'] - row['pattern_deviation_risk']
                coexist_change = row['future_coexistence_risk'] - row['coexistence_risk']
                time_factor_change = row['future_time_of_day_factor'] - row['time_of_day_factor']
                
                changes = []
                if abs(worker_change) >= 1:
                    changes.append((f"작업자 변화 ({worker_change:+.0f}명)", abs(worker_change) * 0.05))
                if abs(density_change) > 0.01:
                    changes.append((f"밀도 변화 ({density_change:+.3f})", abs(density_change)))
                if abs(pattern_change) > 0.01:
                    changes.append((f"패턴편차 변화 ({pattern_change:+.3f})", abs(pattern_change)))
                if abs(time_factor_change) > 0.03:
                    changes.append((f"시간대 계수 변화 (×{row['future_time_of_day_factor']:.2f})", abs(time_factor_change) * 0.5))
                
                if changes:
                    # 영향도 순으로 정렬
                    changes.sort(key=lambda x: x[1], reverse=True)
                    st.write("주요 변화 요인:")
                    for i, (desc, _) in enumerate(changes, 1):
                        st.write(f"{i}. {desc}")
                else:
                    st.write("- 변화 미미 (패턴 유지)")
                    st.info(f"➡️ 유지 예상: {row['predicted_risk_30min']:.3f} ({risk_change:+.3f})")


def display_statistics(df: pd.DataFrame, time_idx: int):
    """통계 요약 표시 (1줄 압축)"""
    df_time = df[df['time_index'] == time_idx]
    
    if df_time.empty:
        return
    
    # 통계 계산
    total_zones = len(df_time)
    avg_risk = df_time['current_risk'].mean()
    caution_count = len(df_time[df_time['risk_level'] == 'Caution'])
    caution_pct = caution_count / total_zones * 100
    critical_count = len(df_time[df_time['risk_level'] == 'Critical'])
    increasing = len(df_time[df_time['risk_change'] > 0.05])
    stable = len(df_time[abs(df_time['risk_change']) <= 0.05])
    decreasing = len(df_time[df_time['risk_change'] < -0.05])
    
    # 한 줄 표시
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <span style="margin-right: 20px;">📊 <b>총 Zone:</b> {total_zones}개</span>
        <span style="margin-right: 20px;">📈 <b>평균 위험도:</b> {avg_risk:.3f}</span>
        <span style="margin-right: 20px;">⚠️ <b>주의:</b> {caution_count}개 ({caution_pct:.1f}%)</span>
        <span style="margin-right: 20px;">🚨 <b>위험:</b> {critical_count}개</span>
        <span style="margin-right: 15px;">|</span>
        <span style="margin-right: 20px;">📈 <b>증가예상:</b> {increasing}개</span>
        <span style="margin-right: 20px;">➡️ <b>유지:</b> {stable}개</span>
        <span><b>📉 감소예상:</b> {decreasing}개</span>
    </div>
    """, unsafe_allow_html=True)


def render_simulator_tab():
    """시뮬레이터 탭 렌더링"""
    st.title("🎮 DeepCon Simulator")
    
    # 날짜 선택 및 예측 방법 선택 (콤팩트하게)
    col_date, col_method, col_time = st.columns([1, 1, 2])
    
    with col_date:
        selected_date = st.selectbox(
            "📅 날짜",
            options=list(AVAILABLE_DATES.keys()),
            format_func=lambda x: AVAILABLE_DATES[x],
            label_visibility="collapsed"
        )
        st.caption(f"📅 {AVAILABLE_DATES[selected_date]}")
    
    with col_method:
        use_transformer = st.checkbox(
            "🤖 Transformer 예측 사용",
            value=True,
            help="체크: Transformer AI 모델 / 해제: 통계 기반 방법"
        )
        if use_transformer:
            st.caption("🤖 AI 모델 (MAE: 0.032)")
        else:
            st.caption("📊 통계 방법 (MAE: 0.073)")
    
    # 데이터 로드
    df = load_simulator_data(selected_date)
    
    if df.empty:
        st.stop()
    
    # 시간 슬라이더
    with col_time:
        # 분 단위 슬라이더 (0~1435분, 5분 단위)
        total_minutes = st.slider(
            "⏰ 시간 선택 (5분 단위)",
            min_value=0,
            max_value=1435,  # 23:55
            value=540,  # 기본값 09:00
            step=5,
            format="",
            label_visibility="collapsed"
        )
        
        selected_hour = total_minutes // 60
        selected_minute = total_minutes % 60
    
    current_time_idx = get_time_index_from_time(selected_hour, selected_minute)
    current_time_str = format_time_index(current_time_idx)
    
    # 30분 후 시간
    future_time_idx = min(current_time_idx + 6, 288)
    future_time_str = format_time_index(future_time_idx)
    
    # 시간 정보와 통계를 한 줄로
    method_emoji = "🤖" if use_transformer else "📊"
    st.markdown(f"<h4 style='margin-bottom: 5px;'>{method_emoji} 🕐 {current_time_str} → 🔮 {future_time_str}</h4>", unsafe_allow_html=True)
    
    # 통계 표시
    display_statistics(df, current_time_idx)
    
    # 현재 vs 예측 비교 (Transformer 옵션 전달)
    create_comparison_heatmap(df, current_time_idx, use_transformer=use_transformer)
    
    # 위험 요인 분석
    st.markdown("---")
    display_risk_breakdown(df, current_time_idx)
    
    # 도움말
    st.markdown("---")
    with st.expander("ℹ️ 위험도 계산 방법"):
        st.markdown("""
        ### 위험도 계산 체계
        
        **5가지 핵심 요소 (가중평균)**
        1. **밀도 위험도** (25%): 구역 면적 대비 작업자 밀집도
        2. **구역 기본 위험도** (20%): 밀폐공간(0.4) > 작업장(0.3) > 휴게실(0.05)
        3. **공존 위험도** (15%): 작업자-장비 동시 존재 시 증가
        4. **패턴 편차 위험도** (35%): 평소 대비 이상치 감지 (Z-score 기반)
        5. **밀폐공간 조정** (5%): 밀폐공간 특성 반영
        
        **조정 계수**
        - **체류시간**: 10분 미만(1.0배) → 60분 이상(1.3배)
        - **시간대**: 새벽(1.25배), 야간(1.15배), 이른 아침(1.08배)
        
        **최종 위험도** = 기본 위험도 × 체류시간 계수 × 시간대 계수
        
        **위험 등급**
        - 🟢 Safe: 0.0 ~ 0.3
        - 🟡 Caution: 0.3 ~ 0.5
        - 🔴 Critical: 0.5 이상
        
        **30분 후 예측**
        - 현재 편차 비율을 유지한다고 가정
        - 주중/주말 패턴 기반 예측
        - 장비는 현재 상태 유지 가정
        """)
