"""
Overview Tab Component
======================
Main overview dashboard showing T31, T41, and Mobile statistics
"""
import streamlit as st
import plotly.graph_objects as go
from src.cached_data_loader import CachedDataLoader
from src.components.styles import THEME
from src.ui.charts import get_chart_layout
from src.utils.formatters import bin_index_to_time_str


def render_overview(loader: CachedDataLoader):
    """Overview 탭 - 전체 현황"""
    summary = loader.get_summary()
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📊 Overview</h1>
        <p>SK Hynix Y1 Cluster 전체 현황 대시보드</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== T31 섹션 =====
    render_t31_overview(loader, summary)
    
    st.markdown("---")
    
    # ===== T41 섹션 =====
    render_t41_overview(loader, summary)
    
    st.markdown("---")
    
    # ===== MobilePhone 섹션 =====
    render_mobile_overview(loader, summary)


def render_t31_overview(loader: CachedDataLoader, summary: dict):
    """T31 Equipment Overview Section"""
    st.markdown("### 🔧 T-Ward Type31 (Equipment)")
    
    # === 데이터 소스: t31_time_series 캐시 사용 (경량 캐시) ===
    t31_time_series = loader.load_t31_time_series()
    
    if t31_time_series is not None and not t31_time_series.empty:
        t31_devices = t31_time_series['total_devices'].iloc[0]
        
        # 일과시간 (07:00~19:00, time_index 85~228) 평균 가동률 계산
        work_hours_ts = t31_time_series[
            (t31_time_series['time_index'] >= 85) & 
            (t31_time_series['time_index'] <= 228)
        ]
        
        if not work_hours_ts.empty and t31_devices > 0:
            work_hours_ts = work_hours_ts.copy()
            work_hours_ts['rate'] = work_hours_ts['active_devices'] / t31_devices * 100
            t31_rate = work_hours_ts['rate'].mean()
        else:
            t31_rate = 0
    else:
        t31_devices = 0
        t31_rate = 0
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{t31_devices}</div>
            <div class="metric-label">Monitoring Equipment</div>
            <div class="metric-delta positive">하루 1회 이상 감지된 Unique MAC</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{t31_rate:.1f}%</div>
            <div class="metric-label">Work Hour Rate</div>
            <div class="metric-delta positive">일과시간(07~19시) 5분단위 평균 가동률</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # T31 5분 단위 가동 장비 차트
        if t31_time_series is not None and not t31_time_series.empty:
            fig = go.Figure()
            
            # Active 장비 (가동 중 - 신호 있음)
            fig.add_trace(go.Scatter(
                x=t31_time_series['time_label'],
                y=t31_time_series['active_devices'],
                mode='lines',
                fill='tozeroy',
                name='가동 (Active)',
                line=dict(color=THEME['t31'], width=1),
                fillcolor='rgba(249, 115, 22, 0.6)',
                hovertemplate='<b>%{x}</b><br>가동: %{y}대<extra></extra>'
            ))
            
            # Inactive 영역
            fig.add_trace(go.Scatter(
                x=t31_time_series['time_label'],
                y=t31_time_series['total_devices'],
                mode='lines',
                fill='tonexty',
                name='비가동 (Inactive)',
                line=dict(color=THEME['t41_inactive'], width=1),
                fillcolor='rgba(203, 213, 225, 0.4)',
                hovertemplate='<b>%{x}</b><br>전체: %{y}대<extra></extra>'
            ))
            
            layout = get_chart_layout('5분 단위 장비 현황 (Active/Inactive)', height=200, show_legend=True)
            layout['xaxis'] = dict(
                tickmode='array',
                tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)],
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                tickfont=dict(color=THEME['text_secondary']),
                title=dict(text='Time', font=dict(color=THEME['text_secondary']))
            )
            layout['yaxis'] = dict(
                tickfont=dict(color=THEME['text_secondary']),
                title=dict(text='Equipment Count', font=dict(color=THEME['text_secondary']))
            )
            layout['legend'] = dict(
                orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                font=dict(color=THEME['text_primary'])
            )
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)


def render_t41_overview(loader: CachedDataLoader, summary: dict):
    """T41 Workers Overview Section"""
    st.markdown("### 👷 T-Ward Type41 (Workers)")
    
    t41_info = summary.get('t41', {}) if summary else {}
    t41_workers = t41_info.get('total_workers', 0)
    max_active = t41_info.get('max_active', 0)
    avg_active = int(t41_info.get('avg_active', 0))
    avg_dwell = t41_info.get('avg_dwell_minutes', 0)
    t41_activity = (avg_active / t41_workers * 100) if t41_workers > 0 else 0
    
    t41_time_series = loader.load_t41_time_series()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{t41_workers:,}</div>
            <div class="metric-label">Total Workers</div>
            <div class="metric-delta positive">하루 1회 이상 감지</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value green">{max_active:,}</div>
            <div class="metric-label">Max Active (최대)</div>
            <div class="metric-delta positive">동시 활성 최대</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_active:,}</div>
            <div class="metric-label">Avg Active (평균)</div>
            <div class="metric-delta positive">{t41_activity:.1f}% 활성률</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_dwell:.0f}</div>
            <div class="metric-label">Avg Dwell (min)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # T41 시간별 Active/Inactive 차트
    if t41_time_series is not None and not t41_time_series.empty:
        fig = go.Figure()
        
        if 'active_workers' in t41_time_series.columns and 'inactive_workers' in t41_time_series.columns:
            t41_ts = t41_time_series.copy()
            t41_ts['total_workers'] = t41_ts['active_workers'] + t41_ts['inactive_workers']
            customdata = t41_ts[['active_workers', 'inactive_workers', 'total_workers']].values
            
            # Active 영역
            fig.add_trace(go.Scatter(
                x=t41_ts['time_label'],
                y=t41_ts['active_workers'],
                fill='tozeroy',
                fillcolor=f"rgba(0, 200, 83, 0.6)",
                line=dict(color=THEME['t41_active'], width=2),
                name='Active',
                customdata=customdata,
                hovertemplate='<b>%{x}</b><br>Active: %{customdata[0]:,}명<br>Inactive: %{customdata[1]:,}명<br>Total: %{customdata[2]:,}명<extra></extra>'
            ))
            
            # Inactive 영역
            fig.add_trace(go.Scatter(
                x=t41_ts['time_label'],
                y=t41_ts['total_workers'],
                fill='tonexty',
                fillcolor=f"rgba(148, 163, 184, 0.5)",
                line=dict(color=THEME['t41_inactive'], width=2),
                name='Inactive (영역)',
                hoverinfo='skip'
            ))
        elif 'active_workers' in t41_time_series.columns:
            fig.add_trace(go.Scatter(
                x=t41_time_series['time_label'],
                y=t41_time_series['active_workers'],
                fill='tozeroy',
                fillcolor=f"rgba(0, 200, 83, 0.6)",
                line=dict(color=THEME['t41_active'], width=2),
                name='Active',
                hovertemplate='<b>%{x}</b><br>Active: %{y}명<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text='시간별 작업자 현황 (5분 단위, 초록=활성, 회색=비활성)', font=dict(size=14, color=THEME['text_primary'])),
            xaxis_title='Time (5분 단위)',
            yaxis_title='해당 시점 인원 수 (Unique MAC)',
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=THEME['text_primary']),
            xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
            yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color=THEME['text_primary'])),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("※ 각 시점(5분)에 데이터가 있는 Unique MAC 수. 초록=활성(움직임 감지), 회색=비활성(정지 상태).")


def render_mobile_overview(loader: CachedDataLoader, summary: dict):
    """Mobile Phone Overview Section"""
    st.markdown("### 📱 MobilePhone")
    
    mobile_info = summary.get('mobile', {}) if summary else {}
    flow_devices = mobile_info.get('total_devices', 0)
    android = mobile_info.get('android_devices', 0)
    iphone = mobile_info.get('iphone_devices', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{flow_devices:,}</div>
            <div class="metric-label">Total Devices</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if flow_devices > 0:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value green">{android:,}</div>
                <div class="metric-label">Android</div>
                <div class="metric-delta positive">{android/flow_devices*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if flow_devices > 0:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {THEME['mobile_iphone']};">{iphone:,}</div>
                <div class="metric-label">iPhone</div>
                <div class="metric-delta info">{iphone/flow_devices*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # 파이 차트
        if flow_devices > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['Android', 'iPhone'],
                values=[android, iphone],
                marker_colors=[THEME['mobile_android'], THEME['mobile_iphone']],
                hole=0.6,
                textinfo='percent',
                textfont_size=11,
                textfont_color='white',
                hovertemplate='%{label}: %{value:,}<extra></extra>'
            )])
            fig.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                annotations=[dict(text='비율', x=0.5, y=0.5, font_size=11, font_color=THEME['text_secondary'], showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 시간별 기기 수 차트
    flow_unit_time = loader.load_flow_unit_time_unique()
    if flow_unit_time is not None and not flow_unit_time.empty:
        if 'bin_index' in flow_unit_time.columns:
            flow_unit_time['time_label'] = flow_unit_time['bin_index'].apply(bin_index_to_time_str)
        elif 'time_label' not in flow_unit_time.columns:
            flow_unit_time['time_label'] = range(len(flow_unit_time))
        
        y_col = 'unique_devices' if 'unique_devices' in flow_unit_time.columns else flow_unit_time.columns[1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flow_unit_time['time_label'],
            y=flow_unit_time[y_col],
            fill='tozeroy',
            fillcolor='rgba(0, 163, 224, 0.3)',
            line=dict(color=THEME['secondary'], width=2),
            hovertemplate='<b>%{x}</b><br>Devices: %{y:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text='시간별 모바일 기기 수 (5분 단위)', font=dict(size=14)),
            xaxis_title='Time',
            yaxis_title='Device Count',
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=THEME['text_primary']),
            xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
            yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
