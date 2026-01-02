"""
Sidebar Component
=================
Sidebar UI with date selection, data summary, and system info
"""
import streamlit as st
import os
from pathlib import Path
from src.cached_data_loader import CachedDataLoader, find_available_datasets as _find_available_datasets

# Import forecast engine if available
try:
    from src.forecast_engine import ForecastEngine
except ImportError:
    ForecastEngine = None


@st.cache_data(ttl=600)
def find_available_datasets_cached(cache_folder):
    """사용 가능한 데이터셋 찾기 (캐시 적용)"""
    return _find_available_datasets(cache_folder)


def render_sidebar():
    """세련된 사이드바 렌더링"""
    with st.sidebar:
        # 로고 영역
        st.markdown("""
        <div class="sidebar-logo">
            <h2>🏭 SK Hynix Y1</h2>
            <p>Industrial Resources Flow Management</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 시스템 정보
        st.markdown("""
        <div style="text-align:center; padding:0.5rem; opacity:0.8;">
            <small>constructed by <strong>SK Ecoplant</strong></small><br>
            <small>system by <strong>TJLABS</strong></small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 캐시 폴더 경로 (DeepCon 루트 디렉토리의 Cache 폴더)
        # __file__ = .../DeepCon/src/ui/sidebar.py
        # 3단계 상위: DeepCon/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_folder = os.path.join(project_root, "Cache")
        
        # 사용 가능한 데이터셋 찾기 (캐시 적용)
        datasets = find_available_datasets_cached(cache_folder)
        
        if not datasets:
            st.error("⚠️ 데이터가 없습니다")
            st.info("precompute_full.py를 실행하세요")
            return None
        
        # 날짜 선택
        st.markdown("### 📅 Date Selection")
        
        date_options = []
        for ds in datasets:
            date_str = ds['date']
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            date_options.append((date_str, formatted_date))
        
        selected_idx = st.selectbox(
            "분석 날짜",
            range(len(date_options)),
            format_func=lambda x: date_options[x][1],
            label_visibility="collapsed",
            key="sidebar_date_select"
        )
        
        selected_date = date_options[selected_idx][0]
        
        # 데이터 로더 세션 캐싱 (Phase 4: Disk I/O 제거)
        if "cached_loader" not in st.session_state or st.session_state.cached_loader.date_str != selected_date:
            with st.spinner("🚀 세션 데이터 로더 초기화 중..."):
                loader = CachedDataLoader(cache_folder, selected_date)
                try:
                    loader.preload_all()
                except Exception:
                    pass
                st.session_state.cached_loader = loader
        else:
            loader = st.session_state.cached_loader

        if not loader.is_valid():
            st.error("❌ 데이터 로드 실패")
            return None
        
        # 데이터 요약 정보
        summary = loader.get_summary()
        if summary:
            st.markdown("### 📊 Data Summary")
            
            formatted_date = f"{selected_date[:4]}-{selected_date[4:6]}-{selected_date[6:]}"
            
            st.markdown(f"""
            <div class="sidebar-info">
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Date</span>
                    <span class="sidebar-info-value">{formatted_date}</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Location</span>
                    <span class="sidebar-info-value">Y1 at Yongin</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # T31 정보
            t31_info = summary.get('t31', {})
            t31_devices = t31_info.get('total_devices', 0)
            t31_rate = t31_info.get('avg_operation_rate', 0)
            
            st.markdown(f"""
            <div class="sidebar-info">
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">🔧 T31 on TL</span>
                    <span class="sidebar-info-value">{t31_devices} units</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Avg Operation</span>
                    <span class="sidebar-info-value">{t31_rate:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # T41 정보
            t41_info = summary.get('t41', {})
            t41_workers = t41_info.get('total_workers', 0)
            t41_avg_active = t41_info.get('avg_active', 0)
            t41_activity = (t41_avg_active / t41_workers * 100) if t41_workers > 0 else 0
            
            st.markdown(f"""
            <div class="sidebar-info">
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">👷 T41 on Worker</span>
                    <span class="sidebar-info-value">{t41_workers:,} workers</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Activity Rate</span>
                    <span class="sidebar-info-value">{t41_activity:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Mobile 정보
            mobile_info = summary.get('mobile', {})
            mobile_devices = mobile_info.get('total_devices', 0)
            android = mobile_info.get('android_devices', 0)
            iphone = mobile_info.get('iphone_devices', 0)
            
            st.markdown(f"""
            <div class="sidebar-info">
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">📱 MobilePhone</span>
                    <span class="sidebar-info-value">{mobile_devices:,} devices</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Android / iPhone</span>
                    <span class="sidebar-info-value">{android:,} / {iphone:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:1rem; opacity:0.5;">
            <small>IRFM Dashboard v2.0</small><br>
            <small>© 2025 TJLABS</small>
        </div>
        """, unsafe_allow_html=True)

        return loader
