"""
UI Styles and Theme Configuration
==================================
Centralized theme colors and CSS styles for the IRFM dashboard
"""

# Theme Color Palette
THEME = {
    'primary': '#0066CC',
    'secondary': '#00A3E0',
    'accent': '#FF6B35',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'dark': '#1E293B',
    'light': '#F8FAFC',
    'gray': '#64748B',
    'text_primary': '#1E293B',
    'text_secondary': '#475569',
    'text_muted': '#94A3B8',
    't31': '#F97316',
    't41_active': '#10B981',
    't41_inactive': '#CBD5E1',
    'mobile_android': '#22C55E',
    'mobile_iphone': '#3B82F6',
    'bg_card': '#FFFFFF',
    'bg_page': '#F8FAFC',
    'border': '#E2E8F0',
}

# Journey Heatmap Colors
JOURNEY_COLORS = {
    # 0: 미수신 (검정), 1: 비활성 (회색)
    'no_signal': 0,
    'inactive': 1,
    # 실외 (검은 회색) - 2
    'Outdoor': 2,
    # WWT 계열 (녹색 계통) - 3~5
    'WWT-B1F': 3,   # (1, 2)
    'WWT-1F': 4,    # (1, 3)
    'WWT-2F': 5,    # (1, 10)
    # FAB 계열 (주황 계통) - 6
    'FAB-1F': 6,    # (2, 4)
    # CUB 계열 (파랑 계통) - 7~8
    'CUB-1F': 7,    # (3, 5)
    'CUB-B1F': 8,   # (3, 6)
    # WTP (노랑) - 9
    'WTP-1F': 9,    # (4, 8)
}

JOURNEY_COLOR_PALETTE = [
    '#1a1a1a',  # 0: 미수신 - 검정
    '#6b7280',  # 1: 비활성 - 회색
    '#374151',  # 2: 실외 - 진한 회색
    '#86efac',  # 3: WWT-B1F - 연두 (연한)
    '#22c55e',  # 4: WWT-1F - 초록 (중간)
    '#15803d',  # 5: WWT-2F - 진초록 (진한)
    '#f97316',  # 6: FAB-1F - 주황
    '#7dd3fc',  # 7: CUB-1F - 연파랑 (연한)
    '#0284c7',  # 8: CUB-B1F - 파랑 (진한)
    '#fde047',  # 9: WTP-1F - 노랑
]

BUILDING_FLOOR_MAP = {
    (0, 0): 'Outdoor',
    (1, 2): 'WWT-B1F',
    (1, 3): 'WWT-1F',
    (1, 10): 'WWT-2F',
    (2, 4): 'FAB-1F',
    (3, 5): 'CUB-1F',
    (3, 6): 'CUB-B1F',
    (4, 8): 'WTP-1F',
}


def get_custom_css() -> str:
    """Generate custom CSS with theme colors"""
    return f"""
<style>
    /* ========== 전역 스타일 리셋 ========== */
    .stApp {{
        background: {THEME['bg_page']} !important;
    }}
    
    /* 모든 텍스트 요소에 진한 색상 강제 적용 */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMarkdownContainer"], 
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* 진한 배경용 흰색 텍스트 예외 */
    .dark-bg, .dark-bg *,
    .dark-bg p, .dark-bg span, .dark-bg div, .dark-bg h3, .dark-bg h4 {{
        color: white !important;
    }}
    .dark-bg .text-muted {{
        color: rgba(255,255,255,0.8) !important;
    }}
    .dark-bg .text-light {{
        color: rgba(255,255,255,0.6) !important;
    }}
    
    /* h1~h6 헤딩 */
    h1, h2, h3, h4, h5, h6,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* st.metric 스타일 */
    [data-testid="stMetricValue"] {{
        color: {THEME['primary']} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {THEME['text_secondary']} !important;
    }}
    
    /* 버튼 */
    .stButton button, .stDownloadButton button {{
        color: white !important;
        background-color: {THEME['primary']} !important;
    }}
    
    /* ========== 메인 헤더 ========== */
    .main-header {{
        background: linear-gradient(135deg, {THEME['primary']} 0%, #0284C7 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,102,204,0.2);
    }}
    .main-header h1 {{
        color: white !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }}
    .main-header p {{
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }}
    
    /* ========== 메트릭 카드 ========== */
    .metric-card {{
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid {THEME['border']};
        transition: all 0.2s ease;
    }}
    .metric-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {THEME['primary']};
        line-height: 1.2;
    }}
    .metric-value.orange {{
        color: {THEME['t31']};
    }}
    .metric-value.green {{
        color: {THEME['t41_active']};
    }}
    .metric-label {{
        color: {THEME['text_secondary']};
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }}
    
    /* ========== 사이드바 스타일 ========== */
    [data-testid="stSidebar"] {{
        background: {THEME['dark']} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: {THEME['dark']} !important;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {{
        color: white !important;
    }}
    
    /* 사이드바 내 모든 드롭다운/셀렉트박스 - 강제 검정 텍스트 */
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {{
        background-color: white !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"],
    [data-testid="stSidebar"] .stSelectbox [role="combobox"] {{
        color: {THEME['text_primary']} !important;
        background-color: white !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {{
        color: {THEME['text_primary']} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] svg,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {{
        fill: {THEME['text_primary']} !important;
    }}
    /* 드롭다운 열렸을 때 옵션 리스트도 검정 텍스트 */
    [data-testid="stSidebar"] [role="listbox"] {{
        background-color: white !important;
    }}
    [data-testid="stSidebar"] [role="option"],
    [data-testid="stSidebar"] [role="option"] span {{
        background-color: white !important;
        color: {THEME['text_primary']} !important;
    }}
    [data-testid="stSidebar"] [role="option"]:hover {{
        background-color: {THEME['bg_page']} !important;
        color: {THEME['text_primary']} !important;
    }}
    
    .sidebar-logo {{
        text-align: center;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }}
    .sidebar-logo h2 {{
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }}
    
    .sidebar-info {{
        background: rgba(255,255,255,0.08);
        padding: 0.875rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .sidebar-info-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .sidebar-info-row:last-child {{
        border-bottom: none;
    }}
    .sidebar-info-label {{
        color: rgba(255,255,255,0.5) !important;
        font-size: 0.75rem;
    }}
    .sidebar-info-value {{
        color: white !important;
        font-weight: 600;
        font-size: 0.8rem;
    }}
    
    /* ========== 탭 스타일 ========== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: white;
        padding: 0.375rem;
        border-radius: 10px;
        border: 1px solid {THEME['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.625rem 1.25rem;
        border-radius: 8px;
        font-weight: 500;
        color: {THEME['text_secondary']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {THEME['primary']} !important;
    }}
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {{
        color: white !important;
    }}
    
    /* ========== 드롭다운 메뉴 스타일 수정 ========== */
    /* Selectbox 드롭다운 배경을 흰색으로 */
    .stSelectbox [data-baseweb="select"] {{
        background-color: white !important;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        background-color: white !important;
    }}
    /* 드롭다운 옵션 리스트 */
    [data-baseweb="popover"] {{
        background-color: white !important;
    }}
    [role="listbox"] {{
        background-color: white !important;
    }}
    [role="option"] {{
        background-color: white !important;
        color: {THEME['text_primary']} !important;
    }}
    [role="option"]:hover {{
        background-color: {THEME['bg_page']} !important;
    }}
    /* 선택된 값 텍스트 */
    .stSelectbox [data-baseweb="select"] input {{
        color: {THEME['text_primary']} !important;
    }}
    .stSelectbox [data-baseweb="select"] span {{
        color: {THEME['text_primary']} !important;
    }}
    /* 드롭다운 화살표 아이콘 */
    .stSelectbox [data-baseweb="select"] svg {{
        fill: {THEME['text_primary']} !important;
    }}
    
    /* 숨기기 */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
</style>
"""


def apply_theme(st_module):
    """Apply theme CSS to Streamlit"""
    st_module.markdown(get_custom_css(), unsafe_allow_html=True)
