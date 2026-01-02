#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SK Hynix Y1 Cluster - IRFM Dashboard
=====================================
Industrial Resources Flow Management System

SK Ecoplant 구축 | TJLABS 시스템
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys
from pathlib import Path
import math
import time
from typing import Dict, List, Optional, Any

# Import CachedDataLoader and helper functions
from src.cached_data_loader import CachedDataLoader, find_available_datasets as _find_available_datasets
from src import config

# Import new configuration and utilities
from src.env_config import app_config
from src.logging_config import DeepConLogger, logger
from src.error_handlers import handle_errors, safe_execute, validate_dataframe
try:
    from src.forecast_engine import ForecastEngine
except Exception:
    try:
        from forecast_engine import ForecastEngine
    except Exception:
        ForecastEngine = None
from src.forecast_engine import ForecastEngine
import streamlit.components.v1 as components
import json as _json
import base64 as _b64


# ==================== Performance Optimization Helpers ====================
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_forecast_json(date_str: str) -> Optional[Dict]:
    """날짜별 예측 데이터 JSON 로드 (캐시 적용)"""
    forecast_dir = Path("Cache")
    f_path = forecast_dir / f"forecast_{date_str}.json"
    if not f_path.exists():
        return None
    try:
        with open(f_path, 'r', encoding='utf-8') as f:
            return _json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=600)
def find_available_datasets_cached(cache_folder):
    """사용 가능한 데이터셋 찾기 (캐시 적용)"""
    return _find_available_datasets(cache_folder)

@st.cache_data(ttl=3600)
def load_aggregated_forecasts(available_dates: List[str], max_days: int = 7) -> List[Dict]:
    """여러 날짜의 요약 데이터를 통합 로드 (캐시 적용)"""
    all_results = []
    for d_str in available_dates[:max_days]:
        data = load_forecast_json(d_str)
        if data and isinstance(data, dict) and "forecasts" in data:
            for fcast in data["forecasts"]:
                fcast["date"] = d_str
                # Only keep minimal data for aggregation to save memory
                all_results.append({
                    "date": d_str,
                    "zone_name": fcast.get("zone_name", ""),
                    "risk_score": fcast.get("risk_score", 0.0),
                    "reasoning": fcast.get("reasoning", "")
                })
    return all_results

def _sanitize_data(data):
    """
    NaN 값을 None으로 변환하여 JSON 직렬화 오류 방지
    - 리스트, numpy array 등 다양한 입력 처리
    """
    if isinstance(data, (list, tuple)):
        return [None if pd.isna(x) else x for x in data]
    elif isinstance(data, (pd.Series, np.ndarray)):
        # numpy array 등은 tolist 후 처리
        return [None if pd.isna(x) else x for x in data.tolist()]
    return data


def _deterministic_jitter(xs, ys, scale=0.6):
    """Apply a small deterministic jitter to coordinate lists.
    Returns tuple (new_xs, new_ys).
    """
    import math
    out_x = []
    out_y = []
    if xs is None or ys is None:
        return xs, ys
    for x, y in zip(xs, ys):
        try:
            kx = int(round(float(x) * 1000))
            ky = int(round(float(y) * 1000))
            seed = (kx & 0xFFFF) ^ ((ky & 0xFFFF) << 16)
            # simple LCG for deterministic pseudo-randomness
            rnd = (seed * 9301 + 49297) % 233280
            ang = (rnd / 233280.0) * 2 * math.pi
            r = (((seed * 7 + 13) % 100) / 100.0) * scale
            dx = r * math.cos(ang)
            dy = r * math.sin(ang)
            out_x.append(float(x) + dx)
            out_y.append(float(y) + dy)
        except Exception:
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def _clean_figure_for_json(fig: go.Figure):
    """Traverse a Plotly figure and replace non-JSON-friendly numeric values (NaN/inf)
    and numpy scalar types with Python-native types or None.
    Modifies the figure in-place.
    """
    import numpy as _np

    def _clean_val(v):
        try:
            if v is None:
                return None
            # numpy types
            if isinstance(v, (_np.floating, float)):
                if not _np.isfinite(v):
                    return None
                return float(v)
            if isinstance(v, (_np.integer, int)):
                return int(v)
            return v
        except Exception:
            return None

    def _clean_seq(seq):
        if seq is None:
            return seq
        out = []
        for item in seq:
            if isinstance(item, (list, tuple)):
                out.append(_clean_seq(item))
            else:
                out.append(_clean_val(item))
        return out

    # Clean top-level traces
    try:
        for tr in list(fig.data or []):
            for key in ('x', 'y', 'z', 'lat', 'lon'):
                try:
                    val = getattr(tr, key, None)
                    if val is None:
                        continue
                    cleaned = _clean_seq(val)
                    setattr(tr, key, cleaned)
                except Exception:
                    continue
    except Exception:
        pass


def _deep_sanitize(obj):
    """Recursively sanitize a nested structure (dict/list/primitive) converting
    numpy scalars to Python types and replacing non-finite numbers with None.
    """
    import numpy as _np

    if obj is None:
        return None
    # Primitives
    if isinstance(obj, (_np.floating, float)):
        return None if not _np.isfinite(obj) else float(obj)
    if isinstance(obj, (_np.integer, int)):
        return int(obj)
    if isinstance(obj, (_np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (str,)):
        return obj
    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [_deep_sanitize(v) for v in obj]
    # Dict-like
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # keys must be strings for JSON
            try:
                key = str(k)
            except Exception:
                key = repr(k)
            out[key] = _deep_sanitize(v)
        return out
    # Fallback: try to coerce numpy arrays
    try:
        import numpy as _np2
        if isinstance(obj, _np2.ndarray):
            return _deep_sanitize(obj.tolist())
    except Exception:
        pass
    # Unknown types: convert to string representation
    try:
        return str(obj)
    except Exception:
        return None

    # Clean frames
    try:
        for fr in list(fig.frames or []):
            # fr.data can be a list/tuple of traces
            for tr in list(fr.data or []):
                for key in ('x', 'y', 'z', 'lat', 'lon'):
                    try:
                        # traces inside frames may be dict-like or object
                        val = getattr(tr, key, None) if hasattr(tr, key) else (tr.get(key) if isinstance(tr, dict) else None)
                        if val is None:
                            continue
                        cleaned = _clean_seq(val)
                        if hasattr(tr, key):
                            setattr(tr, key, cleaned)
                        elif isinstance(tr, dict):
                            tr[key] = cleaned
                    except Exception:
                        continue
    except Exception:
        pass


def load_floor_map_options() -> tuple:
    """Floor map에서 사용 가능한 빌딩과 층 옵션들을 로드"""

    # Cache 루트의 공통 floor_maps 폴더 사용
    cache_root = Path(__file__).parent / 'Cache'
    cache_dir = cache_root / 'floor_maps'

    # Gather available cache pairs to filter irfm.csv rows (so dropdown only shows cached floors)
    available_pairs = set()
    if cache_dir.exists():
        for p in cache_dir.glob('*.json'):
            name = p.stem
            try:
                b_str, f_str = name.split('_')
                available_pairs.add((int(b_str), int(f_str)))
            except Exception:
                continue

    buildings = []
    floors_by_building = {}

    # Try to populate from irfm.csv (preferred source of building/floor names)
    data_folder = Path(__file__).parent / 'Datafile' / 'Yongin_Cluster_202512010'
    irfm_path = data_folder / 'irfm.csv'
    if irfm_path.exists():
        try:
            irfm_df = pd.read_csv(irfm_path)
            # sort by building_number then floor_number for predictable order
            irfm_df = irfm_df.sort_values(['building_number', 'floor_number'], na_position='last')
            for _, row in irfm_df.iterrows():
                try:
                    bno = int(row.get('building_number', 0))
                    fno = int(row.get('floor_number', 0))
                except Exception:
                    continue

                # If cache directory exists, only include pairs that have cache; otherwise include all
                if cache_dir.exists() and (bno, fno) not in available_pairs:
                    continue

                bname = row.get('building_name') if pd.notna(row.get('building_name')) else f'Building {bno}'
                fname = row.get('floor_name') if pd.notna(row.get('floor_name')) else f'Floor {fno}'

                if bname not in buildings:
                    buildings.append(bname)
                floors_by_building.setdefault(bname, []).append({
                    'name': fname,
                    'building_no': bno,
                    'floor_no': fno
                })

            # Sort floors within each building by floor_number when possible
            for b in floors_by_building:
                floors_by_building[b].sort(key=lambda x: (int(x['floor_no']) if str(x['floor_no']).isdigit() else 0, x['name']))
                
        except Exception as e:
            print(f"Error reading irfm.csv: {e}")
            return [], {}
            
    return buildings, floors_by_building


@st.cache_data(ttl=3600, show_spinner=False)
def load_floor_map_cache(building_no: int, floor_no: int) -> dict:
    """Floor map 캐시를 로드"""
    import json
    from pathlib import Path
    # Cache 루트의 공통 floor_maps 폴더 사용
    cache_root = Path(__file__).parent / 'Cache'
    cache_path = cache_root / 'floor_maps' / f'{building_no}_{floor_no}.json'

    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 이미 완성된 Plotly figure JSON이 캐시에 있으면 그대로 전달
                if 'figure_json' in data and data.get('figure_json'):
                    return data
                # 이전 방식의 shapes/annotations 캐시도 지원
                return data
            except Exception:
                return {'shapes': [], 'annotations': [], 'polygons': [], 'length_x': 100, 'length_y': 100, 'floor_name': 'Unknown'}
    return {'shapes': [], 'annotations': [], 'polygons': [], 'length_x': 100, 'length_y': 100, 'floor_name': 'Unknown'}


@st.cache_data(ttl=3600, show_spinner=False)
def load_spot_data_cached() -> tuple:
    """Spot 데이터를 Streamlit 캐시로 로드 (1회)"""
    data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    spot_path = data_folder / 'spot.csv'
    spot_pos_path = data_folder / 'spot_position.csv'
    
    if spot_path.exists() and spot_pos_path.exists():
        return pd.read_csv(spot_path), pd.read_csv(spot_pos_path)
    return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_outdoor_gateway_cached() -> pd.DataFrame:
    """실외 게이트웨이 데이터를 Streamlit 캐시로 로드 (1회)"""
    data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    gateway_path = data_folder / 'gateway.csv'
    
    if gateway_path.exists():
        gw_df = pd.read_csv(gateway_path)
        # 실외 (floor_no가 NaN) + 좌표가 있는 게이트웨이만
        outdoor_gw = gw_df[
            gw_df['floor_no'].isna() & 
            gw_df['location_x'].notna() & 
            gw_df['location_y'].notna()
        ][['gateway_no', 'name', 'location_x', 'location_y']].copy()
        return outdoor_gw
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_indoor_gateway_cached(building_no: int, floor_no: int) -> pd.DataFrame:
    """실내 게이트웨이 데이터를 Streamlit 캐시로 로드"""
    data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    gateway_path = data_folder / 'gateway.csv'
    
    if gateway_path.exists():
        gw_df = pd.read_csv(gateway_path)
        # 해당 층의 게이트웨이 (floor_no uses global ID from irfm.csv, so it is unique)
        indoor_gw = gw_df[
            (gw_df['floor_no'] == floor_no) &
            gw_df['location_x'].notna() & 
            gw_df['location_y'].notna()
        ][['gateway_no', 'name', 'location_x', 'location_y']].copy()
        return indoor_gw
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_flow_cache_cached(cache_folder: str, date_str: str, resolution: str = '5min') -> pd.DataFrame:
    """Flow cache를 Streamlit 캐시로 로드 (1회)

    Args:
        resolution: '5min' (default) or '1min' to select aggregation level
    """
    loader = CachedDataLoader(cache_folder, date_str)
    return loader.load_flow_cache(resolution)


@st.cache_data
def load_t41_location_cache(cache_path: str, date_str: str) -> dict:
    """(Deprecated) Load full location cache. Kept for compatibility."""
    import json
    from pathlib import Path
    
    cache_file = Path(cache_path) / date_str / 't41_location_cache.json'
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# @st.cache_data
@st.cache_data(ttl=3600, show_spinner=False)
def load_split_location_cache(cache_path: str, date_str: str, building_no: int, floor_no: int = None) -> dict:
    """
    Load optimized split location cache.
    building_no=0 -> outdoor.json
    else -> {b}_{f}.json
    
    캐시 적용으로 재로딩 방지
    """
    import json
    from pathlib import Path
    try:
        # Cache 루트의 공통 location_maps 폴더 사용
        base_dir = Path(cache_path) / "location_maps"
        if building_no == 0:
            fname = "outdoor.json"
        else:
            fname = f"{building_no}_{floor_no}.json"
        
        fpath = base_dir / fname
        if fpath.exists():
             with open(fpath, 'r', encoding='utf-8') as f:
                 # Keys are strings in JSON, keep them as strings or convert?
                 # main logic treats keys as ints usually, but let's check parsing.
                 # JSON keys are always strings.
                 return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading split location cache for {building_no}_{floor_no}: {e}")
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def load_floor_info_cached() -> pd.DataFrame:
    """Floor 정보 로드 (irfm.csv에서 length_x, length_y 추출)"""
    data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    irfm_path = data_folder / 'irfm.csv'
    if irfm_path.exists():
        df = pd.read_csv(irfm_path)
        return df[['floor_number', 'building_number', 'floor_name', 'length_x', 'length_y']].copy()
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_spot_info_cached() -> pd.DataFrame:
    """Spot 정보 로드 (spot.csv)"""
    data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    spot_path = data_folder / 'spot.csv'
    if spot_path.exists():
        return pd.read_csv(spot_path)
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_spot_position_cached() -> pd.DataFrame:
    """Spot 위치 정보 로드 (spot_position.csv)"""
    data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    spot_pos_path = data_folder / 'spot_position.csv'
    if spot_pos_path.exists():
        return pd.read_csv(spot_pos_path)
    return pd.DataFrame()


# ==================== 페이지 설정 ====================
# Only run page config when this file is executed directly, not when imported
if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="SK Hynix Y1 - IRFM Dashboard",
            page_icon="🏭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass  # Ignore if already set

# ==================== 테마 & 스타일 ====================
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

# Apply theme only when running as main file
if __name__ == "__main__":
    st.markdown(f"""
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
    
    /* 진한 배경용 흰색 텍스트 예외 (우선순위 높음) */
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
    
    /* st.info, st.warning, st.error 내부 텍스트 */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    .stAlert p, .stAlert span {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* st.metric 스타일 */
    [data-testid="stMetric"],
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        color: {THEME['text_primary']} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {THEME['primary']} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {THEME['text_secondary']} !important;
    }}
    
    /* DataFrame/Table 스타일 */
    .stDataFrame, .stDataFrame td, .stDataFrame th,
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* 버튼 텍스트 */
    .stButton button, .stDownloadButton button {{
        color: white !important;
        background-color: {THEME['primary']} !important;
    }}
    
    /* selectbox, radio 등 */
    .stSelectbox label, .stRadio label, .stCheckbox label {{
        color: {THEME['text_primary']} !important;
    }}
    .stSelectbox > div > div {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* 메인 화면 selectbox 드롭다운 스타일 */
    [data-baseweb="select"] > div {{
        background: white !important;
        border: 1px solid {THEME['border']} !important;
    }}
    [data-baseweb="select"] span {{
        color: {THEME['text_primary']} !important;
    }}
    /* 드롭다운 메뉴 (펜처진 상태) */
    [data-baseweb="popover"] {{
        background: white !important;
    }}
    [data-baseweb="menu"] {{
        background: white !important;
    }}
    [data-baseweb="menu"] li {{
        background: white !important;
        color: {THEME['text_primary']} !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background: {THEME['bg_page']} !important;
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
    .metric-value.gray {{
        color: {THEME['text_secondary']};
    }}
    .metric-label {{
        color: {THEME['text_secondary']};
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }}
    .metric-delta {{
        font-size: 0.8rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
        margin-top: 0.5rem;
        font-weight: 500;
    }}
    .metric-delta.positive {{
        background: rgba(16,185,129,0.1);
        color: {THEME['success']};
    }}
    .metric-delta.warning {{
        background: rgba(245,158,11,0.1);
        color: {THEME['warning']};
    }}
    .metric-delta.negative {{
        background: rgba(239,68,68,0.1);
        color: {THEME['danger']};
    }}
    .metric-delta.info {{
        background: rgba(59,130,246,0.1);
        color: {THEME['primary']};
    }}
    
    /* 섹션 카드 */
    .section-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid {THEME['border']};
        margin-bottom: 1rem;
    }}
    .section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {THEME['text_primary']};
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* ========== 사이드바 스타일 ========== */
    [data-testid="stSidebar"] {{
        background: {THEME['dark']} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: {THEME['dark']} !important;
    }}
    /* 사이드바 내 모든 텍스트 흰색 */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: white !important;
    }}
    /* 사이드바 selectbox */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stRadio label {{
        color: rgba(255,255,255,0.9) !important;
    }}
    /* 사이드바 selectbox 드롭다운 박스 */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] span {{
        color: white !important;
    }}
    
    /* 사이드바 로고 영역 */
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
    .sidebar-logo p {{
        font-size: 0.75rem;
        color: rgba(255,255,255,0.6) !important;
        margin: 0.5rem 0 0 0;
    }}
    
    /* 사이드바 정보 박스 */
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
        background: transparent !important;
        transition: all 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {{
        color: {THEME['text_secondary']} !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: {THEME['bg_page']} !important;
    }}
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {{
        color: {THEME['text_primary']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {THEME['primary']} !important;
    }}
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {{
        color: white !important;
    }}
    /* 탭 패널 컨텐츠 */
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] span,
    .stTabs [data-baseweb="tab-panel"] div {{
        color: {THEME['text_primary']};
    }}
    
    /* 차트 컨테이너 */
    .chart-container {{
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid {THEME['border']};
    }}
    
    /* 데이터 테이블 */
    .dataframe {{
        border-radius: 8px !important;
        overflow: hidden;
    }}
    
    /* ========== Streamlit 기본 요소 오버라이드 ========== */
    /* 컬럼 내 텍스트 */
    [data-testid="column"] p,
    [data-testid="column"] span,
    [data-testid="column"] div {{
        color: {THEME['text_primary']};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        color: {THEME['text_primary']} !important;
        background: white !important;
    }}
    .streamlit-expanderContent {{
        color: {THEME['text_primary']} !important;
        background: white !important;
    }}
    
    /* Caption */
    .stCaption, [data-testid="stCaption"] {{
        color: {THEME['text_secondary']} !important;
    }}
    
    /* Code block */
    .stCodeBlock, code {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* JSON viewer */
    [data-testid="stJson"] {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* ========== 스크롤바 ========== */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {THEME['bg_page']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {THEME['text_muted']};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {THEME['text_secondary']};
    }}
    
    /* 숨기기 */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
</style>
""", unsafe_allow_html=True)

# ==================== 유틸리티 함수 ====================

# FIXED: Use 5-minute resolution for consistency with forecast engine
# All data processing (T31/T41 tabs, Simulator, Forecast) uses 5-min aggregation
CACHE_RESOLUTION = '5min'

def format_number(num, decimals=0):
    """숫자 포맷팅"""
    if pd.isna(num):
        return "N/A"
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        if decimals > 0:
            return f"{num:,.{decimals}f}"
        return f"{num:,.0f}"

def get_chart_layout(title='', height=400, show_legend=True):
    """Plotly 차트 기본 레이아웃 반환 - 모든 텍스트 색상 명시적 설정"""
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=THEME['text_primary'])
        ),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=THEME['text_primary'], size=12),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.08)',
            linecolor=THEME['border'],
            tickfont=dict(color=THEME['text_secondary']),
            title_font=dict(color=THEME['text_secondary'])
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.08)',
            linecolor=THEME['border'],
            tickfont=dict(color=THEME['text_secondary']),
            title_font=dict(color=THEME['text_secondary'])
        ),
        legend=dict(
            font=dict(color=THEME['text_primary']),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ) if show_legend else dict(visible=False),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode='x unified'
    )

def time_index_to_time_str(idx):
    """시간 인덱스 (5분 단위) → 시간 문자열"""
    hours = (idx * 5) // 60
    minutes = (idx * 5) % 60
    return f"{hours:02d}:{minutes:02d}"

def bin_index_to_time_str(idx):
    """bin_index → 시간 문자열 (5분 단위)"""
    hours = (idx * 5) // 60
    minutes = (idx * 5) % 60
    return f"{hours:02d}:{minutes:02d}"

def get_flow_cache(loader: 'CachedDataLoader') -> pd.DataFrame:
    """Streamlit 캐시를 사용하여 flow_cache 로드"""
    # Pass the exact cache folder (date-specific) to avoid returning
    # stale cached results when using the parent Cache folder.
    cache_folder = str(loader.cache_folder)
    return load_flow_cache_cached(cache_folder, loader.date_str, CACHE_RESOLUTION)

def render_location_filter(loader: 'CachedDataLoader', key_prefix: str):
    """
    메인 화면에 위치 필터 UI 렌더링 (Spot 제외)
    - 공간 구조: Sector → Building → Floor (연결됨)
    Returns: (building, floor)
    """
    st.markdown("##### 📍 위치 필터")
    
    # 공간 구조 필터 (한 줄에 3개)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col1:
        # Sector (현재 하나뿐이므로 고정)
        sector_options = ['Y-Project']
        selected_sector = st.selectbox(
            "Sector",
            sector_options,
            index=0,
            key=f'{key_prefix}_sector'
        )
    
    with col2:
        # Building 선택
        building_options = loader.get_building_list()
        selected_building = st.selectbox(
            "Building",
            building_options,
            index=0,
            key=f'{key_prefix}_building'
        )
    
    with col3:
        # Floor 선택 (Building에 따라 동적으로 변경)
        floor_options = loader.get_floor_list(selected_building)
        selected_floor = st.selectbox(
            "Floor",
            floor_options,
            index=0,
            key=f'{key_prefix}_floor'
        )
    
    # 현재 필터 상태 표시
    filter_parts = []
    if selected_building != 'All':
        filter_parts.append(selected_building)
    if selected_floor != 'All':
        filter_parts.append(selected_floor)
    filter_desc = ' > '.join(filter_parts) if filter_parts else '전체'
    
    st.caption(f"🔍 현재 필터: **{filter_desc}**")
    st.markdown("---")
    
    return selected_building, selected_floor

# ==================== 사이드바 ====================
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
        
        # 캐시 폴더 경로
        cache_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cache")
        
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
            key="sidebar_date_select" # Unique key
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
            
            # Mobile 정보 (summary에서는 'mobile' 키 사용)
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

        # (Zone Sort control removed from sidebar — Simulator-only control moved into Simulator tab)

        return loader

# ==================== Overview 탭 ====================
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
    st.markdown("### 🔧 T-Ward Type31 (Equipment)")
    
    # === 데이터 소스: t31_time_series 캐시 사용 (경량 캐시) ===
    t31_time_series = loader.load_t31_time_series()
    
    if t31_time_series is not None and not t31_time_series.empty:
        t31_devices = t31_time_series['total_devices'].iloc[0]  # 모니터링 대상 장비 수
        
        # 일과시간 (07:00~19:00, time_index 85~228) 평균 가동률 계산
        # time_index 85 = 07:00, time_index 228 = 18:55
        work_hours_ts = t31_time_series[
            (t31_time_series['time_index'] >= 85) & 
            (t31_time_series['time_index'] <= 228)
        ]
        
        if not work_hours_ts.empty and t31_devices > 0:
            # 각 5분 단위별 active_devices / total_devices 의 평균
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
        # T31 5분 단위 가동 장비 차트 - t31_time_series 캐시 사용
        if t31_time_series is not None and not t31_time_series.empty:
            # 이미 288개 전체 time_index 포함 (active/inactive 구분)
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
                hovertemplate='<b>%{x}</b><br>가동: %{y}대<extra></extra>' # Fixed tooltip
            ))
            
            # Inactive 영역 (비활성 - 신호 없으나 장비 존재)
            fig.add_trace(go.Scatter(
                x=t31_time_series['time_label'],
                y=t31_time_series['total_devices'],
                mode='lines',
                fill='tonexty',
                name='비가동 (Inactive)',
                line=dict(color=THEME['t41_inactive'], width=1),
                fillcolor='rgba(203, 213, 225, 0.4)',
                hovertemplate='<b>%{x}</b><br>전체: %{y}대<extra></extra>' # Fixed tooltip
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
    
    st.markdown("---")
    
    # ===== T41 섹션 =====
    st.markdown("### 👷 T-Ward Type41 (Workers)")
    
    t41_info = summary.get('t41', {}) if summary else {}
    t41_workers = t41_info.get('total_workers', 0)
    
    # 새로운 통합 캐시에서 데이터 가져오기
    max_active = t41_info.get('max_active', 0)
    avg_active = int(t41_info.get('avg_active', 0))
    avg_dwell = t41_info.get('avg_dwell_minutes', 0)
    
    # 활성률 계산: avg_active / total_workers
    t41_activity = (avg_active / t41_workers * 100) if t41_workers > 0 else 0
    
    # t41_time_series 캐시 로드 (차트용)
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
    
    # T41 시간별 Active/Inactive 차트 (새로운 통합 캐시 사용)
    if t41_time_series is not None and not t41_time_series.empty:
        fig = go.Figure()
        
        # Active/Inactive/Total 데이터 준비
        if 'active_workers' in t41_time_series.columns and 'inactive_workers' in t41_time_series.columns:
            t41_ts = t41_time_series.copy()
            t41_ts['total_workers'] = t41_ts['active_workers'] + t41_ts['inactive_workers']
            customdata = t41_ts[['active_workers', 'inactive_workers', 'total_workers']].values
            
            # Active 영역 (아래)
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
            
            # Inactive 영역 (위) - Total 기준으로 표시
            fig.add_trace(go.Scatter(
                x=t41_ts['time_label'],
                y=t41_ts['total_workers'],
                fill='tonexty',
                fillcolor=f"rgba(148, 163, 184, 0.5)",
                line=dict(color=THEME['t41_inactive'], width=2),
                name='Inactive (영역)',
                hoverinfo='skip'  # 첫 번째 trace에서 모든 정보 표시하므로 중복 방지
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
        
        # 그래프 설명 추가
        st.caption("※ 각 시점(5분)에 데이터가 있는 Unique MAC 수. 초록=활성(움직임 감지), 회색=비활성(정지 상태). 비활성 수는 '해당 시점에 정지 상태로 감지된 수'로, 미사용 태그(하루 종일 활성 없음)와는 다릅니다.")
    
    # T41 미사용 태그 통계 정보 (summary.json에서 로드 - 빠름)
    t41_info = summary.get('t41', {}) if summary else {}
    unused_tags = t41_info.get('unused_tags', 0)
    
    if unused_tags > 0:
        all_t41_macs = t41_info.get('total_workers', 0)
        active_t41_macs = t41_info.get('active_workers', 0)
        unused_pct = unused_tags / all_t41_macs * 100 if all_t41_macs > 0 else 0
        full_day_count = t41_info.get('unused_full_day', 0)
        partial_count = t41_info.get('unused_partial', 0)
        avg_time_slots = t41_info.get('unused_avg_time_slots', 0)
        
        with st.expander("📋 T41 태그 품질 정보", expanded=False):
            st.markdown(f"""
            | 구분 | 태그 수 | 비율 | 설명 |
            |------|---------|------|------|
            | 전체 감지 태그 | **{all_t41_macs:,}개** | 100% | 해당 일자에 1회 이상 수신된 모든 T41 태그 |
            | 활성 기록 있음 | **{active_t41_macs:,}개** | {active_t41_macs/all_t41_macs*100:.1f}% | 1회 이상 움직임(진동)이 감지된 태그 |
            | **미사용 태그** | **{unused_tags:,}개** | **{unused_pct:.1f}%** | 감지되었으나 활성 기록 없음 (보관소 추정) |
            
            ---
            
            #### 💡 미사용 태그 상세 분석
            
            **미사용 태그**는 하루 종일 수신은 되었으나 한 번도 활성(움직임) 상태가 감지되지 않은 태그입니다.  
            주로 **헬멧 보관소**에 보관 중인 태그로 추정되며, 체류시간 분석에서 자동으로 제외됩니다.
            
            | 수신 패턴 | 태그 수 | 설명 |
            |----------|---------|------|
            | 24시간 연속 수신 | {full_day_count:,}개 | 288개 time_index 모두 수신 (안정적 보관 위치) |
            | 부분적 수신 | {partial_count:,}개 | 일부 시간대만 수신 (평균 {avg_time_slots:.0f}개 time_index) |
            
            ⚠️ **참고사항**  
            미사용 태그 중 **{partial_count:,}개**({partial_count/unused_tags*100:.1f}% if unused_tags > 0 else 0)는 일부 시간대에만 수신되었습니다.  
            이는 **보관소 위치의 AP 신호 수신이 불안정**하거나, 태그가 절전 모드로 인해 간헐적으로 신호를 방송하기 때문일 수 있습니다.  
            따라서 시간대별 비활성 인원 수가 미사용 태그 수보다 적게 표시될 수 있습니다.
            """)
    
    st.markdown("---")
    
    # ===== MobilePhone 섹션 =====
    st.markdown("### 📱 MobilePhone")
    
    # 새로운 summary에서 mobile 정보 가져오기
    mobile_info = summary.get('mobile', {}) if summary else {}
    flow_devices = mobile_info.get('total_devices', 0)
    android = mobile_info.get('android_devices', 0)
    iphone = mobile_info.get('iphone_devices', 0)
    max_concurrent = mobile_info.get('max_concurrent', 0)
    
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
            hovertemplate='<b>%{x}</b><br>Devices: %{y:,}<extra></extra>' # Fixed tooltip
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

def render_mobile_zone_analysis(loader: CachedDataLoader):
    """
    Mobile Phone 구역별 분석 - T41과 유사한 형태
    - 위치 필터 → 시간별 기기 수 추이
    - Spot별 분석
    """
    try:
        # 메인 화면에 위치 필터 UI 표시 (Spot 제외)
        selected_building, selected_floor = render_location_filter(loader, 'mobile_zone')
        
        df = get_flow_cache(loader)
        if not df.empty:
            # Mobile: Type 1 (iPhone), Type 10 (Android)
            mobile_data = df[df['type'].isin([1, 10])].copy()
            
            # 필터 적용 (Spot 제외)
            mobile_data = loader.filter_by_location(
                mobile_data, 
                selected_building,
                selected_floor,
                'All'  # Spot은 적용하지 않음
            )
            
            if not mobile_data.empty:
                # time_index별 Android/iPhone 집계 (Unique MAC)
                time_agg = mobile_data.groupby(['time_index', 'type']).agg({
                    'mac_address': 'nunique'
                }).reset_index()
                time_agg.columns = ['time_index', 'type', 'count']
                
                # pivot
                pivot_data = time_agg.pivot(index='time_index', columns='type', values='count').fillna(0).reset_index()
                pivot_data.columns.name = None
                
                # 컬럼 이름 정리
                rename_map = {1: 'iPhone', 10: 'Android'}
                pivot_data = pivot_data.rename(columns=rename_map)
                
                # 시간 레이블 생성
                pivot_data['time_label'] = pivot_data['time_index'].apply(bin_index_to_time_str)
                pivot_data = pivot_data.sort_values('time_index')
                
                # 필터 설명 생성
                filter_parts = []
                if selected_building != 'All':
                    filter_parts.append(selected_building)
                if selected_floor != 'All':
                    filter_parts.append(selected_floor)
                filter_desc = ' > '.join(filter_parts) if filter_parts else '전체 구역'
                
                # 영역 차트 (Android/iPhone 구분)
                fig = go.Figure()
                
                if 'Android' in pivot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['Android'],
                        fill='tozeroy',
                        fillcolor='rgba(34, 197, 94, 0.6)',  # Android 녹색
                        line=dict(color=THEME['mobile_android'], width=2),
                        name='Android',
                        hovertemplate='<b>%{x}</b><br>Android: %{y:,}대<extra></extra>' # Fixed tooltip
                    ))
                
                if 'iPhone' in pivot_data.columns:
                    # 전체 = Android + iPhone
                    if 'Android' in pivot_data.columns:
                        total = pivot_data['Android'] + pivot_data['iPhone']
                    else:
                        total = pivot_data['iPhone']
                    
                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=total,
                        fill='tonexty',
                        fillcolor='rgba(59, 130, 246, 0.5)',  # iPhone 파란색
                        line=dict(color=THEME['mobile_iphone'], width=2),
                        name='iPhone',
                        hovertemplate='<b>%{x}</b><br>iPhone: %{y:,}대<extra></extra>' # Fixed tooltip
                    ))
                
                fig.update_layout(
                    title=dict(text=f'시간별 모바일 기기 수 추이 ({filter_desc})', font=dict(size=14, color=THEME['text_primary'])),
                    xaxis_title='Time',
                    yaxis_title='기기 수 (Unique MAC)',
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 요약
                col1, col2, col3, col4 = st.columns(4)
                
                android_col = 'Android' if 'Android' in pivot_data.columns else None
                iphone_col = 'iPhone' if 'iPhone' in pivot_data.columns else None
                
                if android_col:
                    with col1:
                        st.metric("최대 Android", f"{int(pivot_data[android_col].max()):,}대")
                    with col2:
                        st.metric("평균 Android", f"{pivot_data[android_col].mean():.0f}대")
                
                if iphone_col:
                    with col3:
                        st.metric("최대 iPhone", f"{int(pivot_data[iphone_col].max()):,}대")
                    with col4:
                        st.metric("평균 iPhone", f"{pivot_data[iphone_col].mean():.0f}대")
                
                # ===== Spot 분석 (별도) =====
                st.markdown("---")
                st.markdown("#### 📍 Spot별 모바일 기기 분포")
                render_mobile_spot_analysis(loader, mobile_data)
                
            else:
                st.info("선택된 필터 조건에 해당하는 Mobile Phone 데이터가 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_mobile_spot_analysis(loader: CachedDataLoader, mobile_data: pd.DataFrame):
    """Mobile Spot별 분석 - T41과 유사한 형태"""
    try:
        if 'spot_nos' not in mobile_data.columns:
            st.info("spot_nos 컬럼이 없습니다.")
            return
        
        # Spot 목록 추출
        all_spots = set()
        for spots_str in mobile_data['spot_nos'].dropna():
            for spot in str(spots_str).split(','):
                spot = spot.strip()
                if spot and spot != 'nan':
                    all_spots.add(spot)
        
        if not all_spots:
            st.info("Spot 데이터가 없습니다.")
            return
        
        spot_list = sorted(list(all_spots), key=lambda x: int(x) if x.isdigit() else 0)
        
        # Spot 이름 매핑
        spot_names = loader.get_spot_names() if hasattr(loader, 'get_spot_names') else {}
        
        # ===== 1. Spot 선택 → 시간별 기기 수 추이 =====
        st.markdown("##### 📈 Spot별 시간대 기기 추이")
        
        spot_options = [spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s for s in spot_list]
        spot_value_map = {}
        for s in spot_list:
            name = spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s
            spot_value_map[name] = s
        
        selected_spot_name = st.selectbox(
            "Spot 선택",
            spot_options,
            index=0,
            key='mobile_spot_trend'
        )
        selected_spot = spot_value_map.get(selected_spot_name, spot_list[0])
        
        # 선택된 Spot의 시간별 기기 추이
        spot_time_data = []
        for _, row in mobile_data.iterrows():
            spots = str(row['spot_nos']).split(',') if pd.notna(row['spot_nos']) else []
            if selected_spot in [s.strip() for s in spots]:
                spot_time_data.append({
                    'time_index': row['time_index'],
                    'mac_address': row['mac_address'],
                    'type': row['type']
                })
        
        if spot_time_data:
            spot_time_df = pd.DataFrame(spot_time_data)
            
            # time_index별 Android/iPhone 집계
            time_agg = spot_time_df.groupby(['time_index', 'type']).agg({
                'mac_address': 'nunique'
            }).reset_index()
            time_agg.columns = ['time_index', 'type', 'count']
            
            pivot_time = time_agg.pivot(index='time_index', columns='type', values='count').fillna(0).reset_index()
            pivot_time.columns.name = None
            pivot_time = pivot_time.rename(columns={1: 'iPhone', 10: 'Android'})
            pivot_time['time_label'] = pivot_time['time_index'].apply(bin_index_to_time_str)
            pivot_time = pivot_time.sort_values('time_index')
            
            # 영역 차트
            fig = go.Figure()
            
            if 'Android' in pivot_time.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_time['time_label'],
                    y=pivot_time['Android'],
                    fill='tozeroy',
                    fillcolor='rgba(34, 197, 94, 0.6)',
                    line=dict(color=THEME['mobile_android'], width=2),
                    name='Android',
                    hovertemplate='<b>%{x}</b><br>Android: %{y:,}대<extra></extra>' # Fixed tooltip
                ))
            
            if 'iPhone' in pivot_time.columns:
                if 'Android' in pivot_time.columns:
                    total = pivot_time['Android'] + pivot_time['iPhone']
                else:
                    total = pivot_time['iPhone']
                
                fig.add_trace(go.Scatter(
                    x=pivot_time['time_label'],
                    y=total,
                    fill='tonexty',
                    fillcolor='rgba(59, 130, 246, 0.5)',
                    line=dict(color=THEME['mobile_iphone'], width=2),
                    name='iPhone',
                    hovertemplate='<b>%{x}</b><br>iPhone: %{y:,}대<extra></extra>' # Fixed tooltip
                ))
            
            fig.update_layout(
                title=dict(text=f'{selected_spot_name} - 시간별 기기 수 추이', font=dict(size=14, color=THEME['text_primary'])),
                xaxis_title='Time',
                yaxis_title='기기 수 (Unique MAC)',
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['text_primary']),
                xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 요약
            col1, col2, col3, col4 = st.columns(4)
            android_col = 'Android' if 'Android' in pivot_time.columns else None
            iphone_col = 'iPhone' if 'iPhone' in pivot_time.columns else None
            
            if android_col:
                with col1:
                    st.metric("최대 Android", f"{int(pivot_time[android_col].max()):,}대")
                with col2:
                    st.metric("평균 Android", f"{pivot_time[android_col].mean():.0f}대")
            
            if iphone_col:
                with col3:
                    st.metric("최대 iPhone", f"{int(pivot_time[iphone_col].max()):,}대")
                with col4:
                    st.metric("평균 iPhone", f"{pivot_time[iphone_col].mean():.0f}대")
        else:
            st.info("선택된 Spot에 데이터가 없습니다.")
        
        st.markdown("---")
        
        # ===== 2. 시간대 선택 → Spot별 분포 비교 =====
        st.markdown("##### 📊 시간대별 Spot 분포 비교")
        
        # 시간 선택지 (시작/종료)
        time_options = [bin_index_to_time_str(i) for i in range(288)]
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.selectbox(
                "시작 시간",
                time_options,
                index=0,
                key='mobile_spot_start_time'
            )
        with col2:
            end_time = st.selectbox(
                "종료 시간",
                time_options,
                index=min(17, len(time_options)-1),
                key='mobile_spot_end_time'
            )
        
        # 시간 인덱스 변환
        start_idx = time_options.index(start_time)
        end_idx = time_options.index(end_time)
        
        if start_idx > end_idx:
            st.warning("시작 시간이 종료 시간보다 큽니다.")
        else:
            # 선택된 시간대 필터링
            time_filtered = mobile_data[(mobile_data['time_index'] >= start_idx) & (mobile_data['time_index'] <= end_idx)].copy()
            
            if not time_filtered.empty:
                # Spot별 집계
                spot_data = []
                for _, row in time_filtered.iterrows():
                    spots = str(row['spot_nos']).split(',') if pd.notna(row['spot_nos']) else []
                    for spot in spots:
                        spot = spot.strip()
                        if spot and spot != 'nan':
                            spot_data.append({
                                'spot_no': spot,
                                'mac_address': row['mac_address'],
                                'type': row['type']
                            })
                
                if spot_data:
                    spot_df = pd.DataFrame(spot_data)
                    
                    spot_agg = spot_df.groupby(['spot_no', 'type']).agg({
                        'mac_address': 'nunique'
                    }).reset_index()
                    spot_agg.columns = ['spot_no', 'type', 'count']
                    
                    pivot_spot = spot_agg.pivot(index='spot_no', columns='type', values='count').fillna(0).reset_index()
                    pivot_spot.columns.name = None
                    pivot_spot = pivot_spot.rename(columns={1: 'iPhone', 10: 'Android'})
                    
                    pivot_spot['spot_name'] = pivot_spot['spot_no'].apply(
                        lambda x: spot_names.get(int(x), f'Spot {x}') if str(x).isdigit() else x
                    )
                    
                    fig = go.Figure()
                    
                    if 'Android' in pivot_spot.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_spot['spot_name'],
                            y=pivot_spot['Android'],
                            name='Android',
                            marker_color=THEME['mobile_android'],
                            hovertemplate='<b>Spot: %{x}</b><br>Android: %{y:,}대<extra></extra>' # Fixed tooltip
                        ))
                    
                    if 'iPhone' in pivot_spot.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_spot['spot_name'],
                            y=pivot_spot['iPhone'],
                            name='iPhone',
                            marker_color=THEME['mobile_iphone'],
                            hovertemplate='<b>Spot: %{x}</b><br>iPhone: %{y:,}대<extra></extra>' # Fixed tooltip
                        ))
                    
                    fig.update_layout(
                        title=dict(text=f'Spot별 모바일 기기 분포 ({start_time} ~ {end_time})', font=dict(size=14, color=THEME['text_primary'])),
                        barmode='stack',
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=THEME['text_primary']),
                        xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='Spot', font=dict(color=THEME['text_secondary'])), tickangle=45),
                        yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='기기 수', font=dict(color=THEME['text_secondary']))),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 요약
                    total_android = int(pivot_spot['Android'].sum()) if 'Android' in pivot_spot.columns else 0
                    total_iphone = int(pivot_spot['iPhone'].sum()) if 'iPhone' in pivot_spot.columns else 0
                    st.caption(f"📱 Spot 총 {len(pivot_spot)}개 | Android: {total_android}대 | iPhone: {total_iphone}대")
                else:
                    st.info("선택된 시간대에 Spot 데이터가 없습니다.")
            else:
                st.info("선택된 시간대에 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"Spot 분석 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ==================== T31 탭 ====================
def render_t31_tab(loader: CachedDataLoader):
    """T31 장비 분석 탭"""
    st.markdown("""
    <div class="main-header">
        <h1>🔧 T-Ward Type31</h1>
        <p>장비 (Table Lift) 가동 분석</p>
    </div>
    """, unsafe_allow_html=True)
    
    sub_tabs = st.tabs(["📊 가동 현황", "🏢 구역별 분석", "📈 상세 통계"])
    
    with sub_tabs[0]:
        render_t31_operation(loader)
    
    with sub_tabs[1]:
        render_t31_sward(loader)
    
    with sub_tabs[2]:
        render_t31_statistics(loader)

def render_t31_operation(loader: CachedDataLoader):
    """T31 가동 현황 - 장비가 데이터에 존재하면 '가동 중'으로 판단
    
    Note: T31 장비는 status 필드가 아닌, 해당 시간에 데이터가 존재하면 가동 중으로 판단
    """
    try:
        df = get_flow_cache(loader)
        if not df.empty:
            t31_data = df[df['type'] == 31].copy()
            
            if not t31_data.empty:
                # 시간별 차트
                st.markdown("#### 시간별 장비 현황")
                
                # T31: 데이터 존재 = 가동 중 (status 무시)
                # time_index별 가동 장비 수 집계 (Unique MAC)
                time_agg = t31_data.groupby('time_index').agg({
                    'mac_address': 'nunique'
                }).reset_index()
                time_agg.columns = ['time_index', 'active']
                
                # 전체 장비 수 (하루 동안 한 번이라도 나타난 장비)
                total_devices = t31_data['mac_address'].nunique()
                
                # 비활성 = 전체 - 해당 시점 가동 장비
                time_agg['inactive'] = total_devices - time_agg['active']
                time_agg['inactive'] = time_agg['inactive'].clip(lower=0)  # 음수 방지
                
                pivot_data = time_agg.copy()
                pivot_data['time_label'] = pivot_data['time_index'].apply(bin_index_to_time_str)
                pivot_data = pivot_data.sort_values('time_index')
                
                # 영역 차트 (Active/Inactive 구분)
                fig = go.Figure()
                
                if 'active' in pivot_data.columns:
                    # Ensure total column exists for tooltip
                    if 'total_devices' not in pivot_data.columns:
                         pivot_data['total_devices'] = pivot_data['active'] + pivot_data.get('inactive', 0)

                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['active'],
                        stackgroup='one',
                        fillcolor='rgba(249, 115, 22, 0.6)',  # T31 오렌지색
                        line=dict(color=THEME['t31'], width=2),
                        name='활성 장비',
                        hovertemplate='<b>%{x}</b><br><b>활성</b>: %{y}대<br><b>전체</b>: %{customdata[0]}대<extra></extra>',
                        customdata=pivot_data[['total_devices']]
                    ))
                
                if 'inactive' in pivot_data.columns:
                    # Inactive trace (stacked on Active)
                    # We plot raw 'inactive' values, stackgroup handles stacking
                    
                    # Ensure total is available for tooltip
                    if 'total_devices' not in pivot_data.columns:
                         pivot_data['total_devices'] = pivot_data.get('active', 0) + pivot_data['inactive']

                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['inactive'],
                        stackgroup='one',
                        fillcolor='rgba(148, 163, 184, 0.5)',
                        line=dict(color=THEME['t41_inactive'], width=2),
                        name='비활성 장비',
                        hovertemplate='<b>%{x}</b><br><b>비활성</b>: %{y}대<br><b>전체</b>: %{customdata[0]}대<extra></extra>',
                        customdata=pivot_data[['total_devices']]
                    ))
                
                fig.update_layout(
                    title=dict(text='시간별 장비 현황', font=dict(size=14, color=THEME['text_primary'])),
                    xaxis_title='Time',
                    yaxis_title='장비 수 (Unique MAC)',
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 요약
                col1, col2, col3, col4 = st.columns(4)
                
                active_col = 'active' if 'active' in pivot_data.columns else None
                inactive_col = 'inactive' if 'inactive' in pivot_data.columns else None
                
                if active_col:
                    with col1:
                        st.metric("최대 활성 장비", f"{int(pivot_data[active_col].max()):,}대")
                    with col2:
                        st.metric("평균 활성 장비", f"{pivot_data[active_col].mean():.0f}대")
                
                if inactive_col:
                    with col3:
                        st.metric("최대 비활성 장비", f"{int(pivot_data[inactive_col].max()):,}대")
                    with col4:
                        st.metric("평균 비활성 장비", f"{pivot_data[inactive_col].mean():.0f}대")
                
                # Building별 장비 분포 차트
                st.markdown("---")
                st.markdown("#### Building별 장비 분포")
                render_t31_building_distribution(loader, t31_data)
            else:
                st.info("T31 데이터가 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")

def render_t31_building_distribution(loader: CachedDataLoader, t31_data: pd.DataFrame):
    """T31 Building별 장비 분포 차트 (가동 현황 서브탭용)
    
    Note: T31은 status 대신 '주 Building'에서의 가동 빈도로 표시
    """
    try:
        if not t31_data.empty:
            building_names = loader.get_building_names()
            
            t31_data = t31_data.copy()
            t31_data['building_name'] = t31_data['building_no'].map(
                lambda x: building_names.get(int(x), f'Building {x}') if pd.notna(x) else '알 수 없음'
            )
            
            # T31: 데이터 존재 = 가동 중 (status 무시)
            # 각 MAC의 '주 Building' 결정 (가장 많이 나타난 Building)
            mac_building = t31_data.groupby(['mac_address', 'building_name']).size().reset_index(name='count')
            mac_primary_building = mac_building.loc[mac_building.groupby('mac_address')['count'].idxmax()]
            
            # Building별 장비 수
            pivot_data = mac_primary_building.groupby('building_name').size().reset_index(name='active_count')
            
            fig = go.Figure()
            
            # T31은 존재하면 가동 중으로 표시
            fig.add_trace(go.Bar(
                x=pivot_data['building_name'],
                y=pivot_data['active_count'],
                name='가동 장비',
                marker_color=THEME['t31'],
                hovertemplate='<b>Building: %{x}</b><br>장비 수: %{y:,}대<extra></extra>' # Fixed tooltip
            ))
            
            fig.update_layout(
                title=dict(text='Building별 장비 분포 (Unique MAC)', font=dict(size=14, color=THEME['text_primary'])),
                barmode='stack',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['text_primary']),
                xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='Building', font=dict(color=THEME['text_secondary']))),
                yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='장비 수', font=dict(color=THEME['text_secondary']))),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 요약
            total_devices = int(pivot_data['active_count'].sum())
            st.caption(f"🔧 총 장비: {total_devices:,}대 (하루 동안 감지된 Unique MAC)")
        else:
            st.info("T31 데이터가 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")

def render_t31_sward(loader: CachedDataLoader):
    """
    T31 구역별 분석 - 선택한 구역의 시간별 장비 수
    
    Note: T31 장비는 데이터 존재 = 가동 중으로 판단 (status 무시)
    """
    try:
        # 메인 화면에 위치 필터 UI 표시 (Spot 제외)
        selected_building, selected_floor = render_location_filter(loader, 't31_zone')
        
        df = get_flow_cache(loader)
        if not df.empty:
            t31_data = df[df['type'] == 31].copy()
            
            # 필터 적용 (Spot 제외)
            t31_data = loader.filter_by_location(
                t31_data,
                selected_building,
                selected_floor,
                'All'  # Spot은 적용하지 않음
            )
            
            if not t31_data.empty:
                # T31: 데이터 존재 = 가동 중 (status 무시)
                # time_index별 가동 장비 수 집계 (Unique MAC)
                time_agg = t31_data.groupby('time_index').agg({
                    'mac_address': 'nunique'
                }).reset_index()
                time_agg.columns = ['time_index', 'active']
                
                # 전체 장비 수 (필터 적용 후)
                total_devices = t31_data['mac_address'].nunique()
                
                # 비활성 = 전체 - 해당 시점 가동 장비
                time_agg['inactive'] = total_devices - time_agg['active']
                time_agg['inactive'] = time_agg['inactive'].clip(lower=0)
                
                pivot_data = time_agg.copy()
                
                # 시간 레이블 생성
                pivot_data['time_label'] = pivot_data['time_index'].apply(bin_index_to_time_str)
                pivot_data = pivot_data.sort_values('time_index')
                
                # 필터 설명 생성
                filter_parts = []
                if selected_building != 'All':
                    filter_parts.append(selected_building)
                if selected_floor != 'All':
                    filter_parts.append(selected_floor)
                filter_desc = ' > '.join(filter_parts) if filter_parts else '전체 구역'
                
                # 영역 차트 (Active/Inactive 구분)
                fig = go.Figure()
                
                if 'active' in pivot_data.columns:
                    # Ensure total column exists for tooltip
                    if 'total' not in pivot_data.columns:
                        pivot_data['total'] = pivot_data['active'] + pivot_data.get('inactive', 0)

                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['active'],
                        stackgroup='one',
                        fillcolor='rgba(249, 115, 22, 0.6)',  # T31 오렌지색
                        line=dict(color=THEME['t31'], width=2),
                        name='활성 장비',
                        hovertemplate='<b>%{x}</b><br><b>활성</b>: %{y}대<br><b>전체</b>: %{customdata[0]}대<extra></extra>',
                        customdata=pivot_data[['total']]
                    ))
                
                if 'inactive' in pivot_data.columns:
                    if 'total' not in pivot_data.columns:
                         pivot_data['total'] = pivot_data.get('active', 0) + pivot_data['inactive']

                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['inactive'],
                        stackgroup='one',
                        fillcolor='rgba(148, 163, 184, 0.5)',
                        line=dict(color=THEME['t41_inactive'], width=2),
                        name='비활성 장비',
                        hovertemplate='<b>%{x}</b><br><b>비활성</b>: %{y}대<br><b>전체</b>: %{customdata[0]}대<extra></extra>',
                        customdata=pivot_data[['total']]
                    ))
                
                fig.update_layout(
                    title=dict(text=f'시간별 장비 수 추이 ({filter_desc})', font=dict(size=14, color=THEME['text_primary'])),
                    xaxis_title='Time',
                    yaxis_title='장비 수 (Unique MAC)',
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"※ 장비 데이터가 있으면 '가동 중'으로 판단. 오렌지=가동 중, 회색=미가동(해당 시점 데이터 없음). 전체 장비: {total_devices}대")
                
                # 통계 요약
                col1, col2, col3, col4 = st.columns(4)
                
                active_col = 'active' if 'active' in pivot_data.columns else None
                inactive_col = 'inactive' if 'inactive' in pivot_data.columns else None
                
                if active_col:
                    with col1:
                        st.metric("최대 가동 장비", f"{int(pivot_data[active_col].max()):,}대", help="동시 가동 최대")
                    with col2:
                        st.metric("평균 가동 장비", f"{pivot_data[active_col].mean():.0f}대", help="5분 단위 평균")
                
                if inactive_col:
                    with col3:
                        st.metric("최대 미가동 장비", f"{int(pivot_data[inactive_col].max()):,}대", help="해당 시점 미감지")
                    with col4:
                        st.metric("총 장비 수", f"{total_devices:,}대", help="하루 1회 이상 감지된 Unique MAC")
                
                # ===== Spot 분석 (별도) =====
                st.markdown("---")
                st.markdown("#### 📍 Spot별 장비 분포")
                render_t31_spot_analysis(loader, t31_data)
                
            else:
                st.info("선택된 필터 조건에 해당하는 T31 데이터가 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_t31_spot_analysis(loader: CachedDataLoader, t31_data: pd.DataFrame):
    """T31 Spot별 장비 분포 분석"""
    try:
        if 'spot_nos' in t31_data.columns:
            # spot_nos는 콤마로 구분된 문자열일 수 있음
            # 먼저 고유 spot 목록 추출
            spot_data = []
            for _, row in t31_data.iterrows():
                spots = str(row['spot_nos']).split(',') if pd.notna(row['spot_nos']) else []
                for spot in spots:
                    spot = spot.strip()
                    if spot and spot != 'nan':
                        spot_data.append({
                            'spot_no': spot,
                            'mac_address': row['mac_address'],
                            'status': row['status'],
                            "risk_factors": "Risk Factors",
                            "reasoning": "DeepCon Analysis"
                        })
            
            if spot_data:
                spot_df = pd.DataFrame(spot_data)
                
                # Spot별 활성/비활성 집계
                spot_agg = spot_df.groupby(['spot_no', 'status']).agg({
                    'mac_address': 'nunique'
                }).reset_index()
                spot_agg.columns = ['spot_no', 'status', 'count']
                
                pivot_spot = spot_agg.pivot(index='spot_no', columns='status', values='count').fillna(0).reset_index()
                pivot_spot.columns.name = None
                rename_map = {0: 'inactive', 1: 'active'}
                pivot_spot = pivot_spot.rename(columns=rename_map)
                
                # Spot 이름 매핑 (가능한 경우)
                spot_names = loader.get_spot_names() if hasattr(loader, 'get_spot_names') else {}
                pivot_spot['spot_name'] = pivot_spot['spot_no'].apply(
                    lambda x: spot_names.get(int(x), f'Spot {x}') if str(x).isdigit() else x
                )
                
                fig = go.Figure()
                
                if 'active' in pivot_spot.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_spot['spot_name'],
                        y=pivot_spot['active'],
                        name='활성 장비',
                        marker_color=THEME['t31'],
                        hovertemplate='<b>Spot: %{x}</b><br>활성 장비: %{y:,}대<extra></extra>' # Fixed tooltip
                    ))
                
                if 'inactive' in pivot_spot.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_spot['spot_name'],
                        y=pivot_spot['inactive'],
                        name='비활성 장비',
                        marker_color='rgba(148, 163, 184, 0.7)',
                        hovertemplate='<b>Spot: %{x}</b><br>비활성 장비: %{y:,}대<extra></extra>' # Fixed tooltip
                    ))
                
                fig.update_layout(
                    title=dict(text='Spot별 장비 분포 (Unique MAC)', font=dict(size=14, color=THEME['text_primary'])),
                    barmode='stack',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='Spot', font=dict(color=THEME['text_secondary'])), tickangle=45),
                    yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='장비 수', font=dict(color=THEME['text_secondary']))),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 요약
                total_active = int(pivot_spot['active'].sum()) if 'active' in pivot_spot.columns else 0
                total_inactive = int(pivot_spot['inactive'].sum()) if 'inactive' in pivot_spot.columns else 0
                unique_devices = t31_data['mac_address'].nunique() if t31_data is not None else 0
                st.caption(f"""
                🔧 Spot 총 {len(pivot_spot)}개 | 활성: {total_active}대 | 비활성: {total_inactive}대  
                ⚠️ **주의**: 동일 장비가 여러 Spot에서 감지되면 중복 카운팅됩니다. (실제 Unique 장비: {unique_devices}대)
                """)
            else:
                st.info("Spot 데이터가 없습니다.")
        else:
            st.info("spot_nos 컬럼이 없습니다.")
    except Exception as e:
        st.error(f"Spot 분석 중 오류: {str(e)}")

def render_t31_statistics(loader: CachedDataLoader):
    """T31 상세 통계"""
    device_stats = loader.load_t31_device_stats()
    
    if device_stats is not None and not device_stats.empty:
        st.dataframe(device_stats, use_container_width=True, height=400)
        
        csv = device_stats.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv,
            file_name=f"t31_stats_{loader.date_str}.csv",
            mime="text/csv",
            key=f"dl_t31_stats_{loader.date_str}"
        )
    else:
        st.info("장비 통계 데이터가 없습니다.")

# ==================== T41 탭 ====================
def render_t41_tab(loader: CachedDataLoader):
    """T41 작업자 분석 탭"""
    st.markdown("""
    <div class="main-header">
        <h1>👷 T-Ward Type41</h1>
        <p>작업자 활동 분석</p>
    </div>
    """, unsafe_allow_html=True)
    
    sub_tabs = st.tabs(["📊 인원 현황", "⏱️ 체류 분석", "🏢 구역별 분석", "📈 활동 통계", "🗺️ Journey Heatmap", "📍 위치 분석"])
    
    with sub_tabs[0]:
        render_t41_occupancy_tab(loader)
    
    with sub_tabs[1]:
        render_t41_dwell(loader)
    
    with sub_tabs[2]:
        render_t41_building(loader)
    
    with sub_tabs[3]:
        render_t41_activity(loader)
    
    with sub_tabs[4]:
        render_t41_journey_heatmap(loader)
    
    with sub_tabs[5]:
        render_t41_location_analysis(loader)

def render_t41_occupancy_tab(loader: CachedDataLoader):
    """T41 인원 현황 (활성/비활성 구분) + Building별 분포"""
    # 새로운 통합 캐시 사용
    occupancy_data = loader.load_t41_time_series()
    
    if occupancy_data is not None and not occupancy_data.empty:
        # 시간별 차트
        st.markdown("#### 시간별 인원 현황")
        
        # time_label이 이미 있으면 그대로 사용
        if 'time_label' not in occupancy_data.columns:
            if 'time_index' in occupancy_data.columns:
                occupancy_data['time_label'] = occupancy_data['time_index'].apply(
                    lambda x: f"{(x-1)//12:02d}:{((x-1)%12)*5:02d}"
                )
            elif 'hour' in occupancy_data.columns:
                occupancy_data['time_label'] = occupancy_data['hour'].apply(lambda x: f"{x:02d}:00")
            else:
                occupancy_data['time_label'] = range(len(occupancy_data))
        
        fig = go.Figure()
        
        # Active/Inactive/Total 영역 차트
        if 'active_workers' in occupancy_data.columns:
            # Total 계산
            if 'inactive_workers' in occupancy_data.columns:
                occupancy_data = occupancy_data.copy()
                occupancy_data['total_workers'] = occupancy_data['active_workers'] + occupancy_data['inactive_workers']
            
            # hovertemplate용 customdata 구성 (Active, Inactive, Total)
            customdata = occupancy_data[['active_workers', 'inactive_workers', 'total_workers']].values if 'inactive_workers' in occupancy_data.columns else None
            
            # Active (아래쪽)
            fig.add_trace(go.Scatter(
                x=occupancy_data['time_label'],
                y=occupancy_data['active_workers'],
                fill='tozeroy',
                fillcolor=f"rgba(0, 200, 83, 0.6)",
                line=dict(color=THEME['t41_active'], width=2),
                name='Active',
                customdata=customdata,
                hovertemplate='<b>%{x}</b><br>Active: %{customdata[0]:,}명<br>Inactive: %{customdata[1]:,}명<br>Total: %{customdata[2]:,}명<extra></extra>' if customdata is not None else '<b>%{x}</b><br>Active: %{y}명<extra></extra>',
                showlegend=True
            ))
            
            if 'inactive_workers' in occupancy_data.columns:
                # Total (위쪽, Inactive 영역)
                fig.add_trace(go.Scatter(
                    x=occupancy_data['time_label'],
                    y=occupancy_data['total_workers'],
                    fill='tonexty',
                    fillcolor=f"rgba(148, 163, 184, 0.5)",
                    line=dict(color=THEME['t41_inactive'], width=2),
                    name='Inactive (영역)',
                    hoverinfo='skip',  # 첫 번째 trace에서 모든 정보 표시하므로 중복 방지
                    showlegend=True
                ))
        elif 'worker_count' in occupancy_data.columns:
            fig.add_trace(go.Scatter(
                x=occupancy_data['time_label'],
                y=occupancy_data['worker_count'],
                fill='tozeroy',
                fillcolor=f"rgba(0, 200, 83, 0.6)",
                line=dict(color=THEME['t41_active'], width=2),
                name='Workers',
                hovertemplate='<b>%{x}</b><br>Workers: %{y}명<extra></extra>' # Fixed tooltip
            ))
        
        fig.update_layout(
            title=dict(text='시간별 작업자 현황 (5분 단위, 초록=활성, 회색=비활성)', font=dict(size=14, color=THEME['text_primary'])),
            xaxis_title='Time (5분 단위)',
            yaxis_title='해당 시점 인원 수 (Unique MAC)',
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=THEME['text_primary']),
            xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
            yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("※ 각 시점(5분)에 데이터가 있는 Unique MAC 수. 초록=활성(움직임 감지), 회색=비활성(정지 상태). 위 그래프는 영역차트로, 회색 영역의 높이가 비활성 인원입니다.")
        
        # 통계 요약
        col1, col2, col3, col4 = st.columns(4)
        
        if 'active_workers' in occupancy_data.columns and 'inactive_workers' in occupancy_data.columns:
            with col1:
                st.metric("최대 활성 인원", f"{occupancy_data['active_workers'].max():,}명", help="동시 활성 인원 최대값")
            with col2:
                st.metric("평균 활성 인원", f"{occupancy_data['active_workers'].mean():.0f}명", help="5분 단위 평균")
            with col3:
                st.metric("최대 비활성 인원", f"{occupancy_data['inactive_workers'].max():,}명", help="동시 비활성 인원 최대값")
            with col4:
                st.metric("평균 비활성 인원", f"{occupancy_data['inactive_workers'].mean():.0f}명", help="5분 단위 평균")
        elif 'worker_count' in occupancy_data.columns:
            with col1:
                st.metric("최대 인원", f"{occupancy_data['worker_count'].max():,}명")
            with col2:
                st.metric("평균 인원", f"{occupancy_data['worker_count'].mean():.0f}명")
            with col3:
                st.metric("최소 인원", f"{occupancy_data['worker_count'].min():,}명")
            with col4:
                st.metric("데이터 포인트", f"{len(occupancy_data)}개")
        
        # Building별 작업자 분포 차트 추가
        st.markdown("---")
        st.markdown("#### Building별 작업자 분포")
        render_t41_building_distribution(loader)
        
    else:
        st.info("인원 데이터가 없습니다.")

def render_t41_building_distribution(loader: CachedDataLoader):
    """T41 Building별 작업자 분포 차트 (인원 현황 서브탭용)
    
    수정: 각 MAC의 '주 활동 상태'(가장 많이 나타난 상태)를 기준으로 집계
    - 동일 MAC이 활성/비활성 모두에 중복 카운트되는 문제 해결
    """
    try:
        df = get_flow_cache(loader)
        if not df.empty:
            t41_data = df[df['type'] == 41].copy()
            
            if not t41_data.empty:
                # 이름 매핑
                building_names = loader.get_building_names()
                
                t41_data['building_name'] = t41_data['building_no'].map(
                    lambda x: building_names.get(int(x), f'Building {x}') if pd.notna(x) else '알 수 없음'
                )
                
                # ========== 수정된 로직 ==========
                # 각 MAC의 '주 Building'과 '주 상태'를 결정 (가장 많이 나타난 값)
                # 1단계: 각 MAC별로 가장 많이 나타난 building 찾기
                mac_building = t41_data.groupby(['mac_address', 'building_name']).size().reset_index(name='count')
                mac_primary_building = mac_building.loc[mac_building.groupby('mac_address')['count'].idxmax()]
                
                # 2단계: 각 MAC별로 가장 많이 나타난 status 찾기
                mac_status = t41_data.groupby(['mac_address', 'status']).size().reset_index(name='count')
                mac_primary_status = mac_status.loc[mac_status.groupby('mac_address')['count'].idxmax()]
                
                # 3단계: MAC별 주 Building과 주 status 결합
                mac_summary = mac_primary_building[['mac_address', 'building_name']].merge(
                    mac_primary_status[['mac_address', 'status']], on='mac_address'
                )
                
                # 4단계: Building별 Active/Inactive 집계 (중복 없이)
                agg_data = mac_summary.groupby(['building_name', 'status']).size().reset_index(name='count')
                
                pivot_data = agg_data.pivot(index='building_name', columns='status', values='count').fillna(0).reset_index()
                rename_map = {0: 'inactive_count', 1: 'active_count'}
                pivot_data = pivot_data.rename(columns=rename_map)
                
                fig = go.Figure()
                
                if 'active_count' in pivot_data.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_data['building_name'],
                        y=pivot_data['active_count'],
                        name='활성 인원',
                        marker_color=THEME['t41_active'],
                        hovertemplate='<b>Building: %{x}</b><br>활성 인원: %{y:,}명<extra></extra>' # Fixed tooltip
                    ))
                
                if 'inactive_count' in pivot_data.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_data['building_name'],
                        y=pivot_data['inactive_count'],
                        name='비활성 인원',
                        marker_color=THEME['t41_inactive'],
                        hovertemplate='<b>Building: %{x}</b><br>비활성 인원: %{y:,}명<extra></extra>' # Fixed tooltip
                    ))
                
                fig.update_layout(
                    title=dict(text='Building별 주 활동 작업자 (하루 기준, MAC별 주 상태)', font=dict(size=14, color=THEME['text_primary'])),
                    barmode='stack',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='Building (주로 활동한 Building)', font=dict(color=THEME['text_secondary']))),
                    yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='인원 수 (Unique MAC)', font=dict(color=THEME['text_secondary']))),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 요약 - 명확한 설명 추가
                total_active = int(pivot_data['active_count'].sum()) if 'active_count' in pivot_data.columns else 0
                total_inactive = int(pivot_data['inactive_count'].sum()) if 'inactive_count' in pivot_data.columns else 0
                total_workers = t41_data['mac_address'].nunique()
                st.caption(f"""
                📊 **하루 전체 기준 (00:00~24:00)**  
                👷 총 작업자: **{total_workers:,}명** (Unique MAC)  
                🟢 주로 활성: {total_active:,}명 (50% 이상 활성 상태)  
                ⚪ 주로 비활성: {total_inactive:,}명 (50% 미만 활성 상태)  
                ※ 각 작업자는 가장 많이 활동한 Building에 1회만 카운트됨
                """)
                
                # ===== 시간대별 Building 분포 비교 =====
                st.markdown("---")
                st.markdown("#### 📊 시간대별 Building 분포 비교")
                render_t41_building_time_comparison(loader, t41_data)
            else:
                st.info("T41 데이터가 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")

def render_t41_building_time_comparison(loader: CachedDataLoader, t41_data: pd.DataFrame):
    """T41 시간대별 Building 분포 비교 (인원 현황 서브탭용)
    
    Note: 선택된 시간대 내 '누적 방문 인원'을 표시
    - 동일 MAC이 여러 Building에 나타날 수 있음 (누적 카운트)
    - Building별 합계 > 실제 인원 수 (정상)
    """
    try:
        if t41_data.empty:
            st.info("T41 데이터가 없습니다.")
            return
        
        building_names = loader.get_building_names()
        
        # 시간 선택지 (시작/종료)
        time_options = [bin_index_to_time_str(i) for i in range(288)]
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.selectbox(
                "시작 시간",
                time_options,
                index=84,  # 07:00 기본
                key='t41_building_start_time'
            )
        with col2:
            end_time = st.selectbox(
                "종료 시간",
                time_options,
                index=min(228, len(time_options)-1),  # 19:00 기본
                key='t41_building_end_time'
            )
        
        # time_index는 1-based (1~288), time_options는 0-based (0~287)
        start_idx = time_options.index(start_time) + 1
        end_idx = time_options.index(end_time) + 1
        
        if start_idx > end_idx:
            st.warning("시작 시간이 종료 시간보다 큽니다.")
        else:
            # 선택된 시간대 필터링 (start_idx부터 end_idx까지 포함)
            time_filtered = t41_data[(t41_data['time_index'] >= start_idx) & (t41_data['time_index'] <= end_idx)].copy()
            
            if not time_filtered.empty:
                time_filtered['building_name'] = time_filtered['building_no'].map(
                    lambda x: building_names.get(int(x), f'Building {x}') if pd.notna(x) else '알 수 없음'
                )
                
                # Building별 Active/Inactive 집계 (Unique MAC)
                agg_data = time_filtered.groupby(['building_name', 'status']).agg({
                    'mac_address': 'nunique'
                }).reset_index()
                agg_data.columns = ['building_name', 'status', 'count']
                
                pivot_data = agg_data.pivot(index='building_name', columns='status', values='count').fillna(0).reset_index()
                rename_map = {0: 'inactive', 1: 'active'}
                pivot_data = pivot_data.rename(columns=rename_map)
                
                fig = go.Figure()
                
                if 'active' in pivot_data.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_data['building_name'],
                        y=pivot_data['active'],
                        name='활성 인원',
                        marker_color=THEME['t41_active'],
                        hovertemplate='<b>Building: %{x}</b><br>활성 인원: %{y:,}명<extra></extra>' # Fixed tooltip
                    ))
                
                if 'inactive' in pivot_data.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_data['building_name'],
                        y=pivot_data['inactive'],
                        name='비활성 인원',
                        marker_color=THEME['t41_inactive'],
                        hovertemplate='<b>Building: %{x}</b><br>비활성 인원: %{y:,}명<extra></extra>' # Fixed tooltip
                    ))
                
                fig.update_layout(
                    title=dict(text=f'Building별 누적 방문 인원 ({start_time} ~ {end_time})', font=dict(size=14, color=THEME['text_primary'])),
                    barmode='stack',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='Building', font=dict(color=THEME['text_secondary']))),
                    yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='누적 방문 인원 (Unique MAC)', font=dict(color=THEME['text_secondary']))),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 전체 시간대 실제 Unique MAC 수
                total_unique_mac = time_filtered['mac_address'].nunique()
                
                # 요약 - 누적 방문 개념 명시
                total_active = int(pivot_data['active'].sum()) if 'active' in pivot_data.columns else 0
                total_inactive = int(pivot_data['inactive'].sum()) if 'inactive' in pivot_data.columns else 0
                st.caption(f"""
                📊 **{start_time} ~ {end_time} 누적 방문 통계**  
                👷 실제 인원: **{total_unique_mac:,}명** (Unique MAC)  
                🏢 Building별 합계: 활성 {total_active:,} + 비활성 {total_inactive:,} = **{total_active + total_inactive:,}** (동일 인원이 여러 Building 방문 시 중복 카운트)
                """)
            else:
                st.info("선택된 시간대에 데이터가 없습니다.")
    except Exception as e:
        st.error(f"Building 분포 분석 중 오류: {str(e)}")

def render_t41_dwell(loader: CachedDataLoader):
    """T41 체류 분석 (활성 상태 기준) - 위치 필터 적용"""
    # 메인 화면에 위치 필터 UI 표시 (Spot 제외)
    selected_building, selected_floor = render_location_filter(loader, 't41_dwell')
    
    st.info("💡 체류 시간은 **활성 상태(움직임 감지)** 시간만 계산합니다.")
    
    # 최소 체류시간 필터 추가 (기본값 15분)
    min_dwell_options = [0, 5, 10, 15, 30, 60, 120]
    min_dwell = st.selectbox(
        "최소 체류시간 필터 (분)",
        min_dwell_options,
        index=3,  # 기본값 15분 (index=3)
        help="비활성 상태의 태그가 외부 진동(주변 장비, 통행 등)에 의해 순간적으로 활성 감지될 수 있습니다. 이를 실제 작업으로 오인하지 않도록 최소 체류시간을 설정합니다.",
        key='t41_dwell_min_filter'
    )
    
    # 필터 적용 설명
    if min_dwell > 0:
        st.caption(f"⚠️ **{min_dwell}분 미만 활성 태그 제외**: 비활성 상태에서 외부 진동(주변 장비 가동, 사람 통행 등)에 의해 순간적으로 활성 감지된 태그로 추정하여 분석에서 제외합니다.")
    
    try:
        df = get_flow_cache(loader)
        if not df.empty:
            # T41 활성 상태만 (status=1)
            t41_data = df[(df['type'] == 41) & (df['status'] == 1)].copy()
            
            # 위치 필터 적용 (Spot 제외)
            t41_data = loader.filter_by_location(
                t41_data,
                selected_building,
                selected_floor,
                'All'  # Spot은 적용하지 않음
            )
            
            if not t41_data.empty:
                # MAC별 체류 시간 계산 (time_index 개수 * 5분)
                dwell_calc = t41_data.groupby('mac_address').agg({
                    'time_index': 'nunique'
                }).reset_index()
                dwell_calc.columns = ['mac_address', 'time_slots']
                dwell_calc['dwell_minutes'] = dwell_calc['time_slots'] * 5
                
                # 최소 체류시간 필터 적용
                total_with_active = len(dwell_calc)  # 활성 기록이 있는 태그 수
                short_dwell_count = len(dwell_calc[dwell_calc['dwell_minutes'] < min_dwell]) if min_dwell > 0 else 0
                
                if min_dwell > 0:
                    dwell_calc = dwell_calc[dwell_calc['dwell_minutes'] >= min_dwell].copy()
                
                # 필터 설명
                filter_parts = []
                if selected_building != 'All':
                    filter_parts.append(selected_building)
                if selected_floor != 'All':
                    filter_parts.append(selected_floor)
                filter_desc = ' > '.join(filter_parts) if filter_parts else '전체 구역'
                
                # 분석 대상 요약 (간결하게)
                st.markdown(f"""
                📌 **분석 대상**: 활성 기록이 있는 **{total_with_active:,}개** 태그 중 **{len(dwell_calc):,}개** ({len(dwell_calc)/total_with_active*100:.1f}%)  
                └ {min_dwell}분 미만 활성 태그 **{short_dwell_count:,}개** 제외 (외부 진동에 의한 순간 활성 추정)
                """)
                
                if dwell_calc.empty:
                    st.warning(f"최소 체류시간({min_dwell}분) 이상인 작업자가 없습니다.")
                    return
                
                # 체류 시간 구간별 분포 계산 (15분 기준이므로 15분~1시간 구간 추가)
                if min_dwell >= 15:
                    bins = [15, 60, 120, 240, 480, 720, 1440, float('inf')]
                    labels = ['15분~1시간', '1~2시간', '2~4시간', '4~8시간', '8~12시간', '12~24시간', '24시간+']
                else:
                    bins = [0, 60, 120, 240, 480, 720, 1440, float('inf')]
                    labels = ['1시간 미만', '1~2시간', '2~4시간', '4~8시간', '8~12시간', '12~24시간', '24시간+']
                dwell_calc['dwell_bin'] = pd.cut(dwell_calc['dwell_minutes'], bins=bins, labels=labels, right=False)
                
                # 통계 요약 (먼저 표시)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("분석 대상 작업자", f"{len(dwell_calc):,}명", help=f"최소 {min_dwell}분 이상 활성된 작업자")
                with col2:
                    avg_hours = dwell_calc['dwell_minutes'].mean() / 60
                    st.metric("평균 체류 시간", f"{avg_hours:.1f}시간")
                with col3:
                    median_hours = dwell_calc['dwell_minutes'].median() / 60
                    st.metric("중앙값 체류 시간", f"{median_hours:.1f}시간")
                with col4:
                    pct_8h_plus = len(dwell_calc[dwell_calc['dwell_minutes'] >= 480]) / len(dwell_calc) * 100
                    st.metric("8시간 이상 비율", f"{pct_8h_plus:.1f}%")
                
                # 구간별 분포 표시
                st.markdown("#### 체류 시간 구간별 분포")
                bin_counts = dwell_calc['dwell_bin'].value_counts().sort_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bin_counts.index.astype(str),
                    y=bin_counts.values,
                    marker_color=THEME['t41_active'],
                    text=bin_counts.values,
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>작업자 수: %{y:,}명<extra></extra>' # Fixed tooltip
                ))
                fig.update_layout(
                    title=dict(text=f'작업자 체류 시간 분포 ({filter_desc}) - 최소 {min_dwell}분 이상', font=dict(size=14, color=THEME['text_primary'])),
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='체류 시간 구간', font=dict(color=THEME['text_secondary']))),
                    yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='작업자 수', font=dict(color=THEME['text_secondary'])))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 상위 체류자 테이블
                st.markdown("#### 체류 시간 상위 작업자")
                top_dwellers = dwell_calc.nlargest(10, 'dwell_minutes')
                st.dataframe(top_dwellers, use_container_width=True)
                
                # ===== Spot별 체류 시간 분석 =====
                st.markdown("---")
                st.markdown("#### 📍 Spot별 체류 시간 분석")
                render_t41_dwell_by_spot(loader, t41_data, min_dwell)
                
            else:
                st.info("선택된 필터 조건에 해당하는 활성 인원이 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_t41_dwell_by_spot(loader: CachedDataLoader, t41_active: pd.DataFrame, default_min_dwell: int = 15):
    """T41 Spot별 체류 시간 분포"""
    try:
        if 'spot_nos' not in t41_active.columns:
            st.info("spot_nos 컨럼이 없습니다.")
            return
        
        # Spot 목록 추출
        all_spots = set()
        for spots_str in t41_active['spot_nos'].dropna():
            for spot in str(spots_str).split(','):
                spot = spot.strip()
                if spot and spot != 'nan':
                    all_spots.add(spot)
        
        if not all_spots:
            st.info("Spot 데이터가 없습니다.")
            return
        
        spot_list = sorted(list(all_spots), key=lambda x: int(x) if x.isdigit() else 0)
        
        # Spot 이름 매핑
        spot_names = loader.get_spot_names() if hasattr(loader, 'get_spot_names') else {}
        spot_options = ['All'] + [spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s for s in spot_list]
        spot_value_map = {'All': 'All'}
        for s in spot_list:
            name = spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s
            spot_value_map[name] = s
        
        # Spot 선택 및 최소 체류시간 필터 (같은 행에 배치)
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_spot_name = st.selectbox(
                "Spot 선택",
                spot_options,
                index=0,
                key='t41_dwell_spot'
            )
        with col2:
            min_dwell_options = [0, 5, 10, 15, 30, 60]
            # 기본값을 default_min_dwell로 설정 (상위 필터와 동일)
            default_idx = min_dwell_options.index(default_min_dwell) if default_min_dwell in min_dwell_options else 0
            min_dwell = st.selectbox(
                "최소 체류시간 (분)",
                min_dwell_options,
                index=default_idx,
                help="흡연장 등 짧은 체류가 예상되는 Spot은 0분 또는 5분으로 설정하세요.",
                key='t41_dwell_spot_min'
            )
        
        selected_spot = spot_value_map.get(selected_spot_name, 'All')
        
        # 선택된 Spot에 해당하는 데이터 필터링
        if selected_spot != 'All':
            spot_filtered = t41_active[t41_active['spot_nos'].str.contains(selected_spot, na=False)].copy()
        else:
            spot_filtered = t41_active.copy()
        
        if not spot_filtered.empty:
            # MAC별 체류 시간 계산
            dwell_spot = spot_filtered.groupby('mac_address').agg({
                'time_index': 'nunique'
            }).reset_index()
            dwell_spot.columns = ['mac_address', 'time_slots']
            dwell_spot['dwell_minutes'] = dwell_spot['time_slots'] * 5
            
            # 최소 체류시간 필터 적용
            total_before = len(dwell_spot)
            if min_dwell > 0:
                dwell_spot = dwell_spot[dwell_spot['dwell_minutes'] >= min_dwell].copy()
            filtered_count = total_before - len(dwell_spot)
            
            if dwell_spot.empty:
                st.info(f"선택된 Spot에 {min_dwell}분 이상 체류한 작업자가 없습니다.")
                return
            
            # 체류 시간 분포 히스토그램
            fig = px.histogram(
                dwell_spot,
                x='dwell_minutes',
                nbins=20,
                color_discrete_sequence=[THEME['t41_active']],
                labels={'dwell_minutes': '체류 시간 (분)', 'count': '작업자 수'}
            )
            fig.update_layout(
                title=dict(text=f'체류 시간 분포 ({selected_spot_name}) - 최소 {min_dwell}분 이상', font=dict(size=14, color=THEME['text_primary'])),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['text_primary']),
                xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
            )
            fig.update_traces(
                hovertemplate='<b>체류 시간: %{x}분</b><br>작업자 수: %{y:,}명<extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("분석 대상", f"{len(dwell_spot):,}명", help=f"{min_dwell}분 이상 체류자")
            with col2:
                st.metric("평균 체류", f"{dwell_spot['dwell_minutes'].mean():.0f}분")
            with col3:
                st.metric("최대 체류", f"{dwell_spot['dwell_minutes'].max()}분")
            with col4:
                st.metric("제외됨", f"{filtered_count:,}명", help=f"{min_dwell}분 미만 체류자")
        else:
            st.info("선택된 Spot에 해당하는 데이터가 없습니다.")
    except Exception as e:
        st.error(f"Spot 체류 분석 중 오류: {str(e)}")

def render_t41_building(loader: CachedDataLoader):
    """
    T41 구역별 분석 - 선택한 구역의 시간별 인원수 추이
    """
    try:
        # 메인 화면에 위치 필터 UI 표시 (Spot 제외)
        selected_building, selected_floor = render_location_filter(loader, 't41_zone')
        
        df = get_flow_cache(loader)
        if not df.empty:
            t41_data = df[df['type'] == 41].copy()
            
            # 필터 적용 (Spot 제외)
            t41_data = loader.filter_by_location(
                t41_data,
                selected_building,
                selected_floor,
                'All'  # Spot은 적용하지 않음
            )
            
            if not t41_data.empty:
                # time_index별 활성/비활성 인원 집계 (Unique MAC)
                time_agg = t41_data.groupby(['time_index', 'status']).agg({
                    'mac_address': 'nunique'
                }).reset_index()
                time_agg.columns = ['time_index', 'status', 'count']
                
                # pivot
                pivot_data = time_agg.pivot(index='time_index', columns='status', values='count').fillna(0).reset_index()
                pivot_data.columns.name = None
                
                # 컨럼 이름 정리
                rename_map = {0: 'inactive', 1: 'active'}
                pivot_data = pivot_data.rename(columns=rename_map)
                
                # 시간 레이블 생성 (time_index를 bin_index로 사용)
                pivot_data['time_label'] = pivot_data['time_index'].apply(bin_index_to_time_str)
                pivot_data = pivot_data.sort_values('time_index')
                
                # 영역 차트 (Active/Inactive 구분)
                fig = go.Figure()
                
                has_active = 'active' in pivot_data.columns
                has_inactive = 'inactive' in pivot_data.columns

                if has_active:
                    pivot_data['total'] = pivot_data['active'] + pivot_data.get('inactive', 0)
                    
                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['active'],
                        stackgroup='one',
                        fillcolor='rgba(16, 185, 129, 0.6)',
                        line=dict(color=THEME['t41_active'], width=2),
                        name='활성 인원',
                        hovertemplate='<b>%{x}</b><br>Active: %{y}명<br>Inactive: %{customdata[1]}명<br>Total: %{customdata[0]}명<extra></extra>',
                        customdata=pivot_data[['total', 'inactive']].fillna(0)
                    ))
                
                if has_inactive:
                    pivot_data['total'] = pivot_data.get('active', 0) + pivot_data['inactive']
                    
                    # If active trace exists, skip tooltip here to avoid duplication (unified mode)
                    # If active trace MISSING, show tooltip here
                    hover_settings = dict(hoverinfo='skip') if has_active else dict(
                        hovertemplate='<b>%{x}</b><br>Active: %{customdata[1]}명<br>Inactive: %{y}명<br>Total: %{customdata[0]}명<extra></extra>',
                        customdata=pivot_data[['total', 'active']].fillna(0)
                    )

                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['inactive'],
                        stackgroup='one',
                        fillcolor='rgba(148, 163, 184, 0.5)',
                        line=dict(color=THEME['t41_inactive'], width=2),
                        name='비활성 인원',
                        **hover_settings
                    ))
                
                # 필터 설명 생성
                filter_parts = []
                if selected_building != 'All':
                    filter_parts.append(selected_building)
                if selected_floor != 'All':
                    filter_parts.append(selected_floor)
                filter_desc = ' > '.join(filter_parts) if filter_parts else '전체 구역'
                
                fig.update_layout(
                    title=dict(text=f'시간별 인원수 추이 ({filter_desc})', font=dict(size=14, color=THEME['text_primary'])),
                    xaxis_title='Time',
                    yaxis_title='인원 수 (Unique MAC)',
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 요약
                col1, col2, col3, col4 = st.columns(4)
                
                active_col = 'active' if 'active' in pivot_data.columns else None
                inactive_col = 'inactive' if 'inactive' in pivot_data.columns else None
                
                if active_col:
                    with col1:
                        st.metric("최대 활성 인원", f"{int(pivot_data[active_col].max()):,}명")
                    with col2:
                        st.metric("평균 활성 인원", f"{pivot_data[active_col].mean():.0f}명")
                
                if inactive_col:
                    with col3:
                        st.metric("최대 비활성 인원", f"{int(pivot_data[inactive_col].max()):,}명")
                    with col4:
                        st.metric("평균 비활성 인원", f"{pivot_data[inactive_col].mean():.0f}명")
                
                # ===== Spot 분석 (별도) =====
                st.markdown("---")
                st.markdown("#### 📍 Spot별 작업자 분포")
                render_t41_spot_analysis(loader, t41_data)
                
            else:
                st.info("선택된 필터 조건에 해당하는 T41 데이터가 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_t41_spot_analysis(loader: CachedDataLoader, t41_data: pd.DataFrame):
    """T41 Spot별 작업자 분포 분석 - 개선된 버전"""
    try:
        if 'spot_nos' not in t41_data.columns:
            st.info("spot_nos 컬럼이 없습니다.")
            return
        
        # Spot 목록 추출
        all_spots = set()
        for spots_str in t41_data['spot_nos'].dropna():
            for spot in str(spots_str).split(','):
                spot = spot.strip()
                if spot and spot != 'nan':
                    all_spots.add(spot)
        
        if not all_spots:
            st.info("Spot 데이터가 없습니다.")
            return
        
        spot_list = sorted(list(all_spots), key=lambda x: int(x) if x.isdigit() else 0)
        
        # Spot 이름 매핑
        spot_names = loader.get_spot_names() if hasattr(loader, 'get_spot_names') else {}
        
        # ===== 1. Spot 선택 → 시간별 인원수 추이 =====
        st.markdown("##### 📈 Spot별 시간대 인원 추이")
        
        spot_options = [spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s for s in spot_list]
        spot_value_map = {}
        for s in spot_list:
            name = spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s
            spot_value_map[name] = s
        
        selected_spot_name = st.selectbox(
            "Spot 선택",
            spot_options,
            index=0,
            key='t41_spot_trend'
        )
        selected_spot = spot_value_map.get(selected_spot_name, spot_list[0])
        
        # 선택된 Spot의 시간별 인원 추이 (벡터화된 연산으로 성능 개선)
        # spot_nos 컨키 여부를 빠르게 확인
        spot_mask = t41_data['spot_nos'].fillna('').str.contains(rf'\b{selected_spot}\b', regex=True)
        spot_time_df = t41_data[spot_mask][['time_index', 'mac_address', 'status']].copy()
        
        if not spot_time_df.empty:
            
            # time_index별 활성/비활성 집계
            time_agg = spot_time_df.groupby(['time_index', 'status']).agg({
                'mac_address': 'nunique'
            }).reset_index()
            time_agg.columns = ['time_index', 'status', 'count']
            
            pivot_time = time_agg.pivot(index='time_index', columns='status', values='count').fillna(0).reset_index()
            pivot_time.columns.name = None
            pivot_time = pivot_time.rename(columns={0: 'inactive', 1: 'active'})
            pivot_time['time_label'] = pivot_time['time_index'].apply(bin_index_to_time_str)
            pivot_time = pivot_time.sort_values('time_index')
            
            # 영역 차트 (시간별 인원수 추이와 동일한 형태)
            fig = go.Figure()
            
            if 'active' in pivot_time.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_time['time_label'],
                    y=pivot_time['active'],
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.6)',
                    line=dict(color=THEME['t41_active'], width=2),
                    name='활성 인원',
                    hovertemplate='<b>%{x}</b><br>활성: %{y:,}명<extra></extra>' # Fixed tooltip
                ))
            
            if 'inactive' in pivot_time.columns:
                # 전체 = active + inactive
                if 'active' in pivot_time.columns:
                    total = pivot_time['active'] + pivot_time['inactive']
                else:
                    total = pivot_time['inactive']
                
                fig.add_trace(go.Scatter(
                    x=pivot_time['time_label'],
                    y=total,
                    fill='tonexty',
                    fillcolor='rgba(148, 163, 184, 0.5)',
                    line=dict(color=THEME['t41_inactive'], width=2),
                    name='비활성 인원',
                    hovertemplate='<b>%{x}</b><br>비활성: %{y:,}명<extra></extra>' # Fixed tooltip
                ))
            
            fig.update_layout(
                title=dict(text=f'{selected_spot_name} - 시간별 인원수 추이', font=dict(size=14, color=THEME['text_primary'])),
                xaxis_title='Time',
                yaxis_title='인원 수 (Unique MAC)',
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['text_primary']),
                xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 요약
            col1, col2, col3, col4 = st.columns(4)
            active_col = 'active' if 'active' in pivot_time.columns else None
            inactive_col = 'inactive' if 'inactive' in pivot_time.columns else None
            
            if active_col:
                with col1:
                    st.metric("최대 활성 인원", f"{int(pivot_time[active_col].max()):,}명")
                with col2:
                    st.metric("평균 활성 인원", f"{pivot_time[active_col].mean():.0f}명")
            
            if inactive_col:
                with col3:
                    st.metric("최대 비활성 인원", f"{int(pivot_time[inactive_col].max()):,}명")
                with col4:
                    st.metric("평균 비활성 인원", f"{pivot_time[inactive_col].mean():.0f}명")
        else:
            st.info("선택된 Spot에 데이터가 없습니다.")
        
        st.markdown("---")
        
        # ===== 2. 시간대 선택 → Spot별 분포 비교 =====
        st.markdown("##### 📊 시간대별 Spot 분포 비교")
        
        # 시간 선택지 (시작/종료)
        time_options = [bin_index_to_time_str(i) for i in range(288)]  # 0~287 (24시간 * 12)
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.selectbox(
                "시작 시간",
                time_options,
                index=0,
                key='t41_spot_start_time'
            )
        with col2:
            end_time = st.selectbox(
                "종료 시간",
                time_options,
                index=min(17, len(time_options)-1),  # 기본 01:25
                key='t41_spot_end_time'
            )
        
        # 시간 인덱스 변환
        start_idx = time_options.index(start_time)
        end_idx = time_options.index(end_time)
        
        if start_idx > end_idx:
            st.warning("시작 시간이 종료 시간보다 큽니다.")
        else:
            # 선택된 시간대 필터링
            time_filtered = t41_data[(t41_data['time_index'] >= start_idx) & (t41_data['time_index'] <= end_idx)].copy()
            
            if not time_filtered.empty:
                # Spot별 집계 (벡터화된 방식으로 성능 개선)
                # spot_nos를 분할하여 별도 행으로 확장
                time_filtered['spot_list'] = time_filtered['spot_nos'].fillna('').str.split(',')
                exploded = time_filtered.explode('spot_list')
                exploded['spot_no'] = exploded['spot_list'].str.strip()
                exploded = exploded[exploded['spot_no'].notna() & (exploded['spot_no'] != '') & (exploded['spot_no'] != 'nan')]
                
                if not exploded.empty:
                    spot_agg = exploded.groupby(['spot_no', 'status']).agg({
                        'mac_address': 'nunique'
                    }).reset_index()
                    spot_agg.columns = ['spot_no', 'status', 'count']
                    
                    pivot_spot = spot_agg.pivot(index='spot_no', columns='status', values='count').fillna(0).reset_index()
                    pivot_spot.columns.name = None
                    pivot_spot = pivot_spot.rename(columns={0: 'inactive', 1: 'active'})
                    
                    pivot_spot['spot_name'] = pivot_spot['spot_no'].apply(
                        lambda x: spot_names.get(int(x), f'Spot {x}') if str(x).isdigit() else x
                    )
                    
                    fig = go.Figure()
                    
                    if 'active' in pivot_spot.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_spot['spot_name'],
                            y=pivot_spot['active'],
                            name='활성 인원',
                            marker_color=THEME['t41_active'],
                            hovertemplate='<b>Spot: %{x}</b><br>활성 인원: %{y:,}명<extra></extra>' # Fixed tooltip
                        ))
                    
                    if 'inactive' in pivot_spot.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_spot['spot_name'],
                            y=pivot_spot['inactive'],
                            name='비활성 인원',
                            marker_color=THEME['t41_inactive'],
                            hovertemplate='<b>Spot: %{x}</b><br>비활성 인원: %{y:,}명<extra></extra>' # Fixed tooltip
                        ))
                    
                    fig.update_layout(
                        title=dict(text=f'Spot별 작업자 분포 ({start_time} ~ {end_time})', font=dict(size=14, color=THEME['text_primary'])),
                        barmode='stack',
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=THEME['text_primary']),
                        xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='Spot', font=dict(color=THEME['text_secondary'])), tickangle=45),
                        yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='인원 수', font=dict(color=THEME['text_secondary']))),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 요약
                    total_active = int(pivot_spot['active'].sum()) if 'active' in pivot_spot.columns else 0
                    total_inactive = int(pivot_spot['inactive'].sum()) if 'inactive' in pivot_spot.columns else 0
                    st.caption(f"👷 Spot 총 {len(pivot_spot)}개 | 활성: {total_active}명 | 비활성: {total_inactive}명")
                else:
                    st.info("선택된 시간대에 Spot 데이터가 없습니다.")
            else:
                st.info("선택된 시간대에 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"Spot 분석 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_t41_activity(loader: CachedDataLoader):
    """T41 활동 통계"""
    activity_data = loader.load_t41_activity_analysis()
    
    if activity_data is not None and not activity_data.empty:
        st.dataframe(activity_data.head(100), use_container_width=True, height=400)
        
        csv = activity_data.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv,
            file_name=f"t41_activity_{loader.date_str}.csv",
            mime="text/csv",
            key=f"dl_t41_activity_{loader.date_str}"
        )
    else:
        st.info("활동 분석 데이터가 없습니다.")

# ==================== T41 Journey Heatmap ====================
# Building-Floor 색상 매핑 (IRFM_demo_new 공간 구조 기반)
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

# 색상 팔레트 (인덱스별) - Floor별 톤 차이
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

# Building/Floor 번호를 Building-Floor 문자열로 매핑 (실제 데이터 기반)
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

def render_t41_journey_heatmap(loader: CachedDataLoader):
    """T41 Journey Heatmap - 작업자별 시간대별 위치 이동 패턴"""
    st.markdown("""
    <div class="dark-bg" style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h3 style="margin: 0;">🗺️ Journey Heatmap</h3>
        <p class="text-muted" style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            작업자별 시간대별 위치 이동 패턴을 시각화합니다
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    try:
        df = get_flow_cache(loader)
        if df.empty:
            st.warning("데이터를 불러올 수 없습니다.")
            return
        
        # T41 데이터만 필터링
        t41_data = df[df['type'] == 41].copy()
        
        if t41_data.empty:
            st.warning("T41 데이터가 없습니다.")
            return
        
        # 색상 범례 표시
        _render_journey_color_legend()
        
        # 필터링 옵션
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            max_workers = st.slider("표시할 작업자 수", min_value=50, max_value=500, value=200, step=50,
                                   help="성능을 위해 표시할 최대 작업자 수를 제한합니다", key="journey_max_workers")
        
        with col2:
            sort_option = st.selectbox("정렬 기준", 
                                       ["🤖 AI 추천 (활동 패턴)", "활동량 (많은 순)", "수신 시간 (긴 순)", "빌딩별 그룹화"],
                                       index=0,
                                       help="AI 추천: 활성 비율, 시간대 커버리지, 활성 횟수 조합", key="journey_sort_option")
        
        with col3:
            min_signals = st.slider("최소 활동 기록", min_value=1, max_value=100, value=10, step=5,
                                   help="이 값 이상의 기록이 있는 작업자만 표시", key="journey_min_signals")
        
        # Journey Heatmap 생성
        with st.spinner("Journey Heatmap 생성 중..."):
            heatmap_data = _generate_journey_heatmap(t41_data, max_workers, sort_option, min_signals, loader)
        
        if heatmap_data is not None:
            _display_journey_heatmap(heatmap_data, loader)
        else:
            st.warning("히트맵을 생성할 데이터가 부족합니다.")
            
    except Exception as e:
        st.error(f"Journey Heatmap 생성 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def _render_journey_color_legend():
    """Journey Heatmap 색상 범례 - Floor별 톤 차이 표시"""
    st.markdown("""
    <div class="dark-bg" style="background: rgba(30,41,59,0.95); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <div style="font-weight: 600; margin-bottom: 0.5rem;">🎨 색상 범례 (Building-Floor)</div>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px;">
            <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; background: #1a1a1a; border-radius: 4px; font-size: 0.7rem; border: 1px solid #333;">⬛ 미수신</span>
            <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; background: #6b7280; border-radius: 4px; font-size: 0.7rem;">⬜ 비활성</span>
            <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; background: #374151; border-radius: 4px; font-size: 0.7rem;">🏗️ 실외</span>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px;">
            <span class="text-light" style="font-size: 0.7rem; min-width: 35px;">WWT:</span>
            <span style="display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #86efac; border-radius: 4px; color: #1a1a1a !important; font-size: 0.7rem;">B1F</span>
            <span style="display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #22c55e; border-radius: 4px; font-size: 0.7rem;">1F</span>
            <span style="display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #15803d; border-radius: 4px; font-size: 0.7rem;">2F</span>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px;">
            <span class="text-light" style="font-size: 0.7rem; min-width: 35px;">CUB:</span>
            <span style="display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #7dd3fc; border-radius: 4px; color: #1a1a1a !important; font-size: 0.7rem;">1F</span>
            <span style="display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #0284c7; border-radius: 4px; font-size: 0.7rem;">B1F</span>
            <span class="text-light" style="font-size: 0.7rem; min-width: 35px; margin-left: 12px;">FAB:</span>
            <span style="display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #f97316; border-radius: 4px; font-size: 0.7rem;">1F</span>
            <span class="text-light" style="font-size: 0.7rem; min-width: 35px; margin-left: 12px;">WTP:</span>
            <span style="display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #fde047; border-radius: 4px; color: #1a1a1a !important; font-size: 0.7rem;">1F</span>
        </div>
        <div class="text-light" style="margin-top: 0.5rem; font-size: 0.7rem;">
            💡 <b>활성</b>: 헬멧 진동 감지 (작업 중) | <b>비활성</b>: 신호 수신되나 진동 없음 | <b>미수신</b>: 해당 시간대 신호 없음<br>
            🎨 같은 빌딩 내 층별로 <b>연한색→진한색</b> 톤으로 구분 (예: WWT B1F→1F→2F)
        </div>
    </div>
    """, unsafe_allow_html=True)

def _generate_journey_heatmap(t41_data: pd.DataFrame, max_workers: int, sort_option: str, min_signals: int, loader: CachedDataLoader):
    """Journey Heatmap 데이터 생성 (벡터화된 처리)
    
    핵심 로직:
    - Building/Floor는 시간대별로 중복되지 않음 (시스템 설계)
    - status=1: 활성 (해당 Building 색상)
    - status=0: 비활성 (회색)
    - 해당 시간대 데이터 없음: 미수신 (검정)
    """
    import numpy as np
    
    # =========================================================================
    # Step 1: 작업자별 통계 계산
    # =========================================================================
    worker_stats = t41_data.groupby('mac_address').agg({
        'time_index': ['count', 'nunique'],
        'building_no': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
        'status': ['sum', 'mean']  # 활성 합계 및 활성 비율
    }).reset_index()
    worker_stats.columns = ['mac', 'total_records', 'time_slots', 'primary_building', 'active_count', 'active_ratio']
    
    # 최소 기록 수 필터
    worker_stats = worker_stats[worker_stats['total_records'] >= min_signals]
    
    if worker_stats.empty:
        return None
    
    # =========================================================================
    # Step 2: 정렬 (AI 기반 포함)
    # =========================================================================
    if sort_option == "🤖 AI 추천 (활동 패턴)":
        # AI 추천: 건설현장 정상 작업시간 패턴 기반 정렬
        # - 정상 근무시간: 5~12시간 (60~144 time slots @ 5분 단위)
        # - 과다 활동(20시간+): 헬멧이 장비 근처 방치된 것으로 추정 → 페널티
        
        # 활성 시간 (시간 단위)
        worker_stats['active_hours'] = worker_stats['time_slots'] * 5 / 60  # 5분 단위 → 시간
        
        # 정상 작업시간 적합도 점수 (5~12시간에서 최대, 벗어나면 감소)
        def work_pattern_score(hours):
            if hours < 1:
                return 0.1  # 1시간 미만: 낮은 점수
            elif 5 <= hours <= 12:
                return 1.0  # 정상 범위: 최대 점수
            elif 12 < hours <= 15:
                return 0.7  # 연장 근무: 약간 감소
            elif 15 < hours <= 20:
                return 0.3  # 과다 근무: 크게 감소
            else:  # 20시간 초과
                return 0.05  # 비정상 (장비 방치 추정): 최저 점수
        
        worker_stats['pattern_score'] = worker_stats['active_hours'].apply(work_pattern_score)
        
        # AI 점수 = 패턴 적합도(50%) + 활성 비율(25%) + 활동량(25%)
        worker_stats['ai_score'] = (
            worker_stats['pattern_score'] * 0.5 +  # 정상 작업시간 패턴 가중치
            worker_stats['active_ratio'] * 0.25 +  # 활성 비율
            (worker_stats['active_count'] / worker_stats['active_count'].max()) * 0.25  # 활성 횟수 정규화
        )
        worker_stats = worker_stats.sort_values('ai_score', ascending=False)
    elif sort_option == "활동량 (많은 순)":
        worker_stats = worker_stats.sort_values('active_count', ascending=False)
    elif sort_option == "수신 시간 (긴 순)":
        worker_stats = worker_stats.sort_values('time_slots', ascending=False)
    else:  # 빌딩별 그룹화
        worker_stats = worker_stats.sort_values(['primary_building', 'active_count'], ascending=[True, False])
    
    # 상위 N명 선택
    worker_stats = worker_stats.head(max_workers)
    selected_macs = worker_stats['mac'].tolist()
    
    # =========================================================================
    # Step 3: 히트맵 매트릭스 생성 (벡터화)
    # =========================================================================
    num_bins = 288
    num_workers = len(selected_macs)
    
    # 초기화: 0 = 미수신 (검정)
    heatmap_matrix = np.zeros((num_workers, num_bins), dtype=int)
    
    # MAC → 인덱스 매핑
    mac_to_idx = {mac: i for i, mac in enumerate(selected_macs)}
    
    # 선택된 MAC의 데이터만 필터링
    filtered_data = t41_data[t41_data['mac_address'].isin(selected_macs)].copy()
    
    # Building 색상 매핑 함수
    def get_building_color(building_no, floor_no):
        """Building/Floor에 따른 색상 코드 반환"""
        bf_key = (int(building_no), int(floor_no))
        bf_name = BUILDING_FLOOR_MAP.get(bf_key, None)
        
        if bf_name and bf_name in JOURNEY_COLORS:
            return JOURNEY_COLORS[bf_name]
        
        # 빌딩 번호로 기본 색상 결정 (fallback)
        building_no = int(building_no)
        if building_no == 0:  # 실외
            return JOURNEY_COLORS.get('Outdoor', 2)
        elif building_no == 1:  # WWT
            return JOURNEY_COLORS.get('WWT-1F', 4)
        elif building_no == 2:  # FAB
            return JOURNEY_COLORS.get('FAB-1F', 6)
        elif building_no == 3:  # CUB
            return JOURNEY_COLORS.get('CUB-1F', 7)
        elif building_no == 4:  # WTP
            return JOURNEY_COLORS.get('WTP-1F', 9)
        else:
            return JOURNEY_COLORS.get('Outdoor', 2)
    
    # 각 레코드에 대해 히트맵 값 설정
    for _, row in filtered_data.iterrows():
        mac = row['mac_address']
        time_idx = int(row['time_index']) - 1  # 1-indexed → 0-indexed
        
        if mac not in mac_to_idx or time_idx < 0 or time_idx >= num_bins:
            continue
        
        worker_idx = mac_to_idx[mac]
        status = int(row.get('status', 0))
        building_no = row.get('building_no', 0)
        floor_no = row.get('floor_no', 0)
        
        if status == 1:  # 활성
            color_code = get_building_color(building_no, floor_no)
        else:  # 비활성 (status == 0)
            color_code = JOURNEY_COLORS['inactive']  # 회색
        
        # 같은 시간대에 여러 레코드가 있을 경우, 활성이 우선
        # (활성 색상 > 비활성 색상 > 미수신)
        if color_code > heatmap_matrix[worker_idx, time_idx]:
            heatmap_matrix[worker_idx, time_idx] = color_code
    
    return {
        'matrix': heatmap_matrix,
        'macs': selected_macs,
        'worker_stats': worker_stats
    }

def _display_journey_heatmap(heatmap_data: dict, loader: CachedDataLoader):
    """Journey Heatmap 시각화"""
    import plotly.graph_objects as go
    import numpy as np
    
    matrix = heatmap_data['matrix']
    macs = heatmap_data['macs']
    worker_stats = heatmap_data['worker_stats']
    
    num_workers = len(macs)
    num_bins = matrix.shape[1]
    
    # 통계 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("표시 작업자", f"{num_workers:,}명")
    with col2:
        active_cells = np.sum(matrix > 1)  # 비활성(1) 이상
        st.metric("활성 시간대", f"{active_cells:,}개")
    with col3:
        coverage = (np.sum(matrix > 0) / (num_workers * num_bins) * 100)
        st.metric("수신 커버리지", f"{coverage:.1f}%")
    with col4:
        active_coverage = (active_cells / (num_workers * num_bins) * 100)
        st.metric("활성 커버리지", f"{active_coverage:.1f}%")
    
    # 시간 레이블 생성 (5분 단위)
    time_labels = [f"{(i*5)//60:02d}:{(i*5)%60:02d}" for i in range(num_bins)]
    
    # Y축 레이블 (MAC 주소 축약)
    y_labels = [f"{mac[:8]}..." if len(mac) > 8 else mac for mac in macs]
    
    # 색상 스케일 생성
    num_colors = len(JOURNEY_COLOR_PALETTE)
    colorscale = [[i/(num_colors-1), JOURNEY_COLOR_PALETTE[i]] for i in range(num_colors)]
    
    # 위치 코드 → 이름 매핑 (hover용) - JOURNEY_COLORS와 동기화
    LOCATION_NAMES = {
        0: '미수신',
        1: '비활성',
        2: '실외',
        3: 'WWT-B1F',
        4: 'WWT-1F',
        5: 'WWT-2F',
        6: 'FAB-1F',
        7: 'CUB-1F',
        8: 'CUB-B1F',
        9: 'WTP-1F',
    }
    
    # customdata로 위치 이름 매핑
    location_names_matrix = np.vectorize(lambda x: LOCATION_NAMES.get(x, f'Unknown({x})'))(matrix)
    
    # Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=time_labels,
        y=y_labels,
        customdata=location_names_matrix,
        colorscale=colorscale,
        zmin=0,
        zmax=num_colors - 1,
        showscale=False,
        hovertemplate='<b>시간</b>: %{x}<br><b>작업자</b>: %{y}<br><b>위치</b>: %{customdata}<extra></extra>' # Fixed tooltip
    ))
    
    # 레이아웃 설정
    row_height = 8
    min_height = 400
    max_height = 2000
    calculated_height = num_workers * row_height + 100
    chart_height = max(min_height, min(max_height, calculated_height))
    
    fig.update_layout(
        title=dict(
            text=f'Journey Heatmap ({num_workers}명의 작업자, 5분 단위)',
            font=dict(size=16, color='white')
        ),
        height=chart_height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            title='시간',
            tickangle=45,
            dtick=12,  # 1시간 간격
            tickfont=dict(size=10, color='#9ca3af'),
            showgrid=False
        ),
        yaxis=dict(
            title='작업자',
            tickfont=dict(size=8, color='#9ca3af'),
            showgrid=False,
            dtick=max(1, num_workers // 20)
        ),
        margin=dict(l=100, r=20, t=50, b=80)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 통계 접기
    with st.expander("📊 작업자 상세 통계", expanded=False):
        display_stats = worker_stats.copy()
        # 컬럼명 변환 (ai_score가 있을 수 있음)
        if 'ai_score' in display_stats.columns:
            display_stats = display_stats.rename(columns={
                'mac': 'MAC',
                'total_records': '총 기록',
                'time_slots': '활동 시간대',
                'primary_building': '주요 빌딩',
                'active_count': '활성 횟수',
                'active_ratio': '활성 비율',
                'ai_score': 'AI 점수'
            })
            display_stats['활성 비율'] = display_stats['활성 비율'].apply(lambda x: f"{x*100:.1f}%")
            display_stats['AI 점수'] = display_stats['AI 점수'].apply(lambda x: f"{x:.3f}")
        else:
            display_stats = display_stats.rename(columns={
                'mac': 'MAC',
                'total_records': '총 기록',
                'time_slots': '활동 시간대',
                'primary_building': '주요 빌딩',
                'active_count': '활성 횟수',
                'active_ratio': '활성 비율'
            })
            display_stats['활성 비율'] = display_stats['활성 비율'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(display_stats, use_container_width=True, height=300)

@st.fragment
def render_t41_location_analysis(loader: CachedDataLoader):
    """T41 위치 분석 - Sector Map + Floor Map (동기화된 애니메이션)"""
    
    # 상단 레이아웃: 제목 (좌) + 선택 메뉴 (우)
    col_header_L, col_header_R = st.columns([1, 1])
    
    with col_header_L:
        st.markdown("#### 📍 실시간 위치 분석 (Synchronized)")
        
    with col_header_R:
        # 빌딩/층 선택 (가로 배치)
        buildings, floors_by_building = load_floor_map_options()
        
        if not buildings:
            st.warning("Floor map 데이터가 없습니다.")
            return
        
        # Session state 초기화
        if 'selected_building_name' not in st.session_state:
            st.session_state.selected_building_name = buildings[0]
        
        c1, c2 = st.columns(2)
        with c1:
            selected_building_name = st.selectbox(
                "Building", 
                buildings, 
                index=buildings.index(st.session_state.selected_building_name) if st.session_state.selected_building_name in buildings else 0,
                key="building_select"
            )
        
        # 선택된 빌딩의 층 옵션들
        available_floors = floors_by_building.get(selected_building_name, [])
        floor_names = [f['name'] for f in available_floors]
        
        if 'selected_floor_name' not in st.session_state:
            st.session_state.selected_floor_name = floor_names[0] if floor_names else ""
            
        with c2:
            if floor_names:
                selected_floor_name = st.selectbox(
                    "Floor", 
                    floor_names,
                    index=floor_names.index(st.session_state.selected_floor_name) if st.session_state.selected_floor_name in floor_names else 0,
                    key="floor_select"
                )
            else:
                st.warning("층 데이터 없음")
                return

    # 선택된 층의 building_no와 floor_no 찾기
    selected_floor_info = next((f for f in available_floors if f['name'] == selected_floor_name), None)
    if not selected_floor_info:
        st.error("층 정보를 찾을 수 없습니다.")
        return
    
    building_no = selected_floor_info['building_no']
    floor_no = selected_floor_info['floor_no']
    
    # Session state 업데이트
    st.session_state.selected_building_name = selected_building_name
    st.session_state.selected_floor_name = selected_floor_name
    
    try:
                # 통합 지도 생성 (캐시 사용으로 빠름)
                cache_path = str(loader.cache_folder.parent) if loader._is_new_structure else str(loader.cache_folder)
                fig = _create_synchronized_map(building_no, floor_no, cache_path, loader.date_str)

                # If fig is a dict (pre-serialized), render via HTML to avoid Streamlit JSON parsing issues
                if isinstance(fig, dict):
                        try:
                                div_id = f"plotly_sync_{building_no}_{floor_no}"
                                payload = _json.dumps(fig)
                                # base64 encode to safely embed in HTML/JS without breaking </script>
                                b64 = _b64.b64encode(payload.encode('utf-8')).decode('ascii')
                                html = f"""
<div id=\"{div_id}\" style=\"width:100%;height:920px;\"></div>
<script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
<script>
    try {{
        const txt = atob('{b64}');
        const fig = JSON.parse(txt);
        Plotly.newPlot('{div_id}', fig.data || [], fig.layout || {{}} , {{responsive: true}})
            .then(() => {{
                 try {{
                     if (fig.frames) {{
                         Plotly.addFrames('{div_id}', fig.frames);
                     }}
                 }} catch(frErr) {{
                     console.error('addFrames error', frErr);
                 }}
            }});
    }} catch (e) {{
        const pre = document.createElement('pre');
        pre.textContent = 'Plotly render error: ' + e.toString();
        document.getElementById('{div_id}').appendChild(pre);
    }}
</script>
"""
                                components.html(html, height=920)
                        except Exception:
                                # fallback
                                st.plotly_chart(fig, use_container_width=True, key="synchronized_map")
                else:
                        st.plotly_chart(fig, use_container_width=True, key="synchronized_map")
        
    except Exception as e:
        st.error(f"위치 분석 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


@st.cache_data(ttl=3600, show_spinner="🗺️ 지도 생성 중...")
def _create_animated_sector_map_fast(cache_path: str, date_str: str) -> go.Figure:
    """Plotly 내장 애니메이션 - 클라이언트 측 재생 (스크롤 없음, 빠른 반응)"""
    
    # --- 3. 데이터 로딩 (Optimized Split Loading) ---
    # Load separate caches (Outdoor only)
    outdoor_cache = load_split_location_cache(cache_path, date_str, 0)
    
    if not outdoor_cache:
        st.warning("위치 데이터 캐시가 없습니다")
        return go.Figure()
    
    # --- 4. 애니메이션 프레임 생성 ---
    frames = []
    slider_steps = []
    
    # Get base shapes and annotations for the map
    shapes, base_annotations = _get_background_shapes_cached()
    
    # Get gateway data
    outdoor_gw = load_outdoor_gateway_cached()
    gw_x = outdoor_gw['location_x'].tolist() if not outdoor_gw.empty else []
    gw_y = outdoor_gw['location_y'].tolist() if not outdoor_gw.empty else []
    
    # Building info for bubbles
    buildings_pos = {1: (358, 854), 2: (347, 673), 3: (929, 870), 4: (747, 835)}
    building_names = {1: 'FAB', 2: 'CUB', 3: 'WWT', 4: 'Office'}
    
    # Prepare first-frame data
    first_data = outdoor_cache.get('1', {})
    first_active = first_data.get('active', [])
    first_inactive = first_data.get('inactive', [])

    # Trail & dedupe helpers for smoother motion and correct counts
    TRAIL_LENGTH = 3
    def _dedupe_coords(coords):
        """중복 좌표 제거 - 0.1m 정밀도 (너무 aggressive하지 않게)"""
        seen = set()
        out = []
        for p in coords or []:
            try:
                # 0.1m 정밀도로 변경 (기존 0.001m는 너무 aggressive)
                key = (round(float(p[0]), 1), round(float(p[1]), 1))
            except Exception:
                continue
            if key not in seen:
                seen.add(key)
                out.append([key[0], key[1]])
        return out

    # 5분 단위 24시간 (288개)
    # iterate each 5-min bin (time_idx 1..288). Use bin_index_to_time_str(time_idx-1)
    for time_idx in range(1, 289):
        cache_key = str(time_idx)
        frame_time_str = bin_index_to_time_str(time_idx - 1)
        time_str = frame_time_str
        frame_data = outdoor_cache.get(cache_key, {})

        # New split cache keys: 'active', 'inactive'
        active = frame_data.get('active', [])
        inactive = frame_data.get('inactive', [])
        building_counts = frame_data.get('building_counts', {})
        outdoor_total = frame_data.get('total', 0)
        # Indoor total can be derived from building counts
        indoor_total = sum(building_counts.values()) if building_counts else 0

        # build previous-frame aggregated trail
        prev_keys = [str(k) for k in range(time_idx - 1, time_idx - TRAIL_LENGTH - 1, -1) if k >= 1]
        prev_active = []
        prev_inactive = []
        for pk in prev_keys:
            po = outdoor_cache.get(pk, {})
            if po:
                prev_active.extend(po.get('active', []) or [])
                prev_inactive.extend(po.get('inactive', []) or [])

        # helper: detect whether entries include mac (new cache format)
        def _extract_by_mac(items):
            mac_map = {}
            coord_list = []
            mac_mode = False
            for it in items or []:
                if isinstance(it, dict):
                    mac_mode = True
                    mac = it.get('mac') or it.get('mac_address')
                    try:
                        x = float(it.get('x', 0))
                        y = float(it.get('y', 0))
                    except Exception:
                        continue
                    if mac is None:
                        # fallback to coord list
                        coord_list.append([x, y])
                    else:
                        mac_map[str(mac)] = (x, y)
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    try:
                        coord_list.append([float(it[0]), float(it[1])])
                    except Exception:
                        continue
            return mac_mode, mac_map, coord_list

        prev_mac_mode, prev_mac_map, prev_coord_list = _extract_by_mac(prev_active)
        cur_mac_mode, cur_mac_map, cur_coord_list = _extract_by_mac(active)
        in_mac_mode, in_mac_map, in_coord_list = _extract_by_mac(inactive)

        # Build plotted lists: prefer per-mac mapping when available to ensure 1 marker per device
        if cur_mac_mode:
            sec_x = [v[0] for v in cur_mac_map.values()]
            sec_y = [v[1] for v in cur_mac_map.values()]
            active_cnt_plot = len(cur_mac_map)
        else:
            dedup = _dedupe_coords(cur_coord_list)
            sec_x = [p[0] for p in dedup] if dedup else []
            sec_y = [p[1] for p in dedup] if dedup else []
            active_cnt_plot = len(dedup)

        if in_mac_mode:
            in_x = [v[0] for v in in_mac_map.values()]
            in_y = [v[1] for v in in_mac_map.values()]
            in_cnt_plot = len(in_mac_map)
        else:
            dedup_i = _dedupe_coords(in_coord_list)
            in_x = [p[0] for p in dedup_i] if dedup_i else []
            in_y = [p[1] for p in dedup_i] if dedup_i else []
            in_cnt_plot = len(dedup_i)

        # previous trail
        if prev_mac_mode:
            trail_x = [v[0] for v in prev_mac_map.values()]
            trail_y = [v[1] for v in prev_mac_map.values()]
        else:
            dedup_prev = _dedupe_coords(prev_coord_list)
            trail_x = [p[0] for p in dedup_prev] if dedup_prev else []
            trail_y = [p[1] for p in dedup_prev] if dedup_prev else []

        # apply deterministic jitter for display-only when plotting multiple devices at identical coords
        if cur_mac_mode and sec_x:
            sec_x, sec_y = _deterministic_jitter(sec_x, sec_y, scale=0.35)
        if in_mac_mode and in_x:
            in_x, in_y = _deterministic_jitter(in_x, in_y, scale=0.25)
        if prev_mac_mode and trail_x:
            trail_x, trail_y = _deterministic_jitter(trail_x, trail_y, scale=0.15)

        frame_traces = [
            go.Scatter(x=trail_x, y=trail_y, mode='markers', marker=dict(size=5, color='#93C5FD', opacity=0.35), name='Trail', hoverinfo='skip'),
            go.Scatter(x=sec_x, y=sec_y, mode='markers', marker=dict(size=6, color='#3B82F6', opacity=0.9), name='활성', hoverinfo='skip'),
            go.Scatter(x=in_x, y=in_y, mode='markers', marker=dict(size=4, color='#9CA3AF', opacity=0.6), name='비활성', hoverinfo='skip'),
            go.Scatter(x=gw_x, y=gw_y, mode='markers', marker=dict(size=7, color='#DC2626', symbol='square', opacity=0.9), name='Gateway', hoverinfo='skip')
        ]
        
        # 빌딩 인원수 + 통계 annotations
        frame_annotations = list(base_annotations)
        
        # 빌딩별 인원수
        for bno, (cx, cy) in buildings_pos.items():
            count = building_counts.get(str(bno), building_counts.get(bno, 0))
            frame_annotations.append(dict(
                x=cx, y=cy - 30,
                text=f"<b>{count}</b>",
                showarrow=False,
                font=dict(size=12, color='#1E40AF'),
                bgcolor='rgba(255,255,255,0.9)',
                borderpad=2
            ))
        
        # 상단 통계 annotation
        fab_cnt = building_counts.get('1', building_counts.get(1, 0))
        cub_cnt = building_counts.get('2', building_counts.get(2, 0))
        wwt_cnt = building_counts.get('3', building_counts.get(3, 0))
        office_cnt = building_counts.get('4', building_counts.get(4, 0))
        # use plotted counts (per-device mac mapping when available) so stats match markers
        active_cnt = active_cnt_plot if 'active_cnt_plot' in locals() else 0
        inactive_cnt = in_cnt_plot if 'in_cnt_plot' in locals() else 0

        # stats_text now reflects plotted marker counts (deduped)
        stats_text = f"⏰ {time_str}  │  🌳 실외(플롯): {active_cnt} (비활성 {inactive_cnt})  🏢 실내(총): {indoor_total}  │  FAB {fab_cnt}  CUB {cub_cnt}  WWT {wwt_cnt}  Office {office_cnt}"
        
        frame_annotations.append(dict(
            x=0.5, y=1.08,
            xref='paper', yref='paper',
            text=f"<b>{stats_text}</b>",
            showarrow=False,
            font=dict(size=11, color='#111827'),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#E5E7EB',
            borderwidth=1,
            borderpad=6
        ))
        
        frame_layout = go.Layout(
            annotations=frame_annotations,
            title=dict(text=f"실시간 위치 분석 (Synchronized) <br><sub>Time: {frame_time_str} | Active: {active_cnt + indoor_total}명</sub>",
                       font=dict(size=14, color='#111827'))
        )
        
        frames.append(go.Frame(data=frame_traces, name=cache_key, layout=frame_layout))
        
        slider_steps.append(dict(
            args=[[cache_key], dict(frame=dict(duration=100, redraw=True), mode="immediate", transition=dict(duration=0))],
            label=frame_time_str,
            method="animate"
        ))
    
    # 첫 프레임 데이터: prefer per-mac plotting when cache contains macs
    def _build_plot_lists(items):
        mac_map = {}
        coord_list = []
        mac_mode = False
        for it in items or []:
            if isinstance(it, dict):
                mac_mode = True
                mac = it.get('mac') or it.get('mac_address')
                try:
                    x = float(it.get('x', 0))
                    y = float(it.get('y', 0))
                except Exception:
                    continue
                if mac is None:
                    coord_list.append([x, y])
                else:
                    mac_map[str(mac)] = (x, y)
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                try:
                    coord_list.append([float(it[0]), float(it[1])])
                except Exception:
                    continue
        if mac_mode:
            xs = [v[0] for v in mac_map.values()]
            ys = [v[1] for v in mac_map.values()]
            count_active = len(mac_map)
        else:
            dedup = _dedupe_coords(coord_list)
            xs = [p[0] for p in dedup] if dedup else []
            ys = [p[1] for p in dedup] if dedup else []
            count_active = len(dedup)
        return xs, ys, count_active

    f_x, f_y, f_active_cnt = _build_plot_lists(first_active)
    fi_x, fi_y, f_inactive_cnt = _build_plot_lists(first_inactive)

    fig = go.Figure(
        data=[
            go.Scatter(x=f_x, y=f_y, mode='markers', marker=dict(size=6, color='#3B82F6', opacity=0.8), name='활성'),
            go.Scatter(x=fi_x, y=fi_y, mode='markers', marker=dict(size=4, color='#9CA3AF', opacity=0.5), name='비활성'),
            go.Scatter(x=gw_x, y=gw_y, mode='markers', marker=dict(size=7, color='#DC2626', symbol='square', opacity=0.9), name='Gateway')
        ],
        frames=frames
    )

    # 첫 프레임 통계
    first_bc = first_data.get('building_counts', {})
    first_outdoor = first_data.get('outdoor_total', 0)
    first_indoor = first_data.get('indoor_total', 0)
    first_stats = f"⏰ 00:00  │  🌳 실외: {first_outdoor} (활성 {f_active_cnt} / 비활성 {f_inactive_cnt})  🏢 실내: {first_indoor}  │  FAB {first_bc.get('1', 0)}  CUB {first_bc.get('2', 0)}  WWT {first_bc.get('3', 0)}  Office {first_bc.get('4', 0)}"
    
    first_annotations = list(base_annotations)
    for bno, (cx, cy) in buildings_pos.items():
        count = first_bc.get(str(bno), first_bc.get(bno, 0))
        first_annotations.append(dict(
            x=cx, y=cy - 30,
            text=f"<b>{count}</b>",
            showarrow=False,
            font=dict(size=12, color='#1E40AF'),
            bgcolor='rgba(255,255,255,0.9)',
            borderpad=2
        ))
    first_annotations.append(dict(
        x=0.5, y=1.08,
        xref='paper', yref='paper',
        text=f"<b>{first_stats}</b>",
        showarrow=False,
        font=dict(size=11, color='#111827'),
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='#E5E7EB',
        borderwidth=1,
        borderpad=6
    ))
    
    # 레이아웃
    fig.update_layout(
        xaxis=dict(range=[-20, 1263], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-20, 1112], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', fixedrange=True),
        height=720,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1, font=dict(size=10, color='#111827')),
        shapes=shapes,
        annotations=first_annotations,
        margin=dict(l=10, r=10, t=80, b=80),
        dragmode=False,
        # 재생/정지 버튼
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=-0.05,
            x=0.0,
            xanchor='left',
            yanchor='top',
            pad=dict(t=0, r=10),
            buttons=[
                dict(label='▶️ 재생', method='animate',
                     args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True, mode='immediate')]),
                dict(label='⏸️ 정지', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        )],
        # 시간 슬라이더
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(font=dict(size=12, color='#111827'), prefix='', visible=True, xanchor='center'),
            len=0.85,
            x=0.15,
            y=-0.02,
            pad=dict(t=30, b=10),
            steps=slider_steps,
            tickcolor='#9CA3AF',
            font=dict(color='#374151')
        )]
    )
    
    # sanitize figure for JSON (replace inf/NaN and numpy scalars)
    try:
        _clean_figure_for_json(fig)
    except Exception:
        pass

    # Convert to plain dict via Plotly JSON to avoid Streamlit/client-side JSON parsing issues
    try:
        # Build a pure-python dict from the figure and deeply sanitize it
        fig_dict_raw = fig.to_dict() if hasattr(fig, 'to_dict') else None
        if fig_dict_raw is None:
            import plotly.io as _pio, json as _json
            fig_json = _pio.to_json(fig)
            fig_dict_raw = _json.loads(fig_json)
        fig_dict = _deep_sanitize(fig_dict_raw)
        # debug write for outdoor view
        try:
            if int(building_no) == 0 and int(floor_no) == 0:
                import json as _json
                with open('tmp_synchronized_payload.json', 'w', encoding='utf-8') as _f:
                    _f.write(_json.dumps(fig_dict))
        except Exception:
            pass
        return fig_dict
    except Exception:
        return fig


@st.cache_data(ttl=3600, show_spinner="애니메이션 생성 중...")
def _create_animated_sector_map(location_cache: dict, start_idx: int, end_idx: int, step: int = 3) -> go.Figure:
    """Plotly 애니메이션 기반 Sector Map (클라이언트 측 재생, 경량화)"""
    
    # 배경 shapes 가져오기
    shapes, base_annotations = _get_background_shapes_cached()
    
    # 게이트웨이 데이터
    outdoor_gw = load_outdoor_gateway_cached()
    
    # 빌딩 정보
    buildings_pos = {1: (358, 854), 2: (347, 673), 3: (929, 870), 4: (747, 835)}
    
    # 프레임 데이터 수집 (step 간격으로 샘플링하여 경량화)
    frames = []
    slider_steps = []
    
    for idx in range(start_idx, end_idx + 1, step):
        frame_data = location_cache.get(str(idx), {})
        
        # 시간 문자열
        h = ((idx - 1) * 5) // 60
        m = ((idx - 1) * 5) % 60
        time_str = f"{h:02d}:{m:02d}"
        
        # 활성/비활성 위치
        active = frame_data.get('outdoor_active', [])
        inactive = frame_data.get('outdoor_inactive', [])
        building_counts = frame_data.get('building_counts', {})
        
        # 프레임 데이터
        frame_traces = []
        
        # 활성 마커
        if active:
            frame_traces.append(go.Scattergl(
                x=_sanitize_data([p[0] for p in active]),
                y=_sanitize_data([p[1] for p in active]),
                mode='markers',
                marker=dict(size=6, color='#3B82F6', opacity=0.8),
                name='활성',
                hoverinfo='skip'
            ))
        else:
            frame_traces.append(go.Scattergl(x=[], y=[], mode='markers', name='활성'))
        
        # 비활성 마커
        if inactive:
            frame_traces.append(go.Scattergl(
                x=_sanitize_data([p[0] for p in inactive]),
                y=_sanitize_data([p[1] for p in inactive]),
                mode='markers',
                marker=dict(size=4, color='#9CA3AF', opacity=0.5),
                name='비활성',
                hoverinfo='skip'
            ))
        else:
            frame_traces.append(go.Scattergl(x=[], y=[], mode='markers', name='비활성'))
        
        # 게이트웨이 (고정)
        if not outdoor_gw.empty:
            frame_traces.append(go.Scattergl(
                x=_sanitize_data(outdoor_gw['location_x']),
                y=_sanitize_data(outdoor_gw['location_y']),
                mode='markers',
                marker=dict(size=7, color='#DC2626', symbol='square', opacity=0.9),
                name='Gateway',
                hoverinfo='skip'
            ))
        
        # 빌딩 인원수 annotations
        frame_annotations = list(base_annotations)
        for bno, (cx, cy) in buildings_pos.items():
            count = building_counts.get(str(bno), building_counts.get(bno, 0))
            frame_annotations.append(dict(
                x=cx, y=cy - 30,
                text=f"<b>{count}</b>",
                showarrow=False,
                font=dict(size=12, color='#1E40AF'),
                bgcolor='rgba(255,255,255,0.9)',
                borderpad=2
            ))
        
        # 프레임 추가
        frames.append(go.Frame(
            data=frame_traces,
            name=str(idx),
            layout=go.Layout(
                title=dict(text=f"📍 Sector Map | {time_str}", font=dict(size=14)),
                annotations=frame_annotations
            )
        ))
        
        # 슬라이더 스텝
        slider_steps.append(dict(
            args=[[str(idx)], dict(frame=dict(duration=300, redraw=True), mode='immediate')],
            label=time_str,
            method='animate'
        ))
    
    # 첫 프레임 데이터로 Figure 생성
    first_data = location_cache.get(str(start_idx), {})
    first_active = first_data.get('outdoor_active', [])
    first_inactive = first_data.get('outdoor_inactive', [])
    
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=[p[0] for p in first_active] if first_active else [],
                y=[p[1] for p in first_active] if first_active else [],
                mode='markers',
                marker=dict(size=6, color='#3B82F6', opacity=0.8),
                name='활성'
            ),
            go.Scattergl(
                x=[p[0] for p in first_inactive] if first_inactive else [],
                y=[p[1] for p in first_inactive] if first_inactive else [],
                mode='markers',
                marker=dict(size=4, color='#9CA3AF', opacity=0.5),
                name='비활성'
            ),
            go.Scattergl(
                x=outdoor_gw['location_x'].tolist() if not outdoor_gw.empty else [],
                y=outdoor_gw['location_y'].tolist() if not outdoor_gw.empty else [],
                mode='markers',
                marker=dict(size=7, color='#DC2626', symbol='square', opacity=0.9),
                name='Gateway'
            )
        ],
        frames=frames
    )
    
    # 레이아웃
    h = ((start_idx - 1) * 5) // 60
    m = ((start_idx - 1) * 5) % 60
    
    fig.update_layout(
        title=dict(text=f"📍 Sector Map | {h:02d}:{m:02d}", font=dict(size=14)),
        xaxis=dict(range=[-20, 1263], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-20, 1112], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', fixedrange=True),
        height=650,
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10)),
        shapes=shapes,
        annotations=base_annotations,
        margin=dict(l=10, r=10, t=50, b=80),
        # 애니메이션 컨트롤
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=0,
            x=0.1,
            xanchor='right',
            yanchor='top',
            buttons=[
                dict(label='▶️ 재생', method='animate',
                     args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True, mode='immediate')]),
                dict(label='⏸️ 정지', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(font=dict(size=12), prefix='시간: ', visible=True, xanchor='right'),
            len=0.9,
            x=0.1,
            y=0,
            steps=slider_steps
        )]
    )
    
    return fig


# 전역 캐시: 배경 shapes (Spot 폴리곤 + 빌딩 + Spot 이름)
@st.cache_data(ttl=3600, show_spinner=False)
def _get_background_shapes_cached() -> tuple:
    """배경 shapes를 한 번만 계산하여 캐시 (Spot 폴리곤 + 빌딩 + Spot 이름)"""
    spot_df, spot_pos_df = load_spot_data_cached()
    
    if spot_df.empty:
        return [], []
    
    # 실외 spot 필터링
    outdoor_spots = spot_df[spot_df['floor_no'].isna()].copy()
    
    # 빌딩 정의
    buildings = {
        'WWT': {'no': 3, 'x1': 880, 'y1': 793, 'x2': 978, 'y2': 947, 'color': 'rgba(34,197,94,0.2)', 'border': 'rgba(34,197,94,0.8)'},
        'FAB': {'no': 1, 'x1': 187, 'y1': 754, 'x2': 530, 'y2': 954, 'color': 'rgba(249,115,22,0.2)', 'border': 'rgba(249,115,22,0.8)'},
        'CUB': {'no': 2, 'x1': 225, 'y1': 626, 'x2': 470, 'y2': 721, 'color': 'rgba(59,130,246,0.2)', 'border': 'rgba(59,130,246,0.8)'},
        'Office': {'no': 4, 'x1': 682, 'y1': 753, 'x2': 812, 'y2': 917, 'color': 'rgba(168,85,247,0.2)', 'border': 'rgba(168,85,247,0.8)'}
    }
    
    shapes = []
    spot_labels = []  # Spot 이름 annotations
    
    # Spot 폴리곤을 shapes로 (trace 대신)
    spot_colors = {
        'constructionSite': 'rgba(200,200,200,0.08)',
        'restSpace': 'rgba(16,185,129,0.15)',
        'innerTarget': 'rgba(59,130,246,0.15)',
        'parkingLot': 'rgba(107,114,128,0.15)',
        'etc': 'rgba(156,163,175,0.1)'
    }
    
    for _, spot in outdoor_spots.iterrows():
        spot_no = spot['spot_no']
        spot_name = spot.get('name', '')
        div_type = spot['div'] if pd.notna(spot.get('div')) else 'etc'
        
        spot_coords = spot_pos_df[spot_pos_df['spot_no'] == spot_no].sort_values('point_no')
        if spot_coords.empty:
            continue
        
        # SVG path 생성
        x_list = spot_coords['x'].tolist()
        y_list = spot_coords['y'].tolist()
        
        if len(x_list) >= 3:
            path = f"M {x_list[0]} {y_list[0]}"
            for i in range(1, len(x_list)):
                path += f" L {x_list[i]} {y_list[i]}"
            path += " Z"
            
            shapes.append(dict(
                type="path",
                path=path,
                fillcolor=spot_colors.get(div_type, spot_colors['etc']),
                line=dict(width=0),
                layer="below"
            ))
            
            # Spot 이름 annotation 추가 (이름이 있는 경우만)
            if spot_name and pd.notna(spot_name) and str(spot_name).strip():
                center_x = sum(x_list) / len(x_list)
                center_y = sum(y_list) / len(y_list)
                spot_labels.append(dict(
                    x=center_x, y=center_y,
                    text=str(spot_name).strip(),
                    showarrow=False,
                    font=dict(size=9, color='#374151'),
                    bgcolor='rgba(255,255,255,0.7)',
                    borderpad=2
                ))
    
    # 빌딩 사각형
    for name, coords in buildings.items():
        shapes.append(dict(
            type="rect",
            x0=coords['x1'], y0=coords['y1'],
            x1=coords['x2'], y1=coords['y2'],
            fillcolor=coords['color'],
            line=dict(color=coords['border'], width=2)
        ))
    
    # 빌딩 annotations
    annotations = list(spot_labels)  # Spot 이름부터 추가
    for name, coords in buildings.items():
        center_x = (coords['x1'] + coords['x2']) / 2
        center_y = (coords['y1'] + coords['y2']) / 2
        annotations.append(dict(
            x=center_x, y=center_y + 25,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=12, color='#333'),
            bgcolor='rgba(255,255,255,0.9)',
            borderpad=3
        ))
    
    return shapes, annotations


def _create_sector_map_ultrafast(frame_data: dict, time_index: int) -> go.Figure:
    """초고속 Sector Map - 배경 캐시 + Scattergl 사용"""
    
    # 캐시된 배경 shapes 가져오기
    shapes, base_annotations = _get_background_shapes_cached()
    
    # Sector 크기
    sector_width = 1243
    sector_height = 1092
    
    # Figure 생성 (빈 그림)
    fig = go.Figure()
    
    # T41 활성 위치 (Scattergl - WebGL)
    outdoor_active = frame_data.get('outdoor_active', [])
    if outdoor_active:
        fig.add_trace(go.Scattergl(
            x=[p[0] for p in outdoor_active],
            y=[p[1] for p in outdoor_active],
            mode='markers',
            marker=dict(size=7, color='#3B82F6', opacity=0.8),
            name=f'활성 ({len(outdoor_active)})',
            hoverinfo='skip'
        ))
    
    # T41 비활성 위치 (Scattergl)
    outdoor_inactive = frame_data.get('outdoor_inactive', [])
    if outdoor_inactive:
        fig.add_trace(go.Scattergl(
            x=[p[0] for p in outdoor_inactive],
            y=[p[1] for p in outdoor_inactive],
            mode='markers',
            marker=dict(size=5, color='#9CA3AF', opacity=0.5),
            name=f'비활성 ({len(outdoor_inactive)})',
            hoverinfo='skip'
        ))
    
    # 게이트웨이 (Scattergl)
    outdoor_gw = load_outdoor_gateway_cached()
    if not outdoor_gw.empty:
        fig.add_trace(go.Scattergl(
            x=outdoor_gw['location_x'].tolist(),
            y=outdoor_gw['location_y'].tolist(),
            mode='markers',
            marker=dict(size=8, color='#DC2626', symbol='square', opacity=0.9),
            name=f'Gateway ({len(outdoor_gw)})',
            hoverinfo='skip'
        ))
    
    # 빌딩별 인원수 annotations
    building_counts = frame_data.get('building_counts', {})
    buildings_pos = {
        1: (358, 854), 2: (347, 673), 3: (929, 870), 4: (747, 835)
    }
    
    dynamic_annotations = list(base_annotations)  # 복사
    for bno, (cx, cy) in buildings_pos.items():
        count = building_counts.get(str(bno), building_counts.get(bno, 0))
        dynamic_annotations.append(dict(
            x=cx, y=cy - 30,
            text=f"<b>{count}명</b>",
            showarrow=False,
            font=dict(size=14, color='#1E40AF'),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#3B82F6',
            borderwidth=1,
            borderpad=4
        ))
    
    # 시간 표시
    hours = ((time_index - 1) * 5) // 60
    minutes = ((time_index - 1) * 5) % 60
    
    # 레이아웃 (고정값, 흰 배경 + 검정 글씨)
    fig.update_layout(
        title=dict(text=f"📍 Sector Map | {hours:02d}:{minutes:02d}", font=dict(size=14, color='#111827')),
        xaxis=dict(range=[-20, sector_width + 20], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-20, sector_height + 20], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', fixedrange=True),
        height=700,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10, color='#111827')),
        shapes=shapes,
        annotations=dynamic_annotations,
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode=False  # 드래그 비활성화로 속도 향상
    )
    
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def _get_floor_map_shapes_cached(building_no: int, floor_no: int) -> dict:
    """Floor Map의 배경 Shapes(Floor Rect + Spot Polygons) 및 Annotations 생성"""
    
    # 1. 캐시된 사전 생성 Map (JSON) 로드 시도
    cache_data = load_floor_map_cache(building_no, floor_no)
    if 'figure_json' in cache_data and cache_data['figure_json']:
        import json
        fig_dict = json.loads(cache_data['figure_json'])
        layout = fig_dict.get('layout', {})
        
        # Extract polygons from data traces
        polygons = []
        for trace in fig_dict.get('data', []):
            if trace.get('fill') == 'toself':
                polygons.append({
                    'x': trace.get('x', []),
                    'y': trace.get('y', []),
                    'fillcolor': trace.get('fillcolor', 'rgba(50, 50, 50, 0.3)'),
                    'line_color': trace.get('line', {}).get('color', '#666'),
                    'name': trace.get('name', '')
                })

        return {
            'shapes': layout.get('shapes', []),
            'annotations': layout.get('annotations', []),
            'polygons': polygons, 
            'length_x': cache_data.get('length_x', 100),
            'length_y': cache_data.get('length_y', 100),
            'floor_name': cache_data.get('floor_name', cache_data.get('name', f"Floor {floor_no}"))
        }

    # Floor 정보 로드 (Legacy generation)
    floor_info = load_floor_info_cached()
    floor_data = floor_info[(floor_info['building_number'] == building_no) & 
                            (floor_info['floor_number'] == floor_no)]
    
    if floor_data.empty:
        return {'shapes': [], 'annotations': [], 'polygons': [], 'length_x': 100, 'length_y': 100, 'floor_name': 'Unknown'}
    
    length_x = floor_data.iloc[0]['length_x']
    length_y = floor_data.iloc[0]['length_y']
    floor_name = floor_data.iloc[0]['floor_name']
    
    shapes = []
    annotations = []
    
    # Floor 사각형 (배경) - 흰색 배경이므로 테두리만
    shapes.append(dict(
        type="rect",
        x0=0, y0=0, x1=length_x, y1=length_y,
        line=dict(color="black", width=1),
        layer="below"
    ))
    
    # Spot 다각형 추가
    spot_info = load_spot_info_cached()
    spot_position = load_spot_position_cached()
    
    floor_spots = spot_info[spot_info['floor_no'] == floor_no]
    
    polygons = [] # 별도 trace로 추가하기 위해 저장
    
    for _, spot in floor_spots.iterrows():
        spot_no = spot['spot_no']
        spot_name = spot['name']
        
        # Spot 위치 데이터
        positions = spot_position[spot_position['spot_no'] == spot_no].sort_values('point_no')
        
        if not positions.empty:
            x_coords = positions['x'].tolist()
            y_coords = positions['y'].tolist()
            # 닫힌 다각형
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])
            
            # 사용자 요청: 짙은 회색 반투명
            fill_color = 'rgba(50, 50, 50, 0.3)'
            line_color = 'rgba(50, 50, 50, 0.5)'
            
            polygons.append({
                'x': x_coords,
                'y': y_coords,
                'fillcolor': fill_color,
                'line_color': line_color,
                'name': spot_name
            })
            
            # Spot 이름 Annotation
            center_x = sum(x_coords[:-1]) / len(x_coords[:-1])
            center_y = sum(y_coords[:-1]) / len(y_coords[:-1])
            
            annotations.append(dict(
                x=center_x, y=center_y,
                text=spot_name,
                showarrow=False,
                font=dict(size=10, color="#333"),
                # bgcolor="rgba(255, 255, 255, 0.7)" # 텍스트 배경 제거하여 깔끔하게
            ))
            
    return {
        'shapes': shapes, 
        'annotations': annotations, 
        'polygons': polygons,
        'length_x': length_x, 
        'length_y': length_y,
        'floor_name': floor_name
    }


@st.cache_data(ttl=3600, show_spinner="📍 통합 지도 생성 중...")
def _create_synchronized_map(building_no: int, floor_no: int, cache_path: str, date_str: str) -> go.Figure:
    """Sector Map과 Floor Map을 1x2 서브플롯으로 통합하여 동기화된 애니메이션 생성"""
    
    # 1. 데이터 로드 (Optimized Split Loading)
    sector_shapes, sector_annotations = _get_background_shapes_cached()
    floor_bg = _get_floor_map_shapes_cached(building_no, floor_no)
    
    # Load separate caches
    outdoor_cache = load_split_location_cache(cache_path, date_str, 0)
    indoor_cache = load_split_location_cache(cache_path, date_str, building_no, floor_no)
    
    # 빌딩 정보 (Sector Map용)
    buildings_pos = {1: (358, 854), 2: (347, 673), 3: (929, 870), 4: (747, 835)}
    
    # 2. Subplots 생성
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3], # 70:30 Adjustment
        subplot_titles=("", ""), # Titles handled by annotations for better positioning
        horizontal_spacing=0.03,
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Static Titles (Fixed)
    # Sector Map Title
    fig.add_annotation(dict(
        x=0.35, y=1.08, xref='paper', yref='paper',
        text=f"<b>🏭 Sector Map (Outdoor)</b>",
        showarrow=False,
        font=dict(size=16, color="black"),
        xanchor='center'
    ))
    # Floor Map Title
    # Prefer explicit building+floor selection (WWT - B1F) when available in session
    try:
        bname = st.session_state.get('selected_building_name', None)
        fname = st.session_state.get('selected_floor_name', None)
        if bname and fname:
            display_floor = f"{bname} - {fname}"
        else:
            display_floor = floor_bg.get('floor_name', '')
    except Exception:
        display_floor = floor_bg.get('floor_name', '')

    fig.add_annotation(dict(
        x=0.85, y=1.08, xref='paper', yref='paper',
        text=f"<b>🏢 Floor Map: {display_floor} (Indoor)</b>",
        showarrow=False,
        font=dict(size=16, color="black"),
        xanchor='center'
    ))
    
    # 3. 배경 Shapes 추가 (Static)
    # [Left] Sector Map
    for shape in sector_shapes:
        fig.add_shape(shape, row=1, col=1)
        
    for ann in sector_annotations: # 기본 Annotation (빌딩 이름 등)
        fig.add_annotation(ann, row=1, col=1)    

    # [Right] Floor Map
    for shape in floor_bg['shapes']:
        fig.add_shape(shape, row=1, col=2)
        
    for ann in floor_bg['annotations']:
        fig.add_annotation(ann, row=1, col=2)
        
    # Floor Map Polygons (Trace)
    for poly in floor_bg['polygons']:
        fig.add_trace(go.Scatter(
            x=poly['x'], y=poly['y'],
            fill='toself',
            fillcolor=poly['fillcolor'],
            line=dict(color=poly['line_color'], width=1),
            mode='lines',
            name=poly['name'],
            hoverinfo='name',
            showlegend=False
        ), row=1, col=2)
        
    # [Indoor Gateway] Floor Map Gateways (Static)
    indoor_gw = load_indoor_gateway_cached(building_no, floor_no)
    if not indoor_gw.empty:
        fig.add_trace(go.Scatter(
            x=_sanitize_data(indoor_gw['location_x']), 
            y=_sanitize_data(indoor_gw['location_y']),
            mode='markers+text',
            marker=dict(size=8, color='#EF4444', symbol='diamond', opacity=0.9, line=dict(width=1, color='white')),
            name='Indoor Gateway', hoverinfo='text',
            text=indoor_gw['name'],
            textposition='top center',
            textfont=dict(size=9, color='red'),
            showlegend=True
        ), row=1, col=2)

    # [Gateway] Sector Map Gateways (Static)
    outdoor_gw = load_outdoor_gateway_cached()
    if not outdoor_gw.empty:
        fig.add_trace(go.Scatter(
            x=_sanitize_data(outdoor_gw['location_x']), 
            y=_sanitize_data(outdoor_gw['location_y']),
            mode='markers', # 텍스트 제거하여 깔끔하게
            marker=dict(size=7, color='#DC2626', symbol='square', opacity=0.9),
            name='Gateway', hoverinfo='text', 
            text=outdoor_gw['name'],
            showlegend=True
        ), row=1, col=1)

    # 4. 애니메이션 프레임 생성
    frames = []
    slider_steps = []
    
    # 애니메이션 간격 설정 최적화: 5분 대신 10분 단위로 변경하여 프레임 수 절반으로 감소
    # 288개 → 144개 프레임 (로딩 시간 대폭 단축)
    step_size = 2  # 10분 단위 (원래 1 = 5분 단위)
    indices = range(0, 288, step_size)
    
    # Placeholder Traces for Animation (순서 중요)
    # [Trail] Sector previous-frame trail (faded) to create smoother motion
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=5, color='#93C5FD', opacity=0.35), name='Outdoor Trail'), row=1, col=1)
    # [Left] Sector Active
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=6, color='#3B82F6', opacity=0.9), name='Outdoor Active'), row=1, col=1)
    # [Left] Sector Inactive
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=4, color='#9CA3AF', opacity=0.6), name='Outdoor Inactive'), row=1, col=1)

    # [Trail] Floor previous-frame trail (faded)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=6, color='#93C5FD', opacity=0.35), name='Indoor Trail'), row=1, col=2)
    # [Right] Floor Active (파란 동그라미)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=8, color='#3B82F6', opacity=0.95, line=dict(width=1, color='white')), name='Indoor Active'), row=1, col=2)
    # [Right] Floor Inactive (회색 동그라미)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=6, color='#9CA3AF', opacity=0.6), name='Indoor Inactive'), row=1, col=2)

    trace_indices = [len(fig.data)-6, len(fig.data)-5, len(fig.data)-4, len(fig.data)-3, len(fig.data)-2, len(fig.data)-1]

    # --- Initial State Stats (For Start Display) ---
    base_annotations_clean = list(fig.layout.annotations) # Capture BEFORE adding dynamic stats
    
    if indices:
        first_idx = indices[0]
        f_cache_key = str(first_idx + 1)
        
        # Data for first frame
        f_out = outdoor_cache.get(f_cache_key, {})
        f_in = indoor_cache.get(f_cache_key, {})
        
        def _count_items(items):
            if not items:
                return 0
            if isinstance(items[0], dict):
                # count unique macs
                return len({str(it.get('mac') or it.get('mac_address')) for it in items if it.get('mac') or it.get('mac_address')})
            # legacy list of coords
            return len(items)

        f_o_act = _count_items(f_out.get('active', []))
        f_o_inact = _count_items(f_out.get('inactive', []))
        f_i_act = _count_items(f_in.get('active', []))
        f_i_inact = _count_items(f_in.get('inactive', []))
        
        f_h = ((first_idx * 5) // 60)
        f_m = ((first_idx * 5) % 60)
        f_time_lbl = f"{f_h:02d}:{f_m:02d}"
        
        # Initial Stats Calc
        f_build_active_counts = f_out.get('building_active_counts', {})
        f_total_indoor = sum([int(v) for v in f_build_active_counts.values()])
        f_sector_total_active = f_o_act + f_total_indoor
        
        # Add Initial Annotations to Main Figure (so they show on load)
        fig.add_annotation(dict(
             x=0.55, y=1.05, xref='paper', yref='paper',
             text=f"<b>{f_time_lbl}</b>", showarrow=False, font=dict(size=20, color="black"), xanchor='center'
        ))
        fig.add_annotation(dict(
            x=0.35, y=1.02, xref='paper', yref='paper', 
            text=f"전체: {f_sector_total_active+f_o_inact}명 | 활성: <span style='color:blue'>{f_sector_total_active}명</span> | 비활성: <span style='color:gray'>{f_o_inact}명</span> | 실외활성: {f_o_act}명",
            showarrow=False, font=dict(size=14, color="black"), xanchor='center'
        ))
        fig.add_annotation(dict(
            x=0.85, y=1.02, xref='paper', yref='paper', 
            text=f"전체: {f_i_act+f_i_inact}명 | 활성: <span style='color:blue'>{f_i_act}명</span> | 비활성: <span style='color:gray'>{f_i_inact}명</span>",
            showarrow=False, font=dict(size=14, color="black"), xanchor='center'
        ))

    for idx in indices:
        cache_key = str(idx + 1)
        prev_cache_key = str(idx) if idx > 0 else None
        
        # --- Sector Map Data (Outdoor) ---
        out_data = outdoor_cache.get(cache_key, {})
        sec_active = out_data.get('active', [])
        sec_inactive = out_data.get('inactive', [])
        building_counts_map = out_data.get('building_counts', {})
        
        # --- Floor Map Data (Indoor) ---
        in_data = indoor_cache.get(cache_key, {})
        floor_act = in_data.get('active', [])
        floor_inact = in_data.get('inactive', [])

        # previous-frame data for trail
        prev_out = outdoor_cache.get(prev_cache_key, {}) if prev_cache_key else {}
        prev_in = indoor_cache.get(prev_cache_key, {}) if prev_cache_key else {}
        prev_sec_active = prev_out.get('active', []) if prev_out else []
        prev_floor_active = prev_in.get('active', []) if prev_in else []

        # 프레임 생성
        # [Left] Building Counts Annotation 생성
        current_annotations = []
        for bno, (cx, cy) in buildings_pos.items():
            count = building_counts_map.get(str(bno), building_counts_map.get(bno, 0))
            current_annotations.append(dict(
                x=cx, y=cy - 30,
                text=f"<b>{count}</b>",
                showarrow=False,
                font=dict(size=12, color='#1E40AF'),
                bgcolor='rgba(255,255,255,0.9)',
                borderpad=2,
                xref="x1", yref="y1" # subplot 1
            ))
            
        # 프레임 추가
        # 슬라이더 스텝 및 타임 라벨 (Move up)
        hours = ((idx * 5) // 60)
        minutes = ((idx * 5) % 60)
        time_label = f"{hours:02d}:{minutes:02d}"

        # Stats (use plotted counts when possible)
        o_act = sec_cnt if 'sec_cnt' in locals() else (len(sec_active) if sec_active else 0)
        i_act = floor_cnt if 'floor_cnt' in locals() else (len(floor_act) if floor_act else 0)
        
        # Sector Map "Active" = Outdoor Active + All Buildings Indoor Active (using new active counts)
        # building_active_counts contains counts for each building. Sum them up for Total Indoor Active.
        b_active_counts = out_data.get('building_active_counts', {})
        total_indoor_active = sum([int(v) for v in b_active_counts.values()])
        sector_total_active = o_act + total_indoor_active
        
        # Dynamic Title
        title_text = f"실시간 위치 분석 (Synchronized) <br><sub>Time: {time_label} | Active: {sector_total_active}명 ({o_act} / {i_act})</sub>"

        # 프레임 추가
        # prepare jittered plotting coords (deterministic) and count per-device
        def _prep_items(points, scale=0.6):
            if not points:
                return [], [], 0
            # dict entries with macs
            if isinstance(points[0], dict):
                mac_map = {}
                for it in points:
                    mac = it.get('mac') or it.get('mac_address')
                    try:
                        x = float(it.get('x', 0))
                        y = float(it.get('y', 0))
                    except Exception:
                        continue
                    if mac is not None:
                        mac_map[str(mac)] = (x, y)
                xs = [v[0] for v in mac_map.values()]
                ys = [v[1] for v in mac_map.values()]
                if xs:
                    jx, jy = _deterministic_jitter(xs, ys, scale=scale)
                    return _sanitize_data(jx), _sanitize_data(jy), len(mac_map)
                return [], [], 0
            # legacy list of coords
            try:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ded = _dedupe_coords(points)
                if ded:
                    dx = [p[0] for p in ded]
                    dy = [p[1] for p in ded]
                    jx, jy = _deterministic_jitter(dx, dy, scale=scale)
                    return _sanitize_data(jx), _sanitize_data(jy), len(ded)
            except Exception:
                pass
            return [], [], 0

        # jitter scale을 크게 증가하여 겹치는 작업자들이 보이도록 수정
        p_prev_sec_x, p_prev_sec_y, _ = _prep_items(prev_sec_active, scale=0.5)
        sec_x, sec_y, sec_cnt = _prep_items(sec_active, scale=1.5)  # 0.35 → 1.5
        sec_inact_x, sec_inact_y, sec_inact_cnt = _prep_items(sec_inactive, scale=1.0)  # 0.25 → 1.0
        p_prev_floor_x, p_prev_floor_y, _ = _prep_items(prev_floor_active, scale=0.5)
        floor_x, floor_y, floor_cnt = _prep_items(floor_act, scale=1.5)  # 0.35 → 1.5
        floor_inact_x, floor_inact_y, floor_inact_cnt = _prep_items(floor_inact, scale=1.0)  # 0.25 → 1.0

        frames.append(go.Frame(
            data=[
                # sector previous active (trail)
                go.Scatter(x=p_prev_sec_x, y=p_prev_sec_y),
                # sector current active
                go.Scatter(x=sec_x, y=sec_y),
                # sector current inactive
                go.Scatter(x=sec_inact_x, y=sec_inact_y),
                # floor previous active (trail)
                go.Scatter(x=p_prev_floor_x, y=p_prev_floor_y),
                # floor current active
                go.Scatter(x=floor_x, y=floor_y),
                # floor current inactive
                go.Scatter(x=floor_inact_x, y=floor_inact_y)
            ],
            name=cache_key,
            traces=trace_indices,
            layout=dict(
                annotations=base_annotations_clean + current_annotations + [
                    # [Center] Time Display (Between Titles)
                    dict(
                        x=0.55, y=1.05,xref='paper', yref='paper',
                        text=f"<b>{time_label}</b>",
                        showarrow=False,
                        font=dict(size=20, color="black"),
                        xanchor='center'
                    ),
                    # [Left] Sector Map Stats (Below Title)
                    dict(
                        x=0.35, y=1.02, xref='paper', yref='paper', 
                        text=f"전체: {sector_total_active+sec_inact_cnt}명 | 활성: <span style='color:blue'>{sector_total_active}명</span> | 비활성: <span style='color:gray'>{sec_inact_cnt}명</span> | 실외활성: {o_act}명",
                        showarrow=False,
                        font=dict(size=14, color="black"),
                        xanchor='center'
                    ),
                    # [Right] Floor Map Stats (Below Title)
                    dict(
                        x=0.85, y=1.02, xref='paper', yref='paper', 
                        text=f"전체: {i_act+floor_inact_cnt}명 | 활성: <span style='color:blue'>{i_act}명</span> | 비활성: <span style='color:gray'>{floor_inact_cnt}명</span>",
                        showarrow=False,
                        font=dict(size=14, color="black"),
                        xanchor='center'
                    )
                ], 
                title=dict(text="") # Clear main title as we use annotations
            )
        ))
        
        slider_steps.append(dict(
            args=[[cache_key], dict(frame=dict(duration=400, redraw=True), mode='immediate')],
            label=time_label,
            method='animate'
        ))
        
    fig.frames = frames
    
    # 5. Layout 설정
    fig.update_layout(
        height=910,
        showlegend=True,
        template="plotly_white", # 기본 템플릿
        paper_bgcolor='white',   # 배경색 강제 지정
        plot_bgcolor='white',    # 플롯 배경색 강제 지정
        margin=dict(l=20, r=20, t=50, b=20),
        updatemenus=[dict(
            type='buttons', showactive=False,
            y=-0.1, x=0.0, xanchor='left', yanchor='top',
            pad=dict(t=10, r=10),
            buttons=[
                dict(label='▶️ 재생', method='animate',
                     args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True, mode='immediate')]),
                dict(label='⏸️ 정지', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        )],
        sliders=[dict(
            active=0, yanchor='top', xanchor='left',
            currentvalue=dict(font=dict(size=12, color="black"), prefix='시간: ', visible=True, xanchor='right'),
            len=0.9, x=0.1, y=-0.1,
            steps=slider_steps,
            font=dict(color="black") # 슬라이더 폰트 검정
        )],
        hovermode='closest',
        font=dict(color="black") # 전체 폰트 검정 강제
    )
    
    # Axis 설정 (Range 고정)
    # Sector Map (Left) - 1690 -> 1300 (Range 좁혀서 확대 효과, 70% 영역 꽉 차게)
    fig.update_xaxes(range=[-20, 1300], showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[-20, 1100], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', row=1, col=1)
    
    # Floor Map (Right) - Grid 표시
    fig.update_xaxes(range=[-5, floor_bg['length_x']+5], showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(range=[-5, floor_bg['length_y']+5], showgrid=False, zeroline=False, scaleanchor='x2', scaleratio=1, showticklabels=False, row=1, col=2)

    # sanitize and return JSON-safe dict
    try:
        _clean_figure_for_json(fig)
    except Exception:
        pass
    try:
        import plotly.io as _pio, json as _json
        fig_json = _pio.to_json(fig)
        fig_dict = _json.loads(fig_json)
        return fig_dict
    except Exception:
        return fig



def _create_sector_map_cached_fast(frame_data: dict, time_index: int) -> go.Figure:
    """캐시 데이터를 사용하여 Sector Map 생성 (초경량화 - 전역 캐시 사용)"""
    
    # 전역 캐시에서 spot 데이터 로드 (1회)
    spot_df, spot_pos_df = load_spot_data_cached()
    
    if spot_df.empty:
        return None
    
    # Sector 크기
    sector_width = 1243
    sector_height = 1092
    
    # 실외 spot 필터링
    outdoor_spots = spot_df[spot_df['floor_no'].isna()].copy()
    
    # 빌딩 좌표 및 building_no 매핑
    buildings = {
        'WWT': {'no': 3, 'x1': 880, 'y1': 793, 'x2': 978, 'y2': 947, 'color': 'rgba(34,197,94,0.15)', 'border': 'rgba(34,197,94,0.7)'},
        'FAB': {'no': 1, 'x1': 187, 'y1': 754, 'x2': 530, 'y2': 954, 'color': 'rgba(249,115,22,0.15)', 'border': 'rgba(249,115,22,0.7)'},
        'CUB': {'no': 2, 'x1': 225, 'y1': 626, 'x2': 470, 'y2': 721, 'color': 'rgba(59,130,246,0.15)', 'border': 'rgba(59,130,246,0.7)'},
        'Office': {'no': 4, 'x1': 682, 'y1': 753, 'x2': 812, 'y2': 917, 'color': 'rgba(168,85,247,0.15)', 'border': 'rgba(168,85,247,0.7)'}
    }
    
    # Spot 색상 정의 (constructionSite를 더 투명하게)
    spot_colors = {
        'constructionSite': {'fill': 'rgba(200,200,200,0.05)', 'line': 'rgba(180,180,180,0.2)'},
        'restSpace': {'fill': 'rgba(16,185,129,0.12)', 'line': 'rgba(16,185,129,0.4)'},
        'innerTarget': {'fill': 'rgba(59,130,246,0.12)', 'line': 'rgba(59,130,246,0.4)'},
        'parkingLot': {'fill': 'rgba(107,114,128,0.12)', 'line': 'rgba(107,114,128,0.4)'},
        'etc': {'fill': 'rgba(156,163,175,0.08)', 'line': 'rgba(156,163,175,0.3)'}
    }
    
    # Figure 생성
    fig = go.Figure()
    
    # Spot 폴리곤 그리기 (라벨 없이 빠르게)
    for _, spot in outdoor_spots.iterrows():
        spot_no = spot['spot_no']
        div_type = spot['div'] if pd.notna(spot.get('div')) else 'etc'
        
        spot_coords = spot_pos_df[spot_pos_df['spot_no'] == spot_no].sort_values('point_no')
        if spot_coords.empty:
            continue
        
        x_coords = spot_coords['x'].tolist()
        y_coords = spot_coords['y'].tolist()
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        colors = spot_colors.get(div_type, spot_colors['etc'])
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            fill='toself',
            fillcolor=colors['fill'],
            line=dict(color=colors['line'], width=1),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 빌딩 사각형 추가
    shapes = []
    for name, coords in buildings.items():
        shapes.append(dict(
            type="rect",
            x0=coords['x1'], y0=coords['y1'],
            x1=coords['x2'], y1=coords['y2'],
            fillcolor=coords['color'],
            line=dict(color=coords['border'], width=2)
        ))
    
    # T41 실외 활성 위치 표시 (파란색)
    outdoor_active = frame_data.get('outdoor_active', [])
    if outdoor_active:
        x_active = [p[0] for p in outdoor_active]
        y_active = [p[1] for p in outdoor_active]
        fig.add_trace(go.Scatter(
            x=x_active, y=y_active,
            mode='markers',
            marker=dict(size=6, color='#3B82F6', opacity=0.7),
            name=f'활성 ({len(outdoor_active)})',
            showlegend=True,
            hovertemplate="활성<br>x: %{x}<br>y: %{y}<extra></extra>"
        ))
    
    # T41 실외 비활성 위치 표시 (회색, 더 작게)
    outdoor_inactive = frame_data.get('outdoor_inactive', [])
    if outdoor_inactive:
        x_inactive = [p[0] for p in outdoor_inactive]
        y_inactive = [p[1] for p in outdoor_inactive]
        fig.add_trace(go.Scatter(
            x=x_inactive, y=y_inactive,
            mode='markers',
            marker=dict(size=4, color='#9CA3AF', opacity=0.5),
            name=f'비활성 ({len(outdoor_inactive)})',
            showlegend=True,
            hovertemplate="비활성<br>x: %{x}<br>y: %{y}<extra></extra>"
        ))
    
    # 실외 게이트웨이 표시 (붉은색 네모)
    outdoor_gw = load_outdoor_gateway_cached()
    if not outdoor_gw.empty:
        fig.add_trace(go.Scatter(
            x=outdoor_gw['location_x'].tolist(),
            y=outdoor_gw['location_y'].tolist(),
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(220, 38, 38, 0.8)',  # 붉은색
                symbol='square',  # 네모
                line=dict(color='#7F1D1D', width=1)
            ),
            name=f'Gateway ({len(outdoor_gw)})',
            showlegend=True,
            text=outdoor_gw['name'].tolist(),
            hovertemplate="<b>Gateway</b><br>%{text}<br>x: %{x}<br>y: %{y}<extra></extra>"
        ))
    
    # 빌딩 위에 인원수 표시
    annotations = []
    building_counts = frame_data.get('building_counts', {})
    for name, coords in buildings.items():
        building_no = coords['no']
        # 키가 문자열일 수도 있고 정수일 수도 있음
        count = building_counts.get(str(building_no), building_counts.get(building_no, 0))
        
        center_x = (coords['x1'] + coords['x2']) / 2
        center_y = (coords['y1'] + coords['y2']) / 2
        
        annotations.append(dict(
            x=center_x, y=center_y + 25,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=11, color='#333'),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=2
        ))
        
        annotations.append(dict(
            x=center_x, y=center_y - 15,
            text=f"👷 {count}명",
            showarrow=False,
            font=dict(size=12, color='#1E40AF', family='Arial Black'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#3B82F6',
            borderwidth=1,
            borderpad=3
        ))
    
    # 현재 시간 계산
    hours = ((time_index - 1) * 5) // 60
    minutes = ((time_index - 1) * 5) % 60
    time_str = f"{hours:02d}:{minutes:02d}"
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f"📍 Y-Project Sector Map | ⏰ {time_str}",
            font=dict(size=16, color='#333')
        ),
        xaxis=dict(
            range=[-20, sector_width + 20],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-20, sector_height + 20],
            showgrid=False,
            scaleanchor='x',
            zeroline=False,
            showticklabels=False
        ),
        height=850,
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def _create_sector_map_cached(frame_data: dict, time_index: int) -> go.Figure:
    """캐시 데이터를 사용하여 Sector Map 생성 (경량화)"""
    
    # 데이터 파일 경로
    data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    
    spot_path = data_folder / 'spot.csv'
    spot_pos_path = data_folder / 'spot_position.csv'
    
    # 파일 존재 확인
    if not all(p.exists() for p in [spot_path, spot_pos_path]):
        return None
    
    # 데이터 로드 (캐시)
    @st.cache_data(ttl=3600)
    def load_spot_data():
        spot_df = pd.read_csv(spot_path)
        spot_pos_df = pd.read_csv(spot_pos_path)
        return spot_df, spot_pos_df
    
    spot_df, spot_pos_df = load_spot_data()
    
    # Sector 크기
    sector_width = 1243
    sector_height = 1092
    
    # 실외 spot 필터링
    outdoor_spots = spot_df[spot_df['floor_no'].isna()].copy()
    
    # 빌딩 좌표 및 building_no 매핑
    buildings = {
        'WWT': {'no': 3, 'x1': 880, 'y1': 793, 'x2': 978, 'y2': 947, 'color': 'rgba(34,197,94,0.15)', 'border': 'rgba(34,197,94,0.7)'},
        'FAB': {'no': 1, 'x1': 187, 'y1': 754, 'x2': 530, 'y2': 954, 'color': 'rgba(249,115,22,0.15)', 'border': 'rgba(249,115,22,0.7)'},
        'CUB': {'no': 2, 'x1': 225, 'y1': 626, 'x2': 470, 'y2': 721, 'color': 'rgba(59,130,246,0.15)', 'border': 'rgba(59,130,246,0.7)'},
        'Office': {'no': 4, 'x1': 682, 'y1': 753, 'x2': 812, 'y2': 917, 'color': 'rgba(168,85,247,0.15)', 'border': 'rgba(168,85,247,0.7)'}
    }
    
    # Spot 색상 정의
    spot_colors = {
        'constructionSite': {'fill': 'rgba(200,200,200,0.08)', 'line': 'rgba(180,180,180,0.3)'},
        'restSpace': {'fill': 'rgba(16,185,129,0.15)', 'line': 'rgba(16,185,129,0.5)'},
        'innerTarget': {'fill': 'rgba(59,130,246,0.15)', 'line': 'rgba(59,130,246,0.5)'},
        'parkingLot': {'fill': 'rgba(107,114,128,0.15)', 'line': 'rgba(107,114,128,0.5)'},
        'etc': {'fill': 'rgba(156,163,175,0.12)', 'line': 'rgba(156,163,175,0.4)'}
    }
    
    # Figure 생성
    fig = go.Figure()
    
    # Spot 폴리곤 그리기
    annotations = []
    for _, spot in outdoor_spots.iterrows():
        spot_no = spot['spot_no']
        spot_name = spot['name'] if pd.notna(spot['name']) else f"Spot #{spot_no}"
        div_type = spot['div'] if pd.notna(spot.get('div')) else 'etc'
        
        spot_coords = spot_pos_df[spot_pos_df['spot_no'] == spot_no].sort_values('point_no')
        if spot_coords.empty:
            continue
        
        x_coords = spot_coords['x'].tolist()
        y_coords = spot_coords['y'].tolist()
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        colors = spot_colors.get(div_type, spot_colors['etc'])
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            fill='toself',
            fillcolor=colors['fill'],
            line=dict(color=colors['line'], width=1),
            mode='lines',
            name=spot_name,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 빌딩 사각형 추가
    shapes = []
    for name, coords in buildings.items():
        shapes.append(dict(
            type="rect",
            x0=coords['x1'], y0=coords['y1'],
            x1=coords['x2'], y1=coords['y2'],
            fillcolor=coords['color'],
            line=dict(color=coords['border'], width=2)
        ))
    
    # 캐시에서 T41 데이터 가져오기
    outdoor_active = frame_data.get('outdoor_active', [])
    outdoor_inactive = frame_data.get('outdoor_inactive', [])
    building_counts = frame_data.get('building_counts', {})
    
    # 실외 활성 T41 표시 (파란색)
    if outdoor_active:
        x_active = [p[0] for p in outdoor_active]
        y_active = [p[1] for p in outdoor_active]
        fig.add_trace(go.Scatter(
            x=x_active, y=y_active,
            mode='markers',
            marker=dict(size=6, color='#3B82F6', opacity=0.9, line=dict(width=0.5, color='white')),
            name=f'활성 ({len(outdoor_active)})',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    # 실외 비활성 T41 표시 (회색)
    if outdoor_inactive:
        x_inactive = [p[0] for p in outdoor_inactive]
        y_inactive = [p[1] for p in outdoor_inactive]
        fig.add_trace(go.Scatter(
            x=x_inactive, y=y_inactive,
            mode='markers',
            marker=dict(size=4, color='#9CA3AF', opacity=0.5),
            name=f'비활성 ({len(outdoor_inactive)})',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    # 빌딩 위에 인원수 표시
    for name, coords in buildings.items():
        building_no = coords['no']
        count = building_counts.get(str(building_no), building_counts.get(building_no, 0))
        
        center_x = (coords['x1'] + coords['x2']) / 2
        center_y = (coords['y1'] + coords['y2']) / 2
        
        # 빌딩 이름 + 인원수
        annotations.append(dict(
            x=center_x, y=center_y + 25,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=11, color='#333'),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=2
        ))
        
        annotations.append(dict(
            x=center_x, y=center_y - 15,
            text=f"👷 {count}",
            showarrow=False,
            font=dict(size=16, color='#1E40AF', family='Arial Black'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#3B82F6',
            borderwidth=1,
            borderpad=4
        ))
    
    # 현재 시간 표시
    hours = ((time_index - 1) * 5) // 60
    minutes = ((time_index - 1) * 5) % 60
    time_str = f"{hours:02d}:{minutes:02d}"
    
    # 통계 정보
    total_outdoor = len(outdoor_active) + len(outdoor_inactive)
    total_indoor = sum(int(v) for v in building_counts.values())
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f"📍 Y-Project | ⏰ {time_str} | 실외: {total_outdoor} | 실내: {total_indoor}",
            font=dict(size=16, color='#333')
        ),
        xaxis=dict(
            range=[-20, sector_width + 20],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-20, sector_height + 20],
            showgrid=False,
            scaleanchor='x',
            zeroline=False,
            showticklabels=False
        ),
        height=910,
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def _export_location_animation_gif(location_cache: dict, start_idx: int, end_idx: int, fps: int, cache_folder: Path) -> Path:
    """위치 애니메이션을 GIF로 내보내기"""
    try:
        import io
        from PIL import Image
        import plotly.io as pio
        
        frames = []
        
        for idx in range(start_idx, end_idx + 1):
            frame_data = location_cache.get(str(idx), {})
            fig = _create_sector_map_cached(frame_data, idx)
            if fig is None:
                continue
            
            # Plotly figure를 이미지로 변환
            img_bytes = pio.to_image(fig, format='png', width=1100, height=910, scale=1)
            img = Image.open(io.BytesIO(img_bytes))
            frames.append(img)
        
        if not frames:
            return None
        
        # GIF 저장
        gif_path = cache_folder / f"t41_location_{start_idx}_{end_idx}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
        
        return gif_path
        
    except ImportError as e:
        st.error(f"GIF 생성을 위해 kaleido와 Pillow가 필요합니다: {e}")
        return None
    except Exception as e:
        st.error(f"GIF 생성 오류: {e}")
        return None


def _create_sector_map(loader: CachedDataLoader) -> go.Figure:
    """Sector Map 생성 (실외 영역)"""
    import os
    
    # 데이터 파일 경로 - Datafile 폴더에서 찾기
    base_path = Path(__file__).parent if '__file__' in dir() else Path('.')
    data_folder = base_path / 'Datafile' / 'Yongin_Cluster_202512010'
    
    # 경로가 없으면 상대 경로로 시도
    if not data_folder.exists():
        data_folder = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')
    
    irfm_path = data_folder / 'irfm.csv'
    spot_path = data_folder / 'spot.csv'
    spot_pos_path = data_folder / 'spot_position.csv'
    
    # 파일 존재 확인
    if not all(p.exists() for p in [irfm_path, spot_path, spot_pos_path]):
        return None
    
    # 데이터 로드
    irfm_df = pd.read_csv(irfm_path)
    spot_df = pd.read_csv(spot_path)
    spot_pos_df = pd.read_csv(spot_pos_path)
    
    # Sector 크기 계산
    sector_width = 1243
    sector_height = 1092
    
    # 실외 spot 필터링 (floor_no가 NaN인 경우)
    outdoor_spots = spot_df[spot_df['floor_no'].isna()].copy()
    
    # 빌딩 좌표
    buildings = {
        'WWT': {'x1': 880, 'y1': 793, 'x2': 978, 'y2': 947, 'color': 'rgba(34,197,94,0.12)', 'border': 'rgba(34,197,94,0.6)'},
        'FAB': {'x1': 187, 'y1': 754, 'x2': 530, 'y2': 954, 'color': 'rgba(249,115,22,0.12)', 'border': 'rgba(249,115,22,0.6)'},
        'CUB': {'x1': 225, 'y1': 626, 'x2': 470, 'y2': 721, 'color': 'rgba(59,130,246,0.12)', 'border': 'rgba(59,130,246,0.6)'},
        'Office': {'x1': 682, 'y1': 753, 'x2': 812, 'y2': 917, 'color': 'rgba(168,85,247,0.12)', 'border': 'rgba(168,85,247,0.6)'}
    }
    
    # Spot 색상 정의
    spot_colors = {
        'constructionSite': {'fill': 'rgba(200,200,200,0.08)', 'line': 'rgba(180,180,180,0.3)'},
        'restSpace': {'fill': 'rgba(16,185,129,0.15)', 'line': 'rgba(16,185,129,0.5)'},
        'innerTarget': {'fill': 'rgba(59,130,246,0.15)', 'line': 'rgba(59,130,246,0.5)'},
        'parkingLot': {'fill': 'rgba(107,114,128,0.15)', 'line': 'rgba(107,114,128,0.5)'},
        'etc': {'fill': 'rgba(156,163,175,0.12)', 'line': 'rgba(156,163,175,0.4)'}
    }
    
    # Figure 생성
    fig = go.Figure()
    
    # 라벨 중첩 방지용 위치 리스트
    label_positions = []
    min_distance = 35
    
    def adjust_label_position(x, y, name):
        """라벨 위치 중첩 방지"""
        nonlocal label_positions
        adjusted_x, adjusted_y = x, y
        for _ in range(10):  # 최대 10번 시도
            collision = False
            for lx, ly in label_positions:
                dist = ((adjusted_x - lx)**2 + (adjusted_y - ly)**2)**0.5
                if dist < min_distance:
                    collision = True
                    # 충돌 시 위치 조정
                    adjusted_y += 20
                    break
            # (label collision adjustment only — external lookup removed to avoid
            # referencing variables not in this scope)
            if not collision:
                break
        label_positions.append((adjusted_x, adjusted_y))
        return adjusted_x, adjusted_y
    
    # Spot 폴리곤 그리기
    annotations = []
    for _, spot in outdoor_spots.iterrows():
        spot_no = spot['spot_no']
        spot_name = spot['name'] if pd.notna(spot['name']) else f"Spot #{spot_no}"
        div_type = spot['div'] if pd.notna(spot.get('div')) else 'etc'
        
        # 해당 spot의 좌표 가져오기
        spot_coords = spot_pos_df[spot_pos_df['spot_no'] == spot_no].sort_values('point_no')
        if spot_coords.empty:
            continue
        
        x_coords = spot_coords['x'].tolist()
        y_coords = spot_coords['y'].tolist()
        
        # 폴리곤 닫기
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        # 색상 선택
        colors = spot_colors.get(div_type, spot_colors['etc'])
        
        # Spot 폴리곤 추가
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor=colors['fill'],
            line=dict(color=colors['line'], width=1),
            mode='lines',
            name=spot_name,
            showlegend=False,
            hovertemplate=f"<b>{spot_name}</b><br>Spot #{spot_no}<extra></extra>"
        ))
        
        # 라벨 중심 좌표 계산
        center_x = sum(x_coords[:-1]) / len(x_coords[:-1])
        center_y = sum(y_coords[:-1]) / len(y_coords[:-1])
        
        # 라벨 위치 조정
        adj_x, adj_y = adjust_label_position(center_x, center_y, spot_name)
        
        # 라벨 텍스트 (너무 길면 축약)
        label_text = spot_name if len(spot_name) <= 12 else spot_name[:10] + ".."
        
        annotations.append(dict(
            x=adj_x, y=adj_y,
            text=label_text,
            showarrow=False,
            font=dict(size=7, color='#000000'),
            bgcolor='rgba(255,255,255,0.6)',
            borderpad=1
        ))
    
    # 빌딩 사각형 추가
    for name, coords in buildings.items():
        fig.add_shape(
            type="rect",
            x0=coords['x1'], y0=coords['y1'],
            x1=coords['x2'], y1=coords['y2'],
            fillcolor=coords['color'],
            line=dict(color=coords['border'], width=2)
        )
    
    # 레이아웃 설정 (대시보드에 맞게 스케일 조정)
    fig.update_layout(
        title=dict(text="Y-Project Sector Map (실외)", font=dict(size=16, color='#333')),
        xaxis=dict(
            range=[-20, sector_width + 20],
            title="X",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=False
        ),
        yaxis=dict(
            range=[-20, sector_height + 20],
            title="Y",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            scaleanchor='x',
            zeroline=False
        ),
        height=910,  # 대시보드에 맞게 높이 조정 (130%)
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='white',
        showlegend=False,
        annotations=annotations
    )
    
    return fig

# ==================== MobilePhone 탭 ====================
def render_mobile_tab(loader: CachedDataLoader):
    """모바일폰 분석 탭"""
    st.markdown("""
    <div class="main-header">
        <h1>📱 MobilePhone</h1>
        <p>모바일 기기 유동인구 분석</p>
    </div>
    """, unsafe_allow_html=True)
    
    sub_tabs = st.tabs(["📊 기기 현황", "⏱️ 시간별 분석", "🏢 구역별 분석", "📈 상세 통계"])
    
    with sub_tabs[0]:
        render_mobile_overview_tab(loader)
    
    with sub_tabs[1]:
        render_mobile_hourly(loader)
    
    with sub_tabs[2]:
        render_mobile_sward(loader)
    
    with sub_tabs[3]:
        render_mobile_statistics(loader)

def render_mobile_overview_tab(loader: CachedDataLoader):
    """모바일폰 기기 현황"""
    device_type_stats = loader.load_flow_device_type_stats()
    
    if device_type_stats is not None and not device_type_stats.empty:
        # 캐시 파일의 컬럼명에 맞춰 처리
        count_col = 'total_devices' if 'total_devices' in device_type_stats.columns else (
            'unique_devices' if 'unique_devices' in device_type_stats.columns else device_type_stats.columns[2]
        )
        
        if 'device_name' in device_type_stats.columns:
            type_summary = device_type_stats
        elif 'type_name' in device_type_stats.columns:
            # 캐시에서 type_name 컬럼 활용
            type_summary = device_type_stats.copy()
            type_summary['device_name'] = type_summary['type_name']
        else:
            type_summary = device_type_stats.copy()
            type_col = 'type' if 'type' in device_type_stats.columns else 'device_type'
            type_summary['device_name'] = type_summary[type_col].map({
                config.TYPE_10_ANDROID: 'Android',
                config.TYPE_1_IPHONE: 'iPhone'
            }).fillna('Other')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 파이 차트
            fig = go.Figure(data=[go.Pie(
                labels=type_summary['device_name'],
                values=type_summary[count_col],
                marker_colors=[THEME['mobile_android'], THEME['mobile_iphone']],
                hole=0.5,
                textinfo='label+percent',
                textfont_size=14
            )])
            fig.update_layout(
                title=dict(text='기기 타입 비율', font=dict(size=14, color=THEME['text_primary'])),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['text_primary']),
                legend=dict(font=dict(color=THEME['text_primary']))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 바 차트
            fig = go.Figure()
            for i, row in type_summary.iterrows():
                color = THEME['mobile_android'] if row['device_name'] == 'Android' else THEME['mobile_iphone']
                fig.add_trace(go.Bar(
                    x=[row['device_name']],
                    y=[row[count_col]],
                    marker_color=color,
                    name=row['device_name'],
                    text=[f"{row[count_col]:,}"],
                    textposition='auto'
                ))
            fig.update_layout(
                title=dict(text='기기 타입별 수량', font=dict(size=14, color=THEME['text_primary'])),
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['text_primary']),
                xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary']))
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("기기 타입 데이터가 없습니다.")
    
    # Building별 모바일 기기 분포 추가
    st.markdown("---")
    st.markdown("#### Building별 모바일 기기 분포")
    render_mobile_building_distribution(loader)

def render_mobile_building_distribution(loader: CachedDataLoader):
    """Mobile Building별 분포 차트 (기기 현황 서브탭용, 필터 없음)"""
    try:
        df = get_flow_cache(loader)
        if not df.empty:
            # Mobile: Type 1 (iPhone), Type 10 (Android)
            mobile_data = df[df['type'].isin([1, 10])].copy()
            
            if not mobile_data.empty:
                # 이름 매핑 로드
                building_names = loader.get_building_names()
                
                # Building별 이름 추가
                mobile_data['building_name'] = mobile_data['building_no'].map(
                    lambda x: building_names.get(int(x), f'Building {x}') if pd.notna(x) else '알 수 없음'
                )
                
                # Building별 집계 (Android vs iPhone)
                agg_data = mobile_data.groupby(['building_name', 'type']).agg({
                    'mac_address': 'nunique'
                }).reset_index()
                agg_data.columns = ['building_name', 'type', 'count']
                
                pivot_data = agg_data.pivot(index='building_name', columns='type', values='count').fillna(0).reset_index()
                
                # 컬럼 이름 매핑
                rename_map = {1: 'iPhone', 10: 'Android'}
                pivot_data = pivot_data.rename(columns=rename_map)
                
                # 스택 바 차트
                fig = go.Figure()
                
                if 'Android' in pivot_data.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_data['building_name'],
                        y=pivot_data['Android'],
                        name='Android',
                        marker_color=THEME['mobile_android']
                    ))
                
                if 'iPhone' in pivot_data.columns:
                    fig.add_trace(go.Bar(
                        x=pivot_data['building_name'],
                        y=pivot_data['iPhone'],
                        name='iPhone',
                        marker_color=THEME['mobile_iphone']
                    ))
                
                fig.update_layout(
                    title=dict(text='Building별 모바일 기기 분포 (Unique MAC)', font=dict(size=14, color=THEME['text_primary'])),
                    barmode='stack',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary']), title='Building'),
                    yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary']), title='기기 수'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 요약 통계
                total_android = int(pivot_data['Android'].sum()) if 'Android' in pivot_data.columns else 0
                total_iphone = int(pivot_data['iPhone'].sum()) if 'iPhone' in pivot_data.columns else 0
                st.caption(f"📱 Android: {total_android:,}대 | iPhone: {total_iphone:,}대 | 합계: {total_android + total_iphone:,}대")
            else:
                st.info("모바일 데이터가 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")

def render_mobile_hourly(loader: CachedDataLoader):
    """모바일폰 시간별 분석 - 컨텐츠 준비 중"""
    st.info("시간별 분석 내용은 구역별 분석에서 확인하실 수 있습니다.")

def render_mobile_sward(loader: CachedDataLoader):
    """
    모바일폰 구역별 분석 - T41 구역별 분석과 동일한 형태
    (활성/비활성 없이 카운트만)
    """
    try:
        # 메인 화면에 위치 필터 UI 표시 (Spot 제외)
        selected_building, selected_floor = render_location_filter(loader, 'mobile_zone')
        
        df = get_flow_cache(loader)
        if not df.empty:
            # 모바일 데이터 (type 1=iPhone, 10=Android)
            mobile_data = df[df['type'].isin([1, 10])].copy()
            
            # 필터 적용 (Spot 제외)
            mobile_data = loader.filter_by_location(
                mobile_data,
                selected_building,
                selected_floor,
                'All'  # Spot은 적용하지 않음
            )
            
            if not mobile_data.empty:
                # time_index별 기기 타입별 집계 (Unique MAC)
                time_agg = mobile_data.groupby(['time_index', 'type']).agg({
                    'mac_address': 'nunique'
                }).reset_index()
                time_agg.columns = ['time_index', 'type', 'count']
                
                # pivot
                pivot_data = time_agg.pivot(index='time_index', columns='type', values='count').fillna(0).reset_index()
                pivot_data.columns.name = None
                
                # 컬럼 이름 정리
                rename_map = {1: 'iphone', 10: 'android'}
                pivot_data = pivot_data.rename(columns=rename_map)
                
                # 시간 레이블 생성
                pivot_data['time_label'] = pivot_data['time_index'].apply(bin_index_to_time_str)
                pivot_data = pivot_data.sort_values('time_index')
                
                # 필터 설명 생성
                filter_parts = []
                if selected_building != 'All':
                    filter_parts.append(selected_building)
                if selected_floor != 'All':
                    filter_parts.append(selected_floor)
                filter_desc = ' > '.join(filter_parts) if filter_parts else '전체 구역'
                
                # 영역 차트 (Android/iPhone 구분)
                fig = go.Figure()
                
                if 'android' in pivot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=pivot_data['android'],
                        fill='tozeroy',
                        fillcolor='rgba(34, 197, 94, 0.6)',  # Android 초록색
                        line=dict(color=THEME['mobile_android'], width=2),
                        name='Android'
                    ))
                
                if 'iphone' in pivot_data.columns:
                    if 'android' in pivot_data.columns:
                        total = pivot_data['android'] + pivot_data['iphone']
                    else:
                        total = pivot_data['iphone']
                    
                    fig.add_trace(go.Scatter(
                        x=pivot_data['time_label'],
                        y=total,
                        fill='tonexty',
                        fillcolor='rgba(59, 130, 246, 0.5)',  # iPhone 파란색
                        line=dict(color=THEME['mobile_iphone'], width=2),
                        name='iPhone'
                    ))
                
                fig.update_layout(
                    title=dict(text=f'시간별 기기 수 추이 ({filter_desc})', font=dict(size=14, color=THEME['text_primary'])),
                    xaxis_title='Time',
                    yaxis_title='기기 수 (Unique MAC)',
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text_primary']),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 요약
                col1, col2, col3, col4 = st.columns(4)
                
                android_col = 'android' if 'android' in pivot_data.columns else None
                iphone_col = 'iphone' if 'iphone' in pivot_data.columns else None
                
                if android_col:
                    with col1:
                        st.metric("최대 Android", f"{int(pivot_data[android_col].max()):,}대")
                    with col2:
                        st.metric("평균 Android", f"{pivot_data[android_col].mean():.0f}대")
                
                if iphone_col:
                    with col3:
                        st.metric("최대 iPhone", f"{int(pivot_data[iphone_col].max()):,}대")
                    with col4:
                        st.metric("평균 iPhone", f"{pivot_data[iphone_col].mean():.0f}대")
                
                # ===== Spot 분석 (별도) =====
                st.markdown("---")
                st.markdown("#### 📍 Spot별 기기 분포")
                render_mobile_spot_analysis(loader, mobile_data)
                
            else:
                st.info("선택된 필터 조건에 해당하는 모바일 데이터가 없습니다.")
        else:
            st.info("캐시 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_mobile_spot_analysis(loader: CachedDataLoader, mobile_data: pd.DataFrame):
    """MobilePhone Spot별 기기 분포 분석 - T41과 유사 (활성/비활성 없이)"""
    try:
        if 'spot_nos' not in mobile_data.columns:
            st.info("spot_nos 컬럼이 없습니다.")
            return
        
        # Spot 목록 추출
        all_spots = set()
        for spots_str in mobile_data['spot_nos'].dropna():
            for spot in str(spots_str).split(','):
                spot = spot.strip()
                if spot and spot != 'nan':
                    all_spots.add(spot)
        
        if not all_spots:
            st.info("Spot 데이터가 없습니다.")
            return
        
        spot_list = sorted(list(all_spots), key=lambda x: int(x) if x.isdigit() else 0)
        
        # Spot 이름 매핑
        spot_names = loader.get_spot_names() if hasattr(loader, 'get_spot_names') else {}
        
        # ===== 1. Spot 선택 → 시간별 기기 수 추이 =====
        st.markdown("##### 📈 Spot별 시간대 기기 추이")
        
        spot_options = [spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s for s in spot_list]
        spot_value_map = {}
        for s in spot_list:
            name = spot_names.get(int(s), f'Spot {s}') if s.isdigit() else s
            spot_value_map[name] = s
        
        selected_spot_name = st.selectbox(
            "Spot 선택",
            spot_options,
            index=0,
            key='mobile_spot_trend'
        )
        selected_spot = spot_value_map.get(selected_spot_name, spot_list[0])
        
        # 선택된 Spot의 시간별 기기 추이
        spot_time_data = []
        for _, row in mobile_data.iterrows():
            spots = str(row['spot_nos']).split(',') if pd.notna(row['spot_nos']) else []
            if selected_spot in [s.strip() for s in spots]:
                spot_time_data.append({
                    'time_index': row['time_index'],
                    'mac_address': row['mac_address'],
                    'type': row['type']
                })
        
        if spot_time_data:
            spot_time_df = pd.DataFrame(spot_time_data)
            
            # time_index별 타입별 집계
            time_agg = spot_time_df.groupby(['time_index', 'type']).agg({
                'mac_address': 'nunique'
            }).reset_index()
            time_agg.columns = ['time_index', 'type', 'count']
            
            pivot_time = time_agg.pivot(index='time_index', columns='type', values='count').fillna(0).reset_index()
            pivot_time.columns.name = None
            pivot_time = pivot_time.rename(columns={1: 'iphone', 10: 'android'})
            pivot_time['time_label'] = pivot_time['time_index'].apply(bin_index_to_time_str)
            pivot_time = pivot_time.sort_values('time_index')
            
            # 영역 차트
            fig = go.Figure()
            
            if 'android' in pivot_time.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_time['time_label'],
                    y=pivot_time['android'],
                    fill='tozeroy',
                    fillcolor='rgba(34, 197, 94, 0.6)',
                    line=dict(color=THEME['mobile_android'], width=2),
                    name='Android'
                ))
            
            if 'iphone' in pivot_time.columns:
                if 'android' in pivot_time.columns:
                    total = pivot_time['android'] + pivot_time['iphone']
                else:
                    total = pivot_time['iphone']
                
                fig.add_trace(go.Scatter(
                    x=pivot_time['time_label'],
                    y=total,
                    fill='tonexty',
                    fillcolor='rgba(59, 130, 246, 0.5)',
                    line=dict(color=THEME['mobile_iphone'], width=2),
                    name='iPhone'
                ))
            
            fig.update_layout(
                title=dict(text=f'{selected_spot_name} - 시간별 기기 수 추이', font=dict(size=14, color=THEME['text_primary'])),
                xaxis_title='Time',
                yaxis_title='기기 수 (Unique MAC)',
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['text_primary']),
                xaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickangle=45, tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                yaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color=THEME['text_secondary']), title_font=dict(color=THEME['text_secondary'])),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary'])),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 요약
            col1, col2, col3, col4 = st.columns(4)
            android_col = 'android' if 'android' in pivot_time.columns else None
            iphone_col = 'iphone' if 'iphone' in pivot_time.columns else None
            
            if android_col:
                with col1:
                    st.metric("최대 Android", f"{int(pivot_time[android_col].max()):,}대")
                with col2:
                    st.metric("평균 Android", f"{pivot_time[android_col].mean():.0f}대")
            
            if iphone_col:
                with col3:
                    st.metric("최대 iPhone", f"{int(pivot_time[iphone_col].max()):,}대")
                with col4:
                    st.metric("평균 iPhone", f"{pivot_time[iphone_col].mean():.0f}대")
        else:
            st.info("선택된 Spot에 데이터가 없습니다.")
        
        st.markdown("---")
        
        # ===== 2. 시간대 선택 → Spot별 분포 비교 =====
        st.markdown("##### 📊 시간대별 Spot 분포 비교")
        
        time_options = [bin_index_to_time_str(i) for i in range(288)]
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.selectbox(
                "시작 시간",
                time_options,
                index=0,
                key='mobile_spot_start_time'
            )
        with col2:
            end_time = st.selectbox(
                "종료 시간",
                time_options,
                index=min(17, len(time_options)-1),
                key='mobile_spot_end_time'
            )
        
        start_idx = time_options.index(start_time)
        end_idx = time_options.index(end_time)
        
        if start_idx > end_idx:
            st.warning("시작 시간이 종료 시간보다 큽니다.")
        else:
            time_filtered = mobile_data[(mobile_data['time_index'] >= start_idx) & (mobile_data['time_index'] <= end_idx)].copy()
            
            if not time_filtered.empty:
                spot_data = []
                for _, row in time_filtered.iterrows():
                    spots = str(row['spot_nos']).split(',') if pd.notna(row['spot_nos']) else []
                    for spot in spots:
                        spot = spot.strip()
                        if spot and spot != 'nan':
                            spot_data.append({
                                'spot_no': spot,
                                'mac_address': row['mac_address'],
                                'type': row['type']
                            })
                
                if spot_data:
                    spot_df = pd.DataFrame(spot_data)
                    
                    spot_agg = spot_df.groupby(['spot_no', 'type']).agg({
                        'mac_address': 'nunique'
                    }).reset_index()
                    spot_agg.columns = ['spot_no', 'type', 'count']
                    
                    pivot_spot = spot_agg.pivot(index='spot_no', columns='type', values='count').fillna(0).reset_index()
                    pivot_spot.columns.name = None
                    pivot_spot = pivot_spot.rename(columns={1: 'iphone', 10: 'android'})
                    
                    pivot_spot['spot_name'] = pivot_spot['spot_no'].apply(
                        lambda x: spot_names.get(int(x), f'Spot {x}') if str(x).isdigit() else x
                    )
                    
                    fig = go.Figure()
                    
                    if 'android' in pivot_spot.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_spot['spot_name'],
                            y=pivot_spot['android'],
                            name='Android',
                            marker_color=THEME['mobile_android']
                        ))
                    
                    if 'iphone' in pivot_spot.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_spot['spot_name'],
                            y=pivot_spot['iphone'],
                            name='iPhone',
                            marker_color=THEME['mobile_iphone']
                        ))
                    
                    fig.update_layout(
                        title=dict(text=f'Spot별 기기 분포 ({start_time} ~ {end_time})', font=dict(size=14, color=THEME['text_primary'])),
                        barmode='stack',
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=THEME['text_primary']),
                        xaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='Spot', font=dict(color=THEME['text_secondary'])), tickangle=45),
                        yaxis=dict(tickfont=dict(color=THEME['text_secondary']), title=dict(text='기기 수', font=dict(color=THEME['text_secondary']))),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color=THEME['text_primary']))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    total_android = int(pivot_spot['android'].sum()) if 'android' in pivot_spot.columns else 0
                    total_iphone = int(pivot_spot['iphone'].sum()) if 'iphone' in pivot_spot.columns else 0
                    st.caption(f"📱 Spot 총 {len(pivot_spot)}개 | Android: {total_android}대 | iPhone: {total_iphone}대")
                else:
                    st.info("선택된 시간대에 Spot 데이터가 없습니다.")
            else:
                st.info("선택된 시간대에 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"Spot 분석 중 오류: {str(e)}")

def render_mobile_statistics(loader: CachedDataLoader):
    """모바일폰 상세 통계"""
    device_stats = loader.load_flow_device_stats()
    
    if device_stats is not None and not device_stats.empty:
        st.markdown("#### 기기별 통계")
        st.dataframe(device_stats.head(100), use_container_width=True, height=400)
        
        csv = device_stats.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv,
            file_name=f"mobile_stats_{loader.date_str}.csv",
            mime="text/csv",
            key=f"dl_mobile_stats_{loader.date_str}"
        )
    else:
        st.info("기기 통계 데이터가 없습니다.")

# ==================== 비밀번호 인증 ====================
def check_password():
    """비밀번호 인증 확인"""
    
    # 이미 인증된 경우 즉시 반환 (UI 렌더링 없이)
    if st.session_state.get("password_correct", False):
        return True
    
    # 로그인 페이지 전용 스타일
    st.markdown("""
    <style>
        .stForm [data-testid="stFormSubmitButton"] button {
            background-color: #0066CC !important;
            color: white !important;
            font-weight: 600 !important;
        }
        .stForm [data-testid="stFormSubmitButton"] button:hover {
            background-color: #0052A3 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 비밀번호 입력 화면
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🏭 SK Hynix Y1 Cluster</h1>
        <h3>IRFM Dashboard</h3>
        <p style="color: #64748B;">Industrial Resources Flow Management System</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 환경 변수 또는 Streamlit Cloud Secrets 또는 기본값 사용
        try:
            correct_password = st.secrets.get("password", app_config.password)
        except Exception:
            correct_password = app_config.password
        
        # 프로덕션 환경에서 기본 비밀번호 사용 시 경고
        if app_config.is_production() and correct_password == "admin":
            logger.warning("Production environment using default password! Please set APP_PASSWORD environment variable.")
        
        with st.form("login_form", clear_on_submit=False):
            password = st.text_input(
                "비밀번호를 입력하세요",
                type="password",
                key="password_field"
            )
            submitted = st.form_submit_button("로그인", use_container_width=True)
            
            if submitted:
                if password == correct_password:
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("❌ 비밀번호가 올바르지 않습니다.")
        
        st.caption("© 2024 SK Ecoplant | TJLABS")
    
    st.stop()

# ==================== DeepCon Simulator 탭 ====================
def render_deepcon_simulator(loader: CachedDataLoader):
    """
    DeepCon Command Center - Total Site Monitoring Grid
    5-min 캐시 데이터 기반 실시간 모니터링 (T41 탭과 동일한 데이터 소스 사용)
    """
    st.markdown('<div class="main-header"><h1>🛰️ DeepCon Command Center</h1><p>SK Hynix Y1 Cluster Total Site Risk Monitoring</p></div>', unsafe_allow_html=True)
    
    # Phase 8: Premium CSS with forced dark text
    st.markdown("""
    <style>
    /* Force all text to be dark and visible */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #000000 !important;
    }
    .main p, .main span, .main div {
        color: #000000 !important;
    }
    [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    .command-header {
        background: rgba(15, 17, 22, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    [data-testid="stMetricLabel"] { color: #000000 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #000000 !important; font-size: 1.8rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # 1. Horizontal Control Deck
    st.markdown('<div class="command-header">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2.5, 3, 3, 1.5])
    
    with c1:
        st.markdown(f"**📅 Analysis Target:** {loader.date_str}")
    
    with c2:
        start_hour = st.slider("🕒 Simulation Start", 0, 23, 8)
        st.caption(f"Configured for {start_hour:02d}:00 block analysis")

    with c3:
        if 'sim_running' not in st.session_state: st.session_state.sim_running = False
        # Map hour -> 5-min interval index (time_points indices are 0..287 for 00:05..24:00)
        # Use (hour*60)//5 - 1 so that 08:00 -> index 95 (00:05 is index 0)
        default_idx = max(0, (start_hour * 60) // 5 - 1)
        if 'sim_idx' not in st.session_state: st.session_state.sim_idx = default_idx
        
        st.write("🕹️ System Controls")
        bc1, bc2 = st.columns(2)
        if bc1.button("▶️ ENGAGE", use_container_width=True, type="primary"):
            st.session_state.sim_running = True
            st.session_state.sim_idx = max(0, (start_hour * 60) // 5 - 1)
            st.session_state.risk_trend_history = []
        if bc2.button("⏹️ ABORT", use_container_width=True):
            st.session_state.sim_running = False
            st.session_state.sim_idx = max(0, (start_hour * 60) // 5 - 1)

    with c4:
        st.markdown(f"""
        <div style="text-align:center; padding-top:12px;">
            <small style="color:#8B949E;">ENGINE</small><br>
            <span style="color:{('#00FF00' if st.session_state.sim_running else '#30363D')}; font-size:1.1rem; font-weight:900;">
                {('ACTIVE' if st.session_state.sim_running else 'IDLE')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Data Preparation - 5-min 캐시 기반 (T41 탭과 동일)
    if st.session_state.sim_running:
        # 5-min 캐시에서 데이터 로드 (T41 탭과 동일한 소스)
        try:
            df5 = loader.load_flow_cache(resolution='5min', columns=['time_index', 'type', 'spot_nos', 'status', 'mac_address'])
        except Exception as e:
            st.error(f'5-min 캐시를 로드할 수 없습니다: {e}')
            return
        
        if df5 is None or df5.empty:
            st.error('5-min 캐시 데이터가 없습니다.')
            return
        
        # 작업자 데이터만 필터링
        dfw = df5[df5['type'] == config.TYPE_41_WORKER].copy()
        dfw = dfw.dropna(subset=['spot_nos'])
        
        if dfw.empty:
            st.info('작업자 이벤트가 없습니다.')
            return

        # 3. Main Live Stream Area
        # spot_nos explode 및 집계
        dfw['spot_list'] = dfw['spot_nos'].astype(str).str.split(',')
        expl = dfw.explode('spot_list')
        expl['spot_list'] = expl['spot_list'].str.strip()
        expl = expl[expl['spot_list'] != '']
        expl['spot_list'] = pd.to_numeric(expl['spot_list'], errors='coerce')
        expl = expl.dropna(subset=['spot_list'])
        expl['spot_list'] = expl['spot_list'].astype(int)
        
        # Zone 이름 매핑 (spot_id -> zone_name)
        # Gateway 구조에서 zone 정보 가져오기
        from src.gateway_structure import get_gateway_structure
        gw_struct = get_gateway_structure()
        spot_to_zone = {}
        zone_names_list = []
        
        for building_id, bdata in gw_struct.items():
            for floor_id, fdata in bdata.get('floors', {}).items():
                for spot_id, spot_data in fdata.get('spots', {}).items():
                    zone_name = spot_data.get('name', f'Spot_{spot_id}')
                    spot_to_zone[int(spot_id)] = zone_name
                    if zone_name not in zone_names_list:
                        zone_names_list.append(zone_name)
        
        # 시간 인덱스 범위 (1~288)
        all_time_indices = sorted(expl['time_index'].unique())
        if not all_time_indices:
            st.info('시간 인덱스가 없습니다.')
            return
        
        # heatmap 구축 (zone x time)
        Z = len(zone_names_list)
        T = 288  # 전체 하루 (5분 단위)
        heat = np.zeros((Z, T), dtype=float)
        
        # Active 작업자만 카운트 (status == 1)
        active_df = expl[expl['status'] == 1].copy()
        
        # 시간대별, spot별 unique MAC 주소 카운트
        if not active_df.empty:
            grp = active_df.groupby(['time_index', 'spot_list'])['mac_address'].nunique()
            
            zone_name_to_idx = {name: idx for idx, name in enumerate(zone_names_list)}
            
            for (t_idx, spot), cnt in grp.items():
                tpos = int(t_idx) - 1  # time_index는 1-based
                if 0 <= tpos < T:
                    zone_name = spot_to_zone.get(int(spot))
                    if zone_name and zone_name in zone_name_to_idx:
                        zi = zone_name_to_idx[zone_name]
                        heat[zi, tpos] = float(cnt)
        # Matrix Setup (Cached)
        if 'cached_sim_date' not in st.session_state or st.session_state.cached_sim_date != sim_date or 'sim_matrices' not in st.session_state:
            with st.spinner("Optimizing Site Tensors..."):
                st.session_state.sim_matrices = {
                    'heatmap_z': np.array(sim_data['heatmap']['z'], dtype=np.float32),
                    'step_features': np.array(sim_data['step_features'], dtype=np.float32)
                }
                st.session_state.cached_sim_date = sim_date
        
        # Prefer live 5-min aggregated matrices when available (on-demand)
        if st.session_state.get('sim_matrices_source') == 'live_5min' and st.session_state.get('sim_matrices_live'):
            matrices = st.session_state.sim_matrices_live
            # Use the live time labels stored with the live matrices; fall back to forecast labels
            time_labels = st.session_state.sim_matrices_live.get('time_labels', sim_data['heatmap']['x'])
            zone_names = sim_data['heatmap']['y']
        else:
            matrices = st.session_state.sim_matrices
            time_labels = sim_data['heatmap']['x']
            zone_names = sim_data['heatmap']['y']
        
        # Manual Time Controls (No Auto-play, No Flickering)
        # Simulator-local zone sort control (small, placed above heatmap)
        sort_display = {
            'spot_no': 'Zone ID (spot_no)',
            'name': 'Zone Name (alphabetical)',
            'risk': 'Risk Score (high→low)'
        }
        sort_keys = list(sort_display.keys())
        sort_labels = list(sort_display.values())
        sel_idx_sim = 0
        # initialize from session if present
        if 'zone_sort_by_sim' in st.session_state:
            try:
                sel_idx_sim = sort_keys.index(st.session_state['zone_sort_by_sim'])
            except Exception:
                sel_idx_sim = 0
        sel_idx_sim = st.selectbox(
            "Simulator Zone Sort",
            range(len(sort_labels)),
            format_func=lambda i: sort_labels[i],
            index=sel_idx_sim,
            key='zone_sort_by_sim',
            help='Simulator-only zone sort order (affects heatmap display)',
            label_visibility='collapsed'
        )
        st.markdown("### 🌡️ Zone Risk Status & 3-Hour Timeline")
        # Apply display-only reordering according to simulator-local setting
        sort_choice = st.session_state.get('zone_sort_by_sim', 'spot_no')
        try:
            z_names = list(zone_names)
            z_arr = matrices['heatmap_z']
            sf = matrices['step_features']

            # Derive spot_no when possible
            import re
            spot_nos = []
            for n in z_names:
                m = re.search(r"(\d+)", str(n))
                spot_nos.append(int(m.group(1)) if m else None)

            if sort_choice == 'spot_no' and any(s is not None for s in spot_nos):
                order = sorted(range(len(z_names)), key=lambda i: (spot_nos[i] if spot_nos[i] is not None else 1e9))
            elif sort_choice == 'name':
                order = sorted(range(len(z_names)), key=lambda i: str(z_names[i]).lower())
            elif sort_choice == 'risk':
                # use max risk over the day as sorting key (desc)
                zone_max = np.max(z_arr, axis=1)
                order = sorted(range(len(z_names)), key=lambda i: -float(zone_max[i]))
            else:
                order = list(range(len(z_names)))

            # Reorder for display (do not mutate session storage originals)
            display_heatmap = z_arr[order, :]
            display_step_features = sf[:, order, :]
            display_zone_names = [z_names[i] for i in order]

            # Replace local references used by rendering code below
            matrices = {'heatmap_z': display_heatmap, 'step_features': display_step_features}
            zone_names = display_zone_names
        except Exception:
            # Fallback: keep original
            matrices = matrices
        
        # Time control bar
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 3, 1, 1])
        
        with ctrl_col1:
            if st.button("⏮️ Prev", use_container_width=True):
                if st.session_state.sim_idx > 0:
                    st.session_state.sim_idx -= 1
                    st.rerun()
        
        with ctrl_col2:
            # Time selection presented as HH:MM labels for clarity
            try:
                default_label = time_labels[st.session_state.sim_idx]
            except Exception:
                default_label = time_labels[0]

            selected_label = st.select_slider(
                "Time Selection",
                options=time_labels,
                value=default_label,
                key='sim_time_select',
                label_visibility='collapsed'
            )
            selected_idx = time_labels.index(selected_label) if selected_label in time_labels else st.session_state.sim_idx
            if selected_idx != st.session_state.sim_idx:
                st.session_state.sim_idx = selected_idx
                st.rerun()
        
        with ctrl_col3:
            if st.button("Next ⏭️", use_container_width=True):
                if st.session_state.sim_idx < len(time_labels) - 1:
                    st.session_state.sim_idx += 1
                    st.rerun()
        
        with ctrl_col4:
            current_time = time_labels[st.session_state.sim_idx]
            st.markdown(f"**{current_time}**")
        # Live 5-min cache load (on-demand)
        with ctrl_col4:
            if st.button("🔄 Load 3h from 5-min cache", use_container_width=True, key='load_5min_cache'):
                try:
                    # compute target time index from current_time (HH:MM)
                    hh, mm = [int(x) for x in current_time.split(':')]
                    center_idx = (hh * 60 + mm) // 5 + 1  # 1-based time_index in cache
                except Exception:
                    center_idx = 1

                history_window = 36  # 36 * 5min = 180min = 3h
                start_idx = max(1, center_idx - (history_window - 1))
                end_idx = center_idx

                # Load 5-min cache and aggregate
                try:
                    df5 = loader.load_flow_cache(resolution='5min', columns=['time_index', 'type', 'spot_nos'])
                except Exception:
                    df5 = None

                if df5 is None or df5.empty:
                    st.error('5-min 캐시를 로드할 수 없습니다.')
                else:
                    dfw = df5[df5['time_index'].between(start_idx, end_idx)].copy()
                    dfw = dfw[dfw['type'] == config.TYPE_41_WORKER]
                    dfw = dfw.dropna(subset=['spot_nos'])
                    if dfw.empty:
                        st.info('선택한 창에 작업자 이벤트가 없습니다.')
                    else:
                        # explode spot_nos
                        dfw['spot_list'] = dfw['spot_nos'].astype(str).str.split(',')
                        expl = dfw.explode('spot_list')
                        expl['spot_list'] = expl['spot_list'].str.strip()
                        expl = expl[expl['spot_list'] != '']
                        expl['spot_list'] = pd.to_numeric(expl['spot_list'], errors='coerce')
                        expl = expl.dropna(subset=['spot_list'])
                        expl['spot_list'] = expl['spot_list'].astype(int)

                        # build mapping zone_name -> spot_id from forecast metadata if available
                        fmeta = sim_data.get('forecasts', []) if isinstance(sim_data, dict) else []
                        name_to_spot = {f.get('zone_name'): int(f.get('spot_id')) for f in fmeta if f.get('zone_name') and f.get('spot_id') is not None}

                        window_len = end_idx - start_idx + 1
                        Z = len(zone_names)
                        heat = np.zeros((Z, window_len), dtype=float)

                        # group counts per time_index & spot
                        grp = expl.groupby(['time_index', 'spot_list']).size()
                        for (t_idx, spot), cnt in grp.items():
                            tpos = t_idx - start_idx
                            # find zone index for this spot via name mapping
                            # reverse mapping from name_to_spot: spot -> zone idx
                            # build once
                        
                        spot_to_zone = {v: k for k, v in name_to_spot.items()} if name_to_spot else {}
                        spot_to_zone_idx = {}
                        for zi, zname in enumerate(zone_names):
                            sid = name_to_spot.get(zname)
                            if sid is not None:
                                spot_to_zone_idx[sid] = zi

                        for (t_idx, spot), cnt in grp.items():
                            tpos = int(t_idx - start_idx)
                            if 0 <= tpos < window_len:
                                zi = spot_to_zone_idx.get(int(spot), None)
                                if zi is not None:
                                    heat[zi, tpos] += float(cnt)

                        # minimal step_features placeholder to satisfy rendering (T, Z, D)
                        sf_shape = (window_len, Z, 4)
                        step_feats = np.zeros(sf_shape, dtype=float)

                        # build HH:MM labels for the live window based on 5-min cache time_index (1-based)
                        live_time_labels = []
                        for t_idx in range(start_idx, end_idx + 1):
                            minutes = (int(t_idx) - 1) * 5
                            hh = (minutes // 60) % 24
                            mm = minutes % 60
                            live_time_labels.append(f"{hh:02d}:{mm:02d}")

                        st.session_state.sim_matrices_live = {
                            'heatmap_z': heat,
                            'step_features': step_feats,
                            'time_labels': live_time_labels,
                            'start_idx': int(start_idx),
                            'end_idx': int(end_idx)
                        }
                        # remember previous sim_idx so we can restore later if needed
                        st.session_state.sim_idx_prev = st.session_state.get('sim_idx', 0)
                        # set sim_idx into the live window coordinate space (center position)
                        try:
                            center_pos = int(end_idx - start_idx)
                            st.session_state.sim_idx = min(center_pos, len(live_time_labels) - 1)
                        except Exception:
                            st.session_state.sim_idx = 0
                        st.session_state.sim_matrices_source = 'live_5min'
                        st.success('5-min 캐시 기반 3시간 창 로드 완료')
                        # rerun to refresh UI and ensure indices align
                        st.rerun()
        
        # Display current frame data
        idx = st.session_state.sim_idx
        # Bound-check index
        idx = max(0, min(idx, len(time_labels) - 1))
        st.session_state.sim_idx = idx
        time_str = time_labels[idx]
        current_risks = matrices['heatmap_z'][:, idx].astype(float)
        # Remove negligible noise to avoid false small risk spikes at day boundaries
        current_risks[current_risks < 0.02] = 0.0
        
        # Head-Up Display (Metrics)
        mc1, mc2, mc3 = st.columns([1.5, 1, 1])
        with mc1:
            st.markdown(f"""
            <div style="background:rgba(0,209,255,0.05); border:1px solid rgba(0,209,255,0.2); padding:15px; border-radius:10px; display:flex; align-items:center; justify-content:center; gap:20px;">
                <div style="text-align:left;"><small style="color:#00D1FF;">MISSION TIME</small><br><span style="font-size:2.2rem; font-weight:900; color:#FFFFFF; font-family:monospace;">{time_str}</span></div>
                <div style="width:2px; height:40px; background:rgba(255,255,255,0.1);"></div>
                <div style="text-align:left;"><small style="color:#FF4B4B;">SITE STATUS</small><br><span style="font-size:1.5rem; font-weight:900; color:{('#FF4B4B' if np.max(current_risks)>0.6 else '#00D1FF')};">{('CRITICAL' if np.max(current_risks)>0.6 else 'NOMINAL')}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.metric("Detection Throughput", f"{len(zone_names)} Zones")
        with mc3:
            anomalies = np.sum(current_risks > 0.4)
            st.metric("Anomalous Risks", str(anomalies), delta=f"{anomalies} zones", delta_color="inverse" if anomalies > 0 else "normal")
        
        # Build heatmap data
        history_window = 36
        start_idx = max(0, st.session_state.sim_idx - history_window)
        end_idx = st.session_state.sim_idx + 1
        heatmap_data = matrices['heatmap_z'][:, start_idx:end_idx]
        time_range = time_labels[start_idx:end_idx]
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data, x=time_range, y=zone_names,
            colorscale=[[0.0, '#E8F4F8'], [0.2, '#B3E5FC'], [0.3, '#4FC3F7'], [0.5, '#FFD54F'], [0.7, '#FF9800'], [1.0, '#D32F2F']],
            colorbar=dict(title="Risk", tickmode="linear", tick0=0, dtick=0.2, tickfont=dict(color="#000000", size=10), len=0.7, x=1.0, xanchor='left'),
            hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Risk: %{z:.2f}<extra></extra>',
            zmin=0, zmax=1
        ))
        
        # Add annotations
        annotations = []

        # FALLBACK: If model step_features worker channel is zero for this time range,
        # try to obtain per-spot active worker counts from the 5-min flow cache.
        fallback_counts = None
        try:
            # check whether any zone has non-zero worker feature at current idx
            any_feat_nonzero = False
            try:
                for zi in range(len(zone_names)):
                    # feat index may be out-of-range for display window; use safe access
                    try:
                        if matrices['step_features'][idx, zi, 0] != 0:
                            any_feat_nonzero = True
                            break
                    except Exception:
                        continue
            except Exception:
                any_feat_nonzero = True

            if not any_feat_nonzero:
                # compute fallback counts from 5-min cache for the current time label
                try:
                    hh, mm = [int(x) for x in time_str.split(':')]
                    target_time_idx = (hh * 60 + mm) // 5 + 1
                except Exception:
                    target_time_idx = None

                if target_time_idx is not None:
                    try:
                        df5 = loader.load_flow_cache(resolution='5min', columns=['time_index', 'type', 'spot_nos', 'status', 'mac_address'])
                        if df5 is not None and not df5.empty:
                            # filter worker rows and the selected 5-min bin
                            dfw = df5[(df5['type'] == config.TYPE_41_WORKER) & (df5['time_index'] == int(target_time_idx))]
                            dfw = dfw.dropna(subset=['spot_nos'])
                            if not dfw.empty:
                                dfw = dfw[dfw['spot_nos'].astype(str).str.strip() != '']
                                # explode spot_nos
                                dfw = dfw.assign(spot_list=dfw['spot_nos'].astype(str).str.split(',')).explode('spot_list')
                                dfw['spot_list'] = dfw['spot_list'].str.strip()
                                dfw = dfw[dfw['spot_list'] != '']
                                # count unique mac per spot for active status only
                                active_df = dfw[dfw['status'] == 1]
                                if not active_df.empty:
                                    grp = active_df.groupby('spot_list')['mac_address'].nunique()
                                else:
                                    grp = dfw.groupby('spot_list')['mac_address'].nunique()

                                fallback_counts = {int(k): int(v) for k, v in grp.to_dict().items()}
                    except Exception:
                        fallback_counts = None
        except Exception:
            fallback_counts = None
        # Build mapping zone_name -> spot_id for lookup (if available in forecast metadata)
        name_to_spot = {}
        try:
            fmeta = sim_data.get('forecasts', []) if isinstance(sim_data, dict) else []
            name_to_spot = {f.get('zone_name'): int(f.get('spot_id')) for f in fmeta if f.get('zone_name') and f.get('spot_id') is not None}
        except Exception:
            name_to_spot = {}
        for i, name in enumerate(zone_names):
            score = float(current_risks[i])
            # Safely extract features (T, Z, D) -> features for this zone at current step
            try:
                feat = matrices['step_features'][idx, i]
                feat = np.array(feat, dtype=float)
            except Exception:
                feat = np.zeros(4, dtype=float)

            # Sanitize numeric issues
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

            # Reconstruct intuitive counts: expm1 of stored log1p values, ensure non-negative integers
            try:
                w_cnt = int(round(max(0.0, np.expm1(feat[0]))))
            except Exception:
                w_cnt = 0
            # fallback: if model feature shows zero but 5-min cache has active counts, use them
            try:
                if w_cnt == 0 and fallback_counts:
                    spot_id = name_to_spot.get(name)
                    if spot_id is not None:
                        w_cnt = int(fallback_counts.get(int(spot_id), 0))
            except Exception:
                pass
            try:
                e_cnt = int(round(max(0.0, np.expm1(feat[1]))))
            except Exception:
                e_cnt = 0

            s_risk = float(feat[2]) if np.isfinite(feat[2]) else 0.0
            
            risk_emoji, risk_color, status_label = ("🟢", "#0277BD", "SAFE") if score < 0.3 else ("🟡", "#F57C00", "CAUTION") if score < 0.6 else ("🔴", "#C62828", "CRITICAL")
            status_parts = [f"<b>{risk_emoji} {score:.2f} ({status_label})</b>"]
            
            if w_cnt > 0 or e_cnt > 0:
                state_info = []
                if w_cnt > 0: state_info.append(f"👷{w_cnt}명")
                if e_cnt > 0: state_info.append(f"🚜{e_cnt}대")
                status_parts.append(" ".join(state_info))
            
            if score > 0.35:
                causes = []
                if w_cnt > 15: causes.append("⚠️고밀집작업")
                elif w_cnt > 10: causes.append("⚠️인력밀집")
                if e_cnt > 3: causes.append("⚠️장비과밀")
                elif e_cnt > 2: causes.append("⚠️장비활동")
                if s_risk > 0.6: causes.append("⚠️고위험구역")
                elif s_risk > 0.5: causes.append("⚠️밀폐/고소")
                if causes: status_parts.append(" ".join(causes))
            
            annotations.append(dict(
                x=0.37, y=i, xref='paper', yref='y', text=" | ".join(status_parts),
                showarrow=False, font=dict(size=9, color=risk_color),
                align='left', xanchor='left', yanchor='middle'
            ))
        
        fig_heatmap.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#000000", height=1000,
            margin=dict(l=10, r=100, t=30, b=10),
            xaxis=dict(
                title=dict(text=f"3h ({time_range[0]}~{time_range[-1]})", font=dict(color="#000000", size=10)),
                tickfont=dict(color="#000000", size=8), showgrid=True, gridcolor='#E0E0E0', side='top',
                tickmode='array', tickvals=[time_range[0], time_range[-1]], ticktext=[time_range[0], time_range[-1]],
                domain=[0, 0.35]
            ),
            yaxis=dict(title="", tickfont=dict(color="#000000", size=10), showgrid=True, gridcolor='#E0E0E0', autorange="reversed", fixedrange=True),
            annotations=annotations, uirevision='constant'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})
        st.caption("💡 **3-hour risk evolution** | 🟢 Safe (<0.3) | 🟡 Caution (0.3-0.6) | 🔴 Critical (>0.6) | Use slider or ⏮️/⏭️ buttons to navigate")
    else:
        st.markdown("""
        <div style="padding:100px; text-align:center; background:rgba(255,255,255,0.03); border:1px dashed rgba(255,255,255,0.1); border-radius:20px;">
            <h2 style="color:#8B949E; opacity:0.5;">DEEPCON COMMAND CENTER STANDBY</h2>
            <p style="color:#8B949E; opacity:0.4;">상단 컨트롤 바의 [ENGAGE] 버튼을 클릭하여 전 구역 실시간 모니터링을 시작하십시오.</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== 메인 ====================
def get_forecast_engine():
    """Forecast Engine 캐싱 일시 중지하여 로직 변경사항 반영"""
    return ForecastEngine()

def render_forecast_tab(loader):
    """DeepCon Forecast 탭 렌더링 - 주중/주말 평균 리스크 분석"""
    st.header("🔮 DeepCon Forecast (DeepCon-STAT)")
    
    st.info("""주중 및 주말의 평균 위험도 패턴을 분석합니다. 
    이 데이터는 과거 데이터를 기반으로 시간대별 평균 위험도를 산출한 결과입니다.""")
    
    # Load weekday/weekend average data
    cache_dir = Path("Cache")
    weekday_file = cache_dir / "forecast_weekday_avg.parquet"
    weekend_file = cache_dir / "forecast_weekend_avg.parquet"
    
    if not weekday_file.exists() or not weekend_file.exists():
        st.error("""주중/주말 평균 데이터를 찾을 수 없습니다. 
        다음 명령을 실행하여 데이터를 생성하세요: `python src/precompute_forecast.py`""")
        return
    
    # Load data
    weekday_df = pd.read_parquet(weekday_file)
    weekend_df = pd.read_parquet(weekend_file)
    
    # 위험도 도출 로직 설명
    with st.expander("📝 Risk Derivation Logic (위험도 도출 로직)"):
        st.markdown("""
        DeepCon-STAT 위험도는 다음과 같은 **Spatiotemporal Tensor** 정보를 기반으로 도출됩니다:
        1. **Worker Density ($W$):** 해당 구역의 작업자 밀집도 (log-scaled)
        2. **Equipment Density ($E$):** 중장비 가동 및 근접도 (log-scaled)
        3. **Static Risk ($P$):** 해당 구역의 고유 위험성 (밀폐공간, 추락 위험 등)
        4. **Spatiotemporal Context ($C$):** 최근 60분간의 주변 구역간 상호작용 및 시간적 흐름
        
        **수식 ($Risk$):**
        $$Risk = f_{\\text{DeepCon}}(W, E, P, C)$$
        *현재 대시보드에서는 시각적 가독성을 위해 위험 작업을 판별할 수 있도록 비선형 정규화(Square Root Scaling)를 적용합니다.*
        """)
    
    st.divider()
    
    # 주중/주말 선택
    day_type = st.radio("📅 분석 구분 선택", ["주중 (Weekday)", "주말 (Weekend)"], horizontal=True)
    
    if day_type == "주중 (Weekday)":
        df = weekday_df.copy()
        day_label = "Weekday"
        day_emoji = "📅"
    else:
        df = weekend_df.copy()
        day_label = "Weekend"
        day_emoji = "🏖️"
    
    # time_index를 hour로 변환 (time_index는 1-288, 5분 단위 → hour는 0-23)
    if 'time_index' in df.columns:
        df['hour'] = ((df['time_index'] - 1) // 12).astype(int)
    
    # 통계 요약
    avg_risk = df['avg_risk'].mean()
    max_risk = df['avg_risk'].max()
    high_risk_count = len(df[df['avg_risk'] >= 0.7])
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{day_emoji} Average Risk", f"{avg_risk:.3f}")
    col2.metric("Max Risk", f"{max_risk:.3f}")
    col3.metric("High Risk Zones (≥0.7)", f"{high_risk_count}")
    
    st.divider()
    
    # 탭으로 구성
    tabs = st.tabs(["📊 Risk Overview", "🗺️ Heatmap", "📈 Top Risk Zones", "🔍 Zone Detail"])
    
    with tabs[0]:
        st.subheader(f"{day_emoji} {day_label} Risk Overview")
        
        # 시간대별 평균 위험도 추이
        if 'hour' in df.columns:
            hourly_avg = df.groupby('hour')['avg_risk'].mean().reset_index()
            fig_hourly = px.line(
                hourly_avg, x='hour', y='avg_risk',
                markers=True, title=f"{day_label} Hourly Average Risk Trend",
                labels={'hour': 'Hour', 'avg_risk': 'Avg Risk Score'},
                range_y=[0, 1.0]
            )
            fig_hourly.update_traces(line_color='#FF4B4B', marker=dict(size=8))
            fig_hourly.update_xaxes(tickmode='linear', tick0=0, dtick=2)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # 구역별 평균 위험도 (Top 20)
        zone_avg = df.groupby('spot_name')['avg_risk'].mean().reset_index()
        zone_avg = zone_avg.sort_values('avg_risk', ascending=False).head(20)
        
        fig_zones = px.bar(
            zone_avg, x='spot_name', y='avg_risk',
            color='avg_risk', color_continuous_scale='RdYlGn_r',
            title=f"{day_label} Top 20 High-Risk Zones (Average)",
            labels={'spot_name': 'Zone', 'avg_risk': 'Avg Risk Score'},
            range_y=[0, 1.0]
        )
        fig_zones.update_layout(xaxis={'tickangle': 45})
        st.plotly_chart(fig_zones, use_container_width=True)
    
    with tabs[1]:
        st.subheader(f"🗺️ {day_label} Spatio-Temporal Risk Heatmap")
        
        # 히트맵 데이터 준비
        if 'hour' in df.columns and 'spot_name' in df.columns:
            pivot_data = df.pivot_table(
                values='avg_risk', index='spot_name', columns='hour', aggfunc='mean'
            ).fillna(0)
            
            # 상위 30개 구역만 표시
            zone_avg_for_filter = df.groupby('spot_name')['avg_risk'].mean().sort_values(ascending=False)
            top_zones = zone_avg_for_filter.head(30).index
            pivot_data = pivot_data.loc[pivot_data.index.isin(top_zones)]
            
            num_zones = len(pivot_data)
            plot_height = max(600, num_zones * 20)
            
            fig_heatmap = px.imshow(
                pivot_data,
                labels=dict(x="Hour", y="Zone", color="Risk Score"),
                color_continuous_scale="YlOrRd",
                aspect="auto",
                title=f"{day_label} 24-Hour Risk Evolution (Top 30 Zones)",
                height=plot_height
            )
            fig_heatmap.update_layout(
                margin=dict(l=150, r=20, t=50, b=50),
                yaxis=dict(tickfont=dict(size=9)),
                xaxis=dict(tickangle=0)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("히트맵 데이터를 생성할 수 없습니다.")
    
    with tabs[2]:
        st.subheader(f"📈 {day_label} Top Risk Zones")
        
        # 시간대별 최고 위험 구역
        if 'hour' in df.columns:
            peak_per_hour = df.loc[df.groupby('hour')['avg_risk'].idxmax()]
            
            fig_peak = px.scatter(
                peak_per_hour, x='hour', y='avg_risk',
                color='spot_name', size='avg_risk',
                title=f"{day_label} Peak Risk Zone per Hour",
                labels={'hour': 'Hour', 'avg_risk': 'Peak Risk Score'},
                range_y=[0, 1.0],
                hover_data=['spot_name']
            )
            fig_peak.update_xaxes(tickmode='linear', tick0=0, dtick=2)
            st.plotly_chart(fig_peak, use_container_width=True)
            
            # 테이블로도 표시
            st.dataframe(
                peak_per_hour[['hour', 'spot_name', 'avg_risk']].sort_values('hour'),
                column_config={
                    'hour': 'Hour',
                    'spot_name': 'Peak Zone',
                    'avg_risk': st.column_config.ProgressColumn(
                        'Risk Score', format='%.3f', min_value=0, max_value=1
                    )
                },
                use_container_width=True,
                hide_index=True
            )
    
    with tabs[3]:
        st.subheader(f"🔍 {day_label} Zone Detail Analysis")
        
        # 구역 선택
        all_zones = sorted(df['spot_name'].unique())
        selected_zone = st.selectbox("분석할 구역 선택", all_zones)
        
        zone_data = df[df['spot_name'] == selected_zone].copy()
        
        if not zone_data.empty:
            # 통계
            zone_avg = zone_data['avg_risk'].mean()
            zone_max = zone_data['avg_risk'].max()
            zone_min = zone_data['avg_risk'].min()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Average Risk", f"{zone_avg:.3f}")
            c2.metric("Max Risk", f"{zone_max:.3f}")
            c3.metric("Min Risk", f"{zone_min:.3f}")
            
            # 시간대별 위험도 추이
            if 'hour' in zone_data.columns:
                zone_data = zone_data.sort_values('hour')
                fig_zone = px.area(
                    zone_data, x='hour', y='avg_risk',
                    title=f"{selected_zone} - {day_label} Hourly Risk Pattern",
                    labels={'hour': 'Hour', 'avg_risk': 'Risk Score'},
                    range_y=[0, 1.0]
                )
                fig_zone.update_traces(line_color='#FF4B4B', fillcolor='rgba(255, 75, 75, 0.3)')
                fig_zone.update_xaxes(tickmode='linear', tick0=0, dtick=2)
                st.plotly_chart(fig_zone, use_container_width=True)
            
            # 상세 데이터 테이블
            st.dataframe(
                zone_data[['hour', 'avg_risk']].sort_values('hour'),
                column_config={
                    'hour': 'Hour',
                    'avg_risk': st.column_config.ProgressColumn(
                        'Risk Score', format='%.4f', min_value=0, max_value=1
                    )
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("선택한 구역의 데이터가 없습니다.")

def main():
    """메인 함수"""
    
    # 로거 초기화
    logger.info("="*80)
    logger.info("DeepCon Dashboard Starting...")
    logger.info(f"Environment: {app_config.env}")
    logger.info(f"Features - Transformer: {app_config.enable_transformer}, Forecast: {app_config.enable_forecast}")
    logger.info("="*80)
    
    try:
        # 비밀번호 인증
        if not check_password():
            return
        
        # 첫 로딩 시 프로그레스 바 표시
        if 'initial_load_done' not in st.session_state:
            start_time = time.time()
            with st.spinner("🔄 데이터 로딩 중..."):
                loader = render_sidebar()
                if loader:
                    # 프로그레스 바로 로딩 상태 표시
                    progress_placeholder = st.empty()
                    with progress_placeholder.container():
                        progress_bar = st.progress(0, text="데이터 초기화 중...")
                        
                        # Flow Cache 로드 (가장 무거움)
                        progress_bar.progress(30, text="Flow 캐시 로딩...")
                        cache_folder = str(loader.cache_folder.parent) if loader._is_new_structure else str(loader.cache_folder)
                        _ = load_flow_cache_cached(cache_folder, loader.date_str, CACHE_RESOLUTION)
                        
                        progress_bar.progress(70, text="분석 데이터 준비...")
                        _ = loader.get_summary()
                        _ = loader.load_t31_time_series()
                        _ = loader.load_t41_time_series()
                        
                        progress_bar.progress(100, text="완료!")
                    
                    progress_placeholder.empty()
                    st.session_state.initial_load_done = True
                    
                    # 성능 로깅
                    load_time = time.time() - start_time
                    logger.info(f"Initial data load completed in {load_time:.2f}s")
        else:
            loader = render_sidebar()
        
        if loader is None:
            logger.warning("Data loader failed to initialize")
            st.warning("⚠️ 데이터를 로드할 수 없습니다.")
            st.info("""
            ### 데이터 준비 방법
            
            1. **Raw 데이터 처리**: `python src/precompute_optimized.py`
            2. **대시보드 캐시 생성**: `python precompute_full.py`
            3. **대시보드 실행**: `streamlit run main.py`
            """)
            return
        
        # --- Phase 5: Lazy Tab Rendering Navigation ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 Dashboard Navigation")
        
        menu_options = [
            "📊 Overview",
            "🔧 T-Ward Type31",
            "👷 T-Ward Type41",
            "📱 MobilePhone",
            "🔮 DeepCon Forecast",
            "🎮 DeepCon Simulator"
        ]
        
        # Initialize current tab
        if "current_tab" not in st.session_state:
            st.session_state.current_tab = menu_options[0]
            
        # Sync radio index with session state
        try:
            current_index = menu_options.index(st.session_state.current_tab)
        except ValueError:
            current_index = 0
            
        selected_menu = st.sidebar.radio(
            "Navigation Menu",
            menu_options,
            index=current_index,
            key="main_navigation_radio",
            label_visibility="collapsed"
        )
        
        # Update state
        st.session_state.current_tab = selected_menu
        
        # Footer at bottom of sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align:center; padding:1rem; opacity:0.5;">
            <small>IRFM Dashboard v2.0</small><br>
            <small>© 2025 TJLABS</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Lazy Rendering Block: Only execute the active component
        if selected_menu == menu_options[0]:
            render_overview(loader)
        elif selected_menu == menu_options[1]:
            render_t31_tab(loader)
        elif selected_menu == menu_options[2]:
            render_t41_tab(loader)
        elif selected_menu == menu_options[3]:
            render_mobile_tab(loader)
        elif selected_menu == menu_options[4]:
            if app_config.enable_forecast:
                render_forecast_tab(loader)
            else:
                st.warning("⚠️ Forecast 기능이 비활성화되어 있습니다.")
        elif selected_menu == menu_options[5]:
            if app_config.enable_simulator:
                # src/tabs/simulator_tab.py의 render 함수 사용 (날짜별 선택 가능)
                try:
                    from src.tabs.simulator_tab import render_simulator_tab
                    render_simulator_tab()
                except ImportError:
                    # Fallback: 기존 함수 사용
                    render_deepcon_simulator(loader)
            else:
                st.warning("⚠️ Simulator 기능이 비활성화되어 있습니다.")
    
    except Exception as e:
        logger.error(f"Critical error in main(): {type(e).__name__}: {str(e)}", exc_info=True)
        st.error("❌ 애플리케이션 오류가 발생했습니다.")
        if not app_config.is_production():
            with st.expander("상세 에러 정보 (개발용)"):
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
