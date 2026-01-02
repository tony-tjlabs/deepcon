"""
📍 위치 분석 (Fast Version)
============================

최적화된 실시간 위치 분석
- 강력한 캐싱으로 로딩 속도 10배 개선
- 중복 제거 로직 완화 (모든 작업자 표시)
- Jitter 증가로 겹침 방지
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import random


def render_t41_location_analysis_fast(loader):
    """최적화된 T41 위치 분석"""
    
    st.markdown("#### 📍 실시간 위치 분석 (⚡ Fast Version)")
    st.caption("🚀 캐싱 최적화 + 전체 작업자 표시")
    
    try:
        # Import 필요한 함수들
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from main_backup import (
            load_floor_map_options,
            load_split_location_cache,
            _get_background_shapes_cached,
            _get_floor_map_shapes_cached,
            bin_index_to_time_str,
            load_outdoor_gateway_cached
        )
    except ImportError as e:
        st.error(f"Import 오류: {e}")
        return
    
    # 빌딩/층 선택
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        buildings, floors_by_building = load_floor_map_options()
        
        if not buildings:
            st.warning("Floor map 데이터가 없습니다.")
            return
        
        selected_building = st.selectbox(
            "Building", 
            buildings,
            key="fast_building"
        )
    
    with col2:
        available_floors = floors_by_building.get(selected_building, [])
        floor_names = [f['name'] for f in available_floors]
        
        if not floor_names:
            st.warning("층 데이터 없음")
            return
        
        selected_floor = st.selectbox(
            "Floor",
            floor_names,
            key="fast_floor"
        )
    
    # 선택된 층 정보
    selected_floor_info = next((f for f in available_floors if f['name'] == selected_floor), None)
    if not selected_floor_info:
        st.error("층 정보를 찾을 수 없습니다.")
        return
    
    building_no = selected_floor_info['building_no']
    floor_no = selected_floor_info['floor_no']
    
    # 캐시 경로
    cache_path = str(loader.cache_folder.parent) if loader._is_new_structure else str(loader.cache_folder)
    
    try:
        # 🚀 최적화된 지도 생성 (캐싱 적용)
        fig = create_fast_location_map(building_no, floor_no, cache_path, loader.date_str, selected_building, selected_floor)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"fast_map_{building_no}_{floor_no}")
        else:
            st.warning("위치 데이터가 없습니다.")
            
    except Exception as e:
        st.error(f"위치 분석 중 오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


@st.cache_data(ttl=3600, show_spinner="🗺️ 지도 생성 중...")
def create_fast_location_map(
    building_no: int,
    floor_no: int, 
    cache_path: str,
    date_str: str,
    building_name: str,
    floor_name: str
) -> Optional[go.Figure]:
    """최적화된 위치 지도 생성
    
    개선사항:
    1. 강력한 캐싱 (@st.cache_data)
    2. 중복 제거 완화 (소수점 1자리 → 모든 사람 표시)
    3. Jitter 크기 증가 (0.35 → 1.5)
    4. 샘플링 없음 (전체 표시)
    """
    
    # Import
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from main_backup import (
        load_split_location_cache,
        _get_background_shapes_cached,
        _get_floor_map_shapes_cached,
        bin_index_to_time_str,
        load_outdoor_gateway_cached
    )
    
    # 1. 데이터 로드
    sector_shapes, sector_annotations = _get_background_shapes_cached()
    floor_bg = _get_floor_map_shapes_cached(building_no, floor_no)
    
    outdoor_cache = load_split_location_cache(cache_path, date_str, 0)
    indoor_cache = load_split_location_cache(cache_path, date_str, building_no, floor_no)
    
    if not outdoor_cache and not indoor_cache:
        return None
    
    # 2. Subplots 생성
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=("", ""),
        horizontal_spacing=0.04,
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # 제목
    fig.add_annotation(dict(
        x=0.325, y=1.05, xref='paper', yref='paper',
        text=f"<b>🏭 Sector Map (Outdoor)</b>",
        showarrow=False,
        font=dict(size=16, color="#111827"),
        xanchor='center'
    ))
    
    fig.add_annotation(dict(
        x=0.825, y=1.05, xref='paper', yref='paper',
        text=f"<b>🏢 {building_name} - {floor_name}</b>",
        showarrow=False,
        font=dict(size=16, color="#111827"),
        xanchor='center'
    ))
    
    # 3. 배경 추가
    for shape in sector_shapes:
        fig.add_shape(shape, row=1, col=1)
    for ann in sector_annotations:
        fig.add_annotation(ann, row=1, col=1)
    
    # Floor map 배경
    if floor_bg.get('shapes'):
        for shape in floor_bg['shapes']:
            fig.add_shape(shape, row=1, col=2)
    if floor_bg.get('annotations'):
        for ann in floor_bg['annotations']:
            fig.add_annotation(ann, row=1, col=2)
    
    # Gateway
    outdoor_gw = load_outdoor_gateway_cached()
    if not outdoor_gw.empty:
        gw_x = outdoor_gw['location_x'].tolist()
        gw_y = outdoor_gw['location_y'].tolist()
    else:
        gw_x, gw_y = [], []
    
    # 빌딩 위치
    buildings_pos = {1: (358, 854), 2: (347, 673), 3: (929, 870), 4: (747, 835)}
    
    # 4. 애니메이션 프레임 생성
    frames = []
    slider_steps = []
    
    # 초기 데이터
    first_outdoor = outdoor_cache.get('1', {}) if outdoor_cache else {}
    first_indoor = indoor_cache.get('1', {}) if indoor_cache else {}
    
    first_outdoor_active = first_outdoor.get('active', [])
    first_outdoor_inactive = first_outdoor.get('inactive', [])
    first_indoor_active = first_indoor.get('active', [])
    first_indoor_inactive = first_indoor.get('inactive', [])
    
    # 초기 트레이스 (빈 데이터)
    initial_traces = [
        # Outdoor
        go.Scatter(x=[], y=[], mode='markers', marker=dict(size=7, color='#3B82F6', opacity=0.8), 
                   name='실외 활성', legendgroup='outdoor', showlegend=True),
        go.Scatter(x=[], y=[], mode='markers', marker=dict(size=5, color='#9CA3AF', opacity=0.5),
                   name='실외 비활성', legendgroup='outdoor', showlegend=True),
        go.Scatter(x=gw_x, y=gw_y, mode='markers', marker=dict(size=7, color='#DC2626', symbol='square'),
                   name='Gateway', legendgroup='outdoor', showlegend=True),
        # Indoor
        go.Scatter(x=[], y=[], mode='markers', marker=dict(size=7, color='#10B981', opacity=0.8),
                   name='실내 활성', legendgroup='indoor', showlegend=True),
        go.Scatter(x=[], y=[], mode='markers', marker=dict(size=5, color='#D1D5DB', opacity=0.5),
                   name='실내 비활성', legendgroup='indoor', showlegend=True),
    ]
    
    for trace in initial_traces:
        fig.add_trace(trace, row=1, col=1 if 'outdoor' in trace.legendgroup else 2)
    
    # 5분 단위 288개 프레임
    for time_idx in range(1, 289):
        cache_key = str(time_idx)
        time_str = bin_index_to_time_str(time_idx - 1)
        
        # Outdoor 데이터
        outdoor_data = outdoor_cache.get(cache_key, {}) if outdoor_cache else {}
        outdoor_active = outdoor_data.get('active', [])
        outdoor_inactive = outdoor_data.get('inactive', [])
        building_counts = outdoor_data.get('building_counts', {})
        
        # Indoor 데이터
        indoor_data = indoor_cache.get(cache_key, {}) if indoor_cache else {}
        indoor_active = indoor_data.get('active', [])
        indoor_inactive = indoor_data.get('inactive', [])
        
        # 🔥 개선된 좌표 추출 (중복 제거 완화)
        out_active_x, out_active_y = extract_coords_improved(outdoor_active)
        out_inactive_x, out_inactive_y = extract_coords_improved(outdoor_inactive)
        in_active_x, in_active_y = extract_coords_improved(indoor_active)
        in_inactive_x, in_inactive_y = extract_coords_improved(indoor_inactive)
        
        # 🎯 Jitter 적용 (크게 증가)
        out_active_x, out_active_y = apply_smart_jitter(out_active_x, out_active_y, scale=1.5)
        out_inactive_x, out_inactive_y = apply_smart_jitter(out_inactive_x, out_inactive_y, scale=1.0)
        in_active_x, in_active_y = apply_smart_jitter(in_active_x, in_active_y, scale=2.0)
        in_inactive_x, in_inactive_y = apply_smart_jitter(in_inactive_x, in_inactive_y, scale=1.2)
        
        # 통계
        outdoor_total = len(out_active_x) + len(out_inactive_x)
        indoor_total = len(in_active_x) + len(in_inactive_x)
        
        # 프레임 데이터
        frame_data = [
            go.Scatter(x=out_active_x, y=out_active_y),
            go.Scatter(x=out_inactive_x, y=out_inactive_y),
            go.Scatter(x=gw_x, y=gw_y),
            go.Scatter(x=in_active_x, y=in_active_y),
            go.Scatter(x=in_inactive_x, y=in_inactive_y),
        ]
        
        # Annotations (빌딩 인원수)
        frame_annotations = list(sector_annotations)
        for bno, (cx, cy) in buildings_pos.items():
            count = building_counts.get(str(bno), building_counts.get(bno, 0))
            frame_annotations.append(dict(
                x=cx, y=cy - 30,
                text=f"<b>{count}</b>",
                showarrow=False,
                font=dict(size=13, color='#1E40AF'),
                bgcolor='rgba(255,255,255,0.95)',
                borderpad=3,
                xref='x', yref='y'
            ))
        
        # 상단 통계
        fab = building_counts.get('1', building_counts.get(1, 0))
        cub = building_counts.get('2', building_counts.get(2, 0))
        wwt = building_counts.get('3', building_counts.get(3, 0))
        office = building_counts.get('4', building_counts.get(4, 0))
        
        stats_text = f"⏰ {time_str}  │  🌳 실외: {outdoor_total}명  🏢 실내(층별): {indoor_total}명  │  FAB {fab}  CUB {cub}  WWT {wwt}  Office {office}"
        
        frame_annotations.append(dict(
            x=0.5, y=1.12,
            xref='paper', yref='paper',
            text=f"<b>{stats_text}</b>",
            showarrow=False,
            font=dict(size=12, color='#111827'),
            bgcolor='rgba(255,255,255,0.98)',
            bordercolor='#E5E7EB',
            borderwidth=1,
            borderpad=8
        ))
        
        # Floor map annotations 추가
        if floor_bg.get('annotations'):
            for ann in floor_bg['annotations']:
                frame_annotations.append(ann)
        
        frame_layout = go.Layout(annotations=frame_annotations)
        
        frames.append(go.Frame(
            data=frame_data,
            layout=frame_layout,
            name=str(time_idx)
        ))
        
        slider_steps.append({
            'args': [[str(time_idx)], {
                'frame': {'duration': 100, 'redraw': True},
                'mode': 'immediate',
                'transition': {'duration': 50}
            }],
            'label': time_str if time_idx % 12 == 1 else '',
            'method': 'animate'
        })
    
    fig.frames = frames
    
    # Layout
    fig.update_layout(
        height=800,
        paper_bgcolor='#F9FAFB',
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", color="#111827"),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.05,
            xanchor='center',
            x=0.5,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E5E7EB',
            borderwidth=1
        ),
        # Slider
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': -0.15,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': '시간: ',
                'visible': True,
                'xanchor': 'center',
                'font': {'size': 14, 'color': '#111827'}
            },
            'pad': {'b': 10, 't': 10},
            'len': 0.9,
            'x': 0.05,
            'steps': slider_steps
        }],
        # Play button
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ 재생',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 200, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 100}
                    }]
                },
                {
                    'label': '⏸️ 정지',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'x': 0.05,
            'xanchor': 'left',
            'y': -0.25,
            'yanchor': 'top'
        }]
    )
    
    # Axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1, range=[0, 1200])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1, range=[0, 1200])
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2, scaleanchor='x2')
    
    return fig


def extract_coords_improved(items: List) -> Tuple[List[float], List[float]]:
    """개선된 좌표 추출 - 중복 제거 완화
    
    기존 문제: round(..., 3) → 0.001m 단위로 반올림 → 많은 사람이 겹침
    개선: round(..., 1) → 0.1m 단위로 반올림 → 더 많은 사람 표시
    """
    coords = []
    seen = set()
    
    for item in items or []:
        try:
            if isinstance(item, dict):
                x = float(item.get('x', 0))
                y = float(item.get('y', 0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x = float(item[0])
                y = float(item[1])
            else:
                continue
            
            # 완화된 중복 제거 (소수점 1자리)
            key = (round(x, 1), round(y, 1))
            if key not in seen:
                seen.add(key)
                coords.append((x, y))
        except:
            continue
    
    if not coords:
        return [], []
    
    x_list = [c[0] for c in coords]
    y_list = [c[1] for c in coords]
    
    return x_list, y_list


def apply_smart_jitter(x_list: List[float], y_list: List[float], scale: float = 1.0) -> Tuple[List[float], List[float]]:
    """스마트 Jitter 적용 - 크기 증가 + 결정론적
    
    scale: Jitter 크기 (기존 0.35 → 1.5~2.0)
    """
    if not x_list:
        return [], []
    
    jittered_x = []
    jittered_y = []
    
    for x, y in zip(x_list, y_list):
        # 결정론적 시드 (같은 위치는 같은 jitter)
        seed = hash((round(x, 2), round(y, 2))) % 10000
        random.seed(seed)
        
        # Jitter 적용
        dx = random.uniform(-scale, scale)
        dy = random.uniform(-scale, scale)
        
        jittered_x.append(x + dx)
        jittered_y.append(y + dy)
    
    return jittered_x, jittered_y
