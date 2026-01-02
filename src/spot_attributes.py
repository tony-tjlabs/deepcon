"""
Spot 속성 관리 모듈
각 spot의 면적, 구역 타입, 기본 위험도 정의
"""

import pandas as pd
from typing import Dict, Optional


# Spot 구역 타입별 기본 위험도 (0~1 scale)
ZONE_TYPE_BASE_RISK = {
    'confinedSpace': 0.4,  # 밀폐공간 - 높은 기본 위험도
    'workFloor': 0.3,      # 작업 공간 (WWT, FAB, CUB) - 중상 위험도
    'restSpace': 0.05,     # 휴게실 - 매우 낮은 위험도
    'smokingArea': 0.05,   # 흡연실 - 매우 낮은 위험도
    'etc': 0.1,            # 기타 - 낮은 위험도
    'outdoor': 0.15,       # 야외 - 낮은~중간 위험도
}

# Spot별 추정 면적 (m²)
# 실제 도면 기반으로 추정하거나, 기본값 사용
SPOT_AREA_ESTIMATES = {
    # 밀폐공간 (좁은 공간)
    121: 25.0,  # TB08_밀폐
    165: 30.0,  # TB02_밀폐
    173: 25.0,  # TB17_밀폐
    
    # 작업 공간 (넓은 공간)
    106: 500.0,  # WWT B1F (전체)
    95: 500.0,   # WWT 1F (전체)
    174: 500.0,  # WWT 2F (전체)
    103: 800.0,  # FAB 1F (전체)
    104: 600.0,  # CUB 1F (전체)
    105: 600.0,  # CUB B1F (전체)
    
    # 휴게실 (중간 크기)
    148: 40.0,   # FAB_휴게실_F02_1층4번
    149: 40.0,   # FAB_휴게실_F03_1층2번
    150: 40.0,   # FAB_휴게실_F02_1층1번
    175: 50.0,   # WWT 2F 휴게실
    
    # 기타 spot (기본값)
    'default': 100.0
}


def get_spot_area(spot_id: int) -> float:
    """
    Spot 면적 반환
    
    Args:
        spot_id: Spot 번호
        
    Returns:
        면적 (m²)
    """
    return SPOT_AREA_ESTIMATES.get(spot_id, SPOT_AREA_ESTIMATES['default'])


def get_zone_base_risk(zone_div: str) -> float:
    """
    구역 타입별 기본 위험도 반환
    
    Args:
        zone_div: spot.csv의 div 값 (confinedSpace, workFloor, restSpace 등)
        
    Returns:
        기본 위험도 (0~1)
    """
    return ZONE_TYPE_BASE_RISK.get(zone_div, ZONE_TYPE_BASE_RISK['etc'])


def load_spot_attributes(spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spot DataFrame에 면적, 기본 위험도 속성 추가
    
    Args:
        spot_df: spot.csv에서 로드한 DataFrame
        
    Returns:
        속성이 추가된 DataFrame
    """
    df = spot_df.copy()
    
    # 면적 추가
    df['area'] = df['spot_no'].apply(get_spot_area)
    
    # 기본 위험도 추가
    df['base_risk'] = df['div'].apply(get_zone_base_risk)
    
    # zone_type (한글 설명)
    zone_type_map = {
        'confinedSpace': '밀폐공간',
        'workFloor': '작업공간',
        'restSpace': '휴게실',
        'smokingArea': '흡연실',
        'etc': '기타',
        'outdoor': '야외'
    }
    df['zone_type_kr'] = df['div'].map(zone_type_map).fillna('기타')
    
    return df


def get_spot_info(spot_no: int, spot_df: pd.DataFrame) -> Optional[Dict]:
    """
    특정 Spot의 속성 정보 반환
    
    Args:
        spot_no: Spot 번호
        spot_df: Spot DataFrame (속성 추가된 것)
        
    Returns:
        Spot 정보 딕셔너리 또는 None
    """
    spot_row = spot_df[spot_df['spot_no'] == spot_no]
    
    if spot_row.empty:
        return None
    
    row = spot_row.iloc[0]
    
    return {
        'spot_no': int(row['spot_no']),
        'name': row['name'],
        'zone_div': row['div'],
        'zone_type_kr': row.get('zone_type_kr', '기타'),
        'area': row.get('area', SPOT_AREA_ESTIMATES['default']),
        'base_risk': row.get('base_risk', ZONE_TYPE_BASE_RISK['etc']),
        'color': row.get('color', ''),
        'description': row.get('description', '')
    }
