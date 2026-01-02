"""
IRFM Configuration
==================

데이터 처리를 위한 설정 파일
"""

from pathlib import Path

# ============================================================================
# 경로 설정
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'Datafile' / 'Yongin_Cluster_202512010'
CACHE_DIR = PROJECT_ROOT / 'Cache'
CACHE_DIR.mkdir(exist_ok=True)

RAW_DATA_FILE = DATA_DIR / 'Raw_data_10s_20251210.csv'

# CSV 파일들
GATEWAY_CSV = DATA_DIR / 'gateway.csv'
FLOOR_CSV = DATA_DIR / 'floor.csv'
BUILDING_CSV = DATA_DIR / 'building.csv'
SECTOR_CSV = DATA_DIR / 'sector.csv'
SPOT_CSV = DATA_DIR / 'spot.csv'
SPOT_POSITION_CSV = DATA_DIR / 'spot_position.csv'
IRFM_CSV = DATA_DIR / 'irfm.csv'

# ============================================================================
# 시간 설정
# ============================================================================

# 데이터 수집 간격 (초)
DATA_COLLECTION_INTERVAL = 10  # 10초마다 데이터 수집

# 시간 동기화 반올림 단위 (초)
TIME_ROUNDING = 10  # 10초 단위로 반올림

# 기본 처리 시간 단위 (초)
BASE_TIME_UNIT = 60  # 1분 단위로 1차 처리

# 최종 결과 시간 단위 (초) - UnitTime
UNIT_TIME = 300  # 5분 = 300초

# 하루 시간 인덱스 개수
TIME_INDICES_PER_DAY = (24 * 60 * 60) // UNIT_TIME  # 288개 (5분 단위)

# ============================================================================
# 디바이스 타입
# ============================================================================

# Type 31: 장비 (Table Lift)
TYPE_31_EQUIPMENT = 31

# Type 41: 작업자 (Worker Helmet)
TYPE_41_WORKER = 41

# Type 10: Android
TYPE_10_ANDROID = 10

# Type 1: iPhone
TYPE_1_IPHONE = 1

# ============================================================================
# 활성/비활성 판정 기준
# ============================================================================

# Type 31 (장비)
# - 진동 감지 시: 1분마다 신호 발신 (ttag_work_status=1)
# - 진동 미감지 시: 신호 발신 안 함
T31_ACTIVE_THRESHOLD = 1  # 1분 내 1회 이상 수신 시 활성
T31_SMOOTHING_OLD = 0.95  # 기존 위치 가중치 (매우 낮은 다이나믹스)
T31_SMOOTHING_NEW = 0.05  # 새 위치 가중치
T31_USE_WORK_STATUS = True  # ttag_work_status 컬럼 사용 여부

# Type 41 (작업자)
# - 진동 감지 시: 10초마다 신호 발신
# - 진동 미감지 시: 1분마다 신호 발신
T41_ACTIVE_THRESHOLD = 2  # 1분 내 2회 이상 수신 시 활성
T41_INACTIVE_THRESHOLD = 1  # 1분 내 1회 수신 시 비활성
# 0회는 미감지

# UnitTime(5분) 판정
UNITTIME_ACTIVE_THRESHOLD = 1  # UnitTime 내 1회 이상 활성이면 최종 활성

# ============================================================================
# 좌표계 설정
# ============================================================================

# 실외/실내 구분
OUTDOOR_FLOOR_NO = 0  # floor_no = 0이면 실외
OUTDOOR_BUILDING_NO = 0  # building_no = 0이면 실외

# ============================================================================
# RSSI 설정
# ============================================================================

# RSSI 임계값 (필요시 사용)
RSSI_THRESHOLD = -80
# RSSI 집계 방식 (per 1-minute group): 'median' | 'mean' | 'weighted'
RSSI_AGG_METHOD = 'mean'
# 시간 가중 평균용 타우 (초) - weighted 모드에서 사용
RSSI_DECAY_TAU = 20.0
# ---------------------------------------------------------------------------
# Log-distance (RSSI->distance) model parameters for simple inverse-square
# weighting (used by position estimator): d = 10^((A - r)/(10*n))
# - RSSI_REF_AT_1M: RSSI (dBm) measured at 1 meter (environment dependent)
# - PATH_LOSS_EXP: path-loss exponent (2.0 free-space, higher for indoor)
# - RSSI_CLIP_MIN/MAX: clip bounds for RSSI before computing distance
# - MIN_DISTANCE: lower bound on computed distance to avoid infinite weight
RSSI_REF_AT_1M = -41.0
PATH_LOSS_EXP = 2.0
RSSI_CLIP_MIN = -120.0
RSSI_CLIP_MAX = -30.0
MIN_DISTANCE = 0.5

# Randomization parameters for low-candidate scenarios
# When only one gateway is observed, place a random point around the GW.
# SINGLE_GW_RANDOM_FACTOR: fraction of estimated distance used as stddev for radial noise
SINGLE_GW_RANDOM_FACTOR = 0.3
# Minimum stddev (meters) for single-GW randomization
SINGLE_GW_RANDOM_MIN = 0.5

# When two gateways are observed, add perpendicular jitter to avoid colinear points.
# TWO_GW_JITTER_STD: standard deviation (meters) of perpendicular noise
TWO_GW_JITTER_STD = 0.5

# ============================================================================
# 캐시 데이터 설정
# ============================================================================

# 캐시 파일 포맷
CACHE_FORMAT = 'parquet'  # 'parquet' or 'hdf5' or 'csv'

# 캐시 파일명 패턴
CACHE_FILE_PATTERN = 'flow_cache_{date}_{unit_time}min.{format}'

# ============================================================================
# 데이터 컬럼명
# ============================================================================

# Raw 데이터 컬럼
RAW_COLUMNS = {
    'time': 'dt',                                    # 시간 (HH:MM:SS)
    'seq_no': 'flow_device_data_10s_no',            # 일련번호
    'gateway': 'gateway_no',                         # Gateway 번호
    'mac': 'mac_address',                            # MAC 주소
    'rssi': 'rssi',                                  # 수신신호세기
    'type': 'type',                                  # 디바이스 타입
    'apple_type': 'apple_device_type_code',          # Apple 디바이스 타입
    'work_status': 'ttag_work_status',               # T-Ward 진동 상태 (1=진동감지, 0=미감지)
    'hitting_count': 'apple_device_hitting_count',   # Apple 히팅 카운트
    'battery': 'ttag_batt',                          # T-Ward 배터리 전압
    'pressure': 'pressure',                          # T-Ward 기압센서
    'acceleration': 'max_av',                        # T-Ward 가속도 값
}

# Raw 데이터 설명
RAW_COLUMNS_DESC = {
    'dt': '시간 (HH:MM:SS) - 0시~23:59:52',
    'flow_device_data_10s_no': '일련번호 (계속 1씩 증가)',
    'gateway_no': 'Gateway 번호 (신호 수신한 Gateway)',
    'mac_address': '사용자 디바이스 MAC 주소',
    'rssi': '수신신호세기',
    'type': '디바이스 타입 (41=작업자, 31=장비, 10=Android, 1=iPhone)',
    'apple_device_type_code': 'Apple 디바이스 타입 코드',
    'ttag_work_status': 'T-Ward 진동 상태 (1=진동감지, 0=미감지)',
    'apple_device_hitting_count': 'Apple 신호 히팅 카운트',
    'ttag_batt': 'T-Ward 배터리 전압',
    'pressure': 'T-Ward 기압센서 측정치',
    'max_av': 'T-Ward 가속도 값',
}

# 캐시 데이터 컬럼
CACHE_COLUMNS = [
    'time_index',      # 시간 인덱스 (1~288)
    'mac_address',     # MAC 주소
    'type',            # 디바이스 타입 (31, 41, 10, 1)
    'floor_no',        # Floor 번호 (0=실외)
    'building_no',     # Building 번호 (0=실외)
    'sector_no',       # Sector 번호
    'x',               # X 좌표
    'y',               # Y 좌표
    'status',          # 활성(1) / 비활성(0)
    'spot_nos',        # Spot 번호 리스트 (복수 가능)
    'snapped',         # 위치가 폴리곤 내부로 스냅 되었는지 (True/False)
    'position_confidence',  # 위치 신뢰도 (0.0-1.0)
    # diagnostic columns (added temporarily for analysis)
    'used_gw_count',   # unique gateways observed in the 5-min window
    'max_min_gw_count',# max distinct gateways observed in any 1-min within the 5-min window
]

# ============================================================================
# 날짜 설정
# ============================================================================

# 분석 대상 날짜
TARGET_DATE = '2025-12-10'

# ============================================================================
# 로깅 설정
# ============================================================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# 병렬 처리 설정
# ============================================================================

# 청크 크기 (메모리 효율)
CHUNK_SIZE = 100000

# 병렬 처리 워커 수
N_WORKERS = 4

# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_time_index(hour: int, minute: int) -> int:
    """시간을 time_index로 변환 (1~288)
    
    Args:
        hour: 시간 (0~23)
        minute: 분 (0~59)
        
    Returns:
        int: time_index (1~288)
    """
    total_minutes = hour * 60 + minute
    time_index = (total_minutes // (UNIT_TIME // 60)) + 1
    return time_index

def get_time_from_index(time_index: int) -> tuple:
    """time_index를 시간으로 변환
    
    Args:
        time_index: 시간 인덱스 (1~288)
        
    Returns:
        tuple: (hour, minute)
    """
    total_minutes = (time_index - 1) * (UNIT_TIME // 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return hour, minute

def round_time_to_unit(seconds: int) -> int:
    """시간을 TIME_ROUNDING 단위로 반올림
    
    Args:
        seconds: 초
        
    Returns:
        int: 반올림된 초
    """
    return round(seconds / TIME_ROUNDING) * TIME_ROUNDING

# ============================================================================
# 설정 검증
# ============================================================================

def validate_config():
    """설정 유효성 검증"""
    assert DATA_DIR.exists(), f"데이터 디렉토리 없음: {DATA_DIR}"
    assert RAW_DATA_FILE.exists(), f"Raw 데이터 파일 없음: {RAW_DATA_FILE}"
    assert UNIT_TIME % BASE_TIME_UNIT == 0, "UNIT_TIME은 BASE_TIME_UNIT의 배수여야 함"
    assert UNIT_TIME >= BASE_TIME_UNIT, "UNIT_TIME은 BASE_TIME_UNIT보다 커야 함"
    assert TIME_INDICES_PER_DAY == 288, f"하루 인덱스 개수 오류: {TIME_INDICES_PER_DAY}"
    print("✅ 설정 검증 완료")

if __name__ == "__main__":
    validate_config()
    
    print("\n" + "=" * 60)
    print("IRFM Configuration")
    print("=" * 60)
    print(f"\n📁 프로젝트 경로: {PROJECT_ROOT}")
    print(f"📁 데이터 경로: {DATA_DIR}")
    print(f"📁 캐시 경로: {CACHE_DIR}")
    
    print(f"\n⏰ 시간 설정:")
    print(f"  - 데이터 수집 간격: {DATA_COLLECTION_INTERVAL}초")
    print(f"  - 시간 반올림 단위: {TIME_ROUNDING}초")
    print(f"  - 기본 처리 단위: {BASE_TIME_UNIT}초 (1분)")
    print(f"  - UnitTime: {UNIT_TIME}초 ({UNIT_TIME//60}분)")
    print(f"  - 하루 인덱스 개수: {TIME_INDICES_PER_DAY}개")
    
    print(f"\n📊 디바이스 타입:")
    print(f"  - Type 31: 장비 (Equipment)")
    print(f"  - Type 41: 작업자 (Worker)")
    print(f"  - Type 10: Android")
    print(f"  - Type 1: iPhone")
    
    print(f"\n✅ 활성/비활성 기준:")
    print(f"  - T31: {T31_ACTIVE_THRESHOLD}회 이상 → 활성")
    print(f"  - T41: {T41_ACTIVE_THRESHOLD}회 이상 → 활성, {T41_INACTIVE_THRESHOLD}회 → 비활성")
    print(f"  - UnitTime: {UNITTIME_ACTIVE_THRESHOLD}회 이상 활성 → 최종 활성")
    
    print(f"\n📍 좌표계:")
    print(f"  - 실외: floor_no={OUTDOOR_FLOOR_NO}, building_no={OUTDOOR_BUILDING_NO}")
    print(f"  - 실내: floor_no>0, building_no>0")
    
    print(f"\n💾 캐시 설정:")
    print(f"  - 포맷: {CACHE_FORMAT}")
    print(f"  - 패턴: {CACHE_FILE_PATTERN}")
    
    print("\n" + "=" * 60)
