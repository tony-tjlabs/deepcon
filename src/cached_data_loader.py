"""
IRFM Demo New - Cached Data Loader
===================================

사전 처리된 캐시 데이터를 로드하는 클래스
IRFM_demo 구조를 참고하되 IRFM_demo_new 데이터 사용
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

# Config import
try:
    import config
except ModuleNotFoundError:
    from src import config


def find_available_datasets(cache_folder = None) -> List[Dict]:
    """사용 가능한 데이터셋 찾기
    
    날짜별 폴더 구조 지원:
    - Cache/20251210/summary.json (새 구조)
    - Cache/summary_20251210.json (기존 구조, 하위 호환)
    
    Returns:
        List of dict with keys: name, date, cache_path, created_at, 
                               t31_records, t41_records, flow_records
    """
    if cache_folder is None:
        cache_folder = Path(__file__).parent.parent / "Cache"
    elif isinstance(cache_folder, str):
        cache_folder = Path(cache_folder)
    
    datasets = []
    
    # 1. 새 구조: 날짜별 폴더 (Cache/YYYYMMDD/summary.json)
    for date_folder in cache_folder.iterdir():
        if date_folder.is_dir() and date_folder.name.isdigit() and len(date_folder.name) == 8:
            summary_file = date_folder / "summary.json"
            # 5분 단위 캐시 parquet 파일 존재 여부 확인
            parquet_5min_1 = date_folder / "flow_cache_5min.parquet"
            parquet_5min_2 = date_folder / f"flow_cache_{date_folder.name}_5min.parquet"
            if summary_file.exists() and (parquet_5min_1.exists() or parquet_5min_2.exists()):
                date_str = date_folder.name
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    datasets.append({
                        'name': f"Yongin_Cluster_{date_str}",
                        'date': date_str,
                        'cache_path': str(date_folder),  # 날짜별 폴더 경로
                        'created_at': summary.get('created_at', ''),
                        't31_records': summary.get('t31', {}).get('total_records', 0),
                        't41_records': summary.get('t41', {}).get('total_records', 0),
                        'flow_records': summary.get('flow', {}).get('total_records', 0)
                    })
                except Exception:
                    pass
    
    # 2. 기존 구조: Cache/summary_YYYYMMDD.json (하위 호환)
    for summary_file in cache_folder.glob("summary_*.json"):
        date_str = summary_file.stem.replace("summary_", "")
        
        # 이미 새 구조에서 추가된 날짜는 스킵
        if any(d['date'] == date_str for d in datasets):
            continue
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            datasets.append({
                'name': f"Yongin_Cluster_{date_str}",
                'date': date_str,
                'cache_path': str(cache_folder),  # 기존 Cache 폴더
                'created_at': summary.get('created_at', ''),
                't31_records': summary.get('t31', {}).get('total_records', 0),
                't41_records': summary.get('t41', {}).get('total_records', 0),
                'flow_records': summary.get('flow', {}).get('total_records', 0)
            })
        except Exception:
            pass
    
    # 날짜 역순 정렬
    datasets.sort(key=lambda x: x['date'], reverse=True)
    
    return datasets

class CachedDataLoader:
    """
    캐시된 분석 데이터 로더

    지원하는 폴더 구조:
    - 새 구조: Cache/YYYYMMDD/파일명.parquet (날짜 suffix 없음)
    - 기존 구조: Cache/파일명_YYYYMMDD.parquet
    """

    def preload_all(self):
        """
        날짜별 주요 parquet 파일을 한 번에 미리 메모리에 올려둠 (최초 1회만)
        대용량 데이터가 많을 때 날짜 변경 시 즉시 반환을 위함
        """
        # 주요 분석 결과 파일명 리스트 (필요시 추가)
        parquet_files = [
                "flow_cache_5min.parquet",
                f"flow_cache_{self.date_str}_5min.parquet",
                f"t31_time_series_{self.date_str}.parquet",
                f"t41_time_series_{self.date_str}.parquet",
                f"flow_results_unit_time_unique_{self.date_str}.parquet",
                f"t31_results_hourly_activity_{self.date_str}.parquet",
                f"t41_results_occupancy_{self.date_str}.parquet",
                f"flow_results_hourly_flow_{self.date_str}.parquet",
                f"flow_results_hourly_avg_from_2min_{self.date_str}.parquet",
                f"t31_results_device_stats_{self.date_str}.parquet",
                f"t41_results_worker_dwell_{self.date_str}.parquet",
                f"t41_results_building_occupancy_{self.date_str}.parquet",
                f"t41_results_space_type_stats_{self.date_str}.parquet",
                f"t41_results_activity_analysis_{self.date_str}.parquet",
                f"t41_results_journey_heatmap_{self.date_str}.parquet",
                f"flow_results_two_min_unique_mac_{self.date_str}.parquet",
                f"t31_results_two_min_unique_mac_{self.date_str}.parquet",
                f"t31_results_operation_heatmap_{self.date_str}.parquet",
                # ... 필요시 추가 ...
        ]
        for fname in parquet_files:
            try:
                self._load_parquet(fname)
            except Exception:
                pass

    def __init__(self, cache_folder: str, date_str: str = None):
        """
        Args:
            cache_folder: Cache 폴더 또는 날짜별 폴더 경로
            date_str: 날짜 문자열 (YYYYMMDD), None이면 자동 탐지
        """
        self.cache_folder = Path(cache_folder)
        
        # 날짜 자동 탐지
        if date_str is None:
            datasets = find_available_datasets(self.cache_folder)
            if datasets:
                date_str = datasets[0]['date']
                # 새 구조인 경우 cache_folder도 업데이트
                self.cache_folder = Path(datasets[0]['cache_path'])
            else:
                date_str = datetime.now().strftime("%Y%m%d")
        
        self.date_str = date_str
        self._cache: Dict[str, Any] = {}
        self._metadata: Optional[Dict] = None
        
        # 새 구조인지 확인
        # 1) 현재 폴더가 날짜 폴더인 경우 (Cache/20251210/)
        # 2) 상위 폴더 아래 날짜 폴더가 있는 경우 (Cache/ -> Cache/20251210/)
        if self.cache_folder.name == date_str:
            self._is_new_structure = True
        else:
            # 날짜별 하위 폴더가 있는지 확인
            date_subfolder = self.cache_folder / date_str
            if date_subfolder.exists() and (date_subfolder / "summary.json").exists():
                self.cache_folder = date_subfolder
                self._is_new_structure = True
            else:
                self._is_new_structure = False
    
    def is_valid(self) -> bool:
        """캐시가 유효한지 확인"""
        if self._is_new_structure:
            summary_path = self.cache_folder / "summary.json"
        else:
            summary_path = self.cache_folder / f"summary_{self.date_str}.json"
        return summary_path.exists()
    
    def get_summary(self) -> Dict:
        """전체 요약 정보 로드"""
        if self._is_new_structure:
            summary_path = self.cache_folder / "summary.json"
        else:
            summary_path = self.cache_folder / f"summary_{self.date_str}.json"
        
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _get_filename(self, base_name: str, extension: str = "parquet") -> str:
        """구조에 따라 파일명 생성"""
        if self._is_new_structure:
            return f"{base_name}.{extension}"
        else:
            return f"{base_name}_{self.date_str}.{extension}"
    
    def _load_parquet(self, filename: str, columns: List[str] = None) -> pd.DataFrame:
        """Parquet 파일 로드 (캐싱)
        
        새 구조일 경우 날짜 suffix를 제거한 파일명으로 시도
        """
        # Cache key needs to include columns to differentiate partial loads
        # If columns is None, load all.
        # But wait, our cache is file-based.
        # If we load partial first, then full load later, we need to handle that.
        # Simple strategy: If columns is specified, don't use the main cache, or use a separate key?
        # Or just read from disk every time if columns is optimized (to save memory)?
        # Let's use a separate cache key if columns are provided.
        
        cache_key = filename
        if columns:
            cache_key = f"{filename}_{tuple(sorted(columns))}"
            
        if cache_key not in self._cache:
            path = self.cache_folder / filename
            
            # 1. 요청된 파일이 없을 경우 (Logic same as before)
            if not path.exists():
                # Case A: 새 구조인데 날짜 suffix가 없는 파일 요청 -> suffix 추가하여 시도
                if self._is_new_structure:
                    if filename.startswith('flow_cache_') and not '20' in filename:
                        prefix, rest = filename.split('flow_cache_', 1)
                        suffixed_name = f"flow_cache_{self.date_str}_{rest}"
                        suffixed_path = self.cache_folder / suffixed_name
                        if suffixed_path.exists():
                            path = suffixed_path
                    
                    if not path.exists():
                        name_parts = filename.rsplit('.', 1)
                        if len(name_parts) == 2:
                            base, ext = name_parts
                            if self.date_str not in base:
                                suffixed_name = f"{base}_{self.date_str}.{ext}"
                                suffixed_path = self.cache_folder / suffixed_name
                                if suffixed_path.exists():
                                    path = suffixed_path
            
                # Case B: 새 구조인데 날짜 suffix가 있는 파일 요청
                    if not path.exists() and self._is_new_structure:
                        new_filename = filename.replace(f"_{self.date_str}", "")
                        path = self.cache_folder / new_filename

                # Case C: Parent Directory Fallback
                if not path.exists() and self._is_new_structure:
                    if 'suffixed_name' in locals():
                        parent_path = self.cache_folder.parent / suffixed_name
                        if parent_path.exists():
                            path = parent_path
                    if not path.exists():
                        parent_path = self.cache_folder.parent / filename
                        if parent_path.exists():
                            path = parent_path
            
            if path.exists():
                # Pass columns to read_parquet for optimization
                self._cache[cache_key] = pd.read_parquet(path, columns=columns)
            else:
                self._cache[cache_key] = pd.DataFrame()
        return self._cache[cache_key].copy()
    
    # ========================================================================
    # Raw Data (flow_cache)
    # ========================================================================
    
    def load_flow_cache(self, resolution: str = '5min', columns: List[str] = None) -> pd.DataFrame:
        """Flow cache 로드 (원본 데이터)

        Args:
            resolution: '5min' (default) or '1min'
            columns: List of columns to read (Optimization)
        """
        res = str(resolution)
        if res not in ('5min', '1min'):
            res = '5min'

        tried = []
        if self._is_new_structure:
            filename1 = f"flow_cache_{res}.parquet"
            filename2 = f"flow_cache_{self.date_str}_{res}.parquet"
            # 우선 filename1, 없으면 filename2
            try:
                return self._load_parquet(filename1, columns=columns)
            except Exception as e1:
                tried.append(str(e1))
                try:
                    return self._load_parquet(filename2, columns=columns)
                except Exception as e2:
                    tried.append(str(e2))
                    print(f"[load_flow_cache] Tried {filename1}, {filename2} but failed: {tried}")
                    return pd.DataFrame()
        else:
            filename = f"flow_cache_{self.date_str}_{res}.parquet"
            try:
                return self._load_parquet(filename, columns=columns)
            except Exception as e:
                print(f"[load_flow_cache] Tried {filename} but failed: {e}")
                return pd.DataFrame()
    
    def load_t31_data(self) -> pd.DataFrame:
        """Type31 데이터만 필터링 (캐시됨)"""
        cache_key = f"_t31_filtered_{self.date_str}"
        if cache_key not in self._cache:
            df = self.load_flow_cache()
            if df.empty:
                self._cache[cache_key] = df
            else:
                self._cache[cache_key] = df[df['type'] == config.TYPE_31_EQUIPMENT].copy()
        return self._cache[cache_key].copy()
    
    def load_t41_data(self) -> pd.DataFrame:
        """Type41 데이터만 필터링 (캐시됨)"""
        cache_key = f"_t41_filtered_{self.date_str}"
        if cache_key not in self._cache:
            df = self.load_flow_cache()
            if df.empty:
                self._cache[cache_key] = df
            else:
                self._cache[cache_key] = df[df['type'] == config.TYPE_41_WORKER].copy()
        return self._cache[cache_key].copy()
    
    def load_flow_data(self) -> pd.DataFrame:
        """Flow 데이터만 필터링 (Type 1, 10) (캐시됨)"""
        cache_key = f"_flow_filtered_{self.date_str}"
        if cache_key not in self._cache:
            df = self.load_flow_cache()
            if df.empty:
                self._cache[cache_key] = df
            else:
                self._cache[cache_key] = df[df['type'].isin([config.TYPE_10_ANDROID, config.TYPE_1_IPHONE])].copy()
        return self._cache[cache_key].copy()
    
    # ========================================================================
    # T31 (장비) 분석 결과
    # ========================================================================
    
    def load_t31_hourly_activity(self) -> pd.DataFrame:
        """T31 시간대별 활동"""
        return self._load_parquet(f"t31_results_hourly_activity_{self.date_str}.parquet")
    
    def load_t31_device_stats(self) -> pd.DataFrame:
        """T31 장비별 통계"""
        return self._load_parquet(f"t31_results_device_stats_{self.date_str}.parquet")
    
    def load_t31_sward_activity(self) -> pd.DataFrame:
        """T31 S-Ward별 활동"""
        return self._load_parquet(f"t31_results_sward_activity_{self.date_str}.parquet")
    
    def load_t31_two_min_unique(self) -> pd.DataFrame:
        """T31 2분 단위 unique MAC"""
        return self._load_parquet(f"t31_results_two_min_unique_mac_{self.date_str}.parquet")
    
    def load_t31_operation_heatmap(self) -> pd.DataFrame:
        """T31 Operation Heatmap (10분 단위)"""
        return self._load_parquet(f"t31_results_operation_heatmap_{self.date_str}.parquet")
    
    # ========================================================================
    # T41 (작업자) 분석 결과
    # ========================================================================
    
    def load_t41_occupancy(self) -> pd.DataFrame:
        """T41 시간대별 작업자 수"""
        return self._load_parquet(f"t41_results_occupancy_{self.date_str}.parquet")
    
    def load_t41_worker_dwell(self) -> pd.DataFrame:
        """T41 작업자별 체류시간"""
        # Wrap parquet loader and normalize return to a clean DataFrame.
        try:
            df = self._load_parquet(f"t41_results_worker_dwell_{self.date_str}.parquet")
        except Exception:
            return pd.DataFrame()

        # If parquet loader returned non-DataFrame (defensive), try to normalize
        if isinstance(df, dict) or isinstance(df, list):
            try:
                df = pd.json_normalize(df)
            except Exception:
                try:
                    df = pd.DataFrame(df)
                except Exception:
                    return pd.DataFrame()

        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        # Ensure expected columns exist; try to detect common variants
        cols = set(df.columns.astype(str))
        # Normalize column names to lower-case for matching
        col_map = {c: c for c in df.columns}
        lower_map = {c.lower(): c for c in df.columns}

        # map common names
        if 'dwell_minutes' not in cols:
            for alt in ('dwell_min', 'dwell', 'avg_dwell_minutes', 'avg_dwell'):
                if alt in lower_map:
                    col_map[lower_map[alt]] = 'dwell_minutes'
                    df = df.rename(columns={lower_map[alt]: 'dwell_minutes'})
                    break

        if 'spot_no' not in cols and 'spot_nos' in lower_map:
            df = df.rename(columns={lower_map['spot_nos']: 'spot_no'})

        # If zone name is present in other key, keep as-is; otherwise no-op
        # Coerce dwell_minutes to numeric and drop rows without numeric dwell
        if 'dwell_minutes' in df.columns:
            df['dwell_minutes'] = pd.to_numeric(df['dwell_minutes'], errors='coerce')
        else:
            # If no dwell column, create placeholder from other duration-like columns
            df['dwell_minutes'] = pd.to_numeric(df.iloc[:, 0], errors='coerce') if not df.empty else pd.Series(dtype=float)

        # Drop rows with NaN dwell_minutes
        df = df.dropna(subset=['dwell_minutes'])

        # If spot_no exists but as list-like strings, normalize to integers where possible
        if 'spot_no' in df.columns:
            try:
                # handle comma-separated strings
                if df['spot_no'].dtype == object:
                    # take first spot if multiple
                    df['spot_no'] = df['spot_no'].astype(str).str.split(',').str[0]
                df['spot_no'] = pd.to_numeric(df['spot_no'], errors='coerce').dropna().astype(int)
            except Exception:
                pass

        # Defensive: if any column contains dict objects (unhashable), convert them to JSON strings
        try:
            import json as _json
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, dict)).any():
                    df[col] = df[col].apply(lambda x: _json.dumps(x) if isinstance(x, dict) else x)
        except Exception:
            pass

        return df.reset_index(drop=True)
    
    def load_t41_building_occupancy(self) -> pd.DataFrame:
        """T41 Building/Level별 작업자 수"""
        return self._load_parquet(f"t41_results_building_occupancy_{self.date_str}.parquet")
    
    def load_t41_space_type_stats(self) -> pd.DataFrame:
        """T41 공간 유형별 통계"""
        return self._load_parquet(f"t41_results_space_type_stats_{self.date_str}.parquet")
    
    def load_t41_activity_analysis(self) -> pd.DataFrame:
        """T41 Activity Analysis (MAC별 시간대별 활동)"""
        return self._load_parquet(f"t41_results_activity_analysis_{self.date_str}.parquet")
    
    def load_t41_journey_heatmap(self) -> pd.DataFrame:
        """T41 Journey Heatmap (10분 단위)"""
        return self._load_parquet(f"t41_results_journey_heatmap_{self.date_str}.parquet")
    
    # ========================================================================
    # Flow (모바일) 분석 결과
    # ========================================================================
    
    def load_flow_unit_time_unique(self) -> pd.DataFrame:
        """Flow UnitTime 단위 unique devices"""
        return self._load_parquet(f"flow_results_unit_time_unique_{self.date_str}.parquet")
    
    def load_flow_two_min_unique(self) -> pd.DataFrame:
        """Flow 2분 단위 unique MAC"""
        return self._load_parquet(f"flow_results_two_min_unique_mac_{self.date_str}.parquet")
    
    def load_flow_hourly_avg_from_2min(self) -> pd.DataFrame:
        """Flow 시간대별 평균 (2분 bins의 평균)"""
        return self._load_parquet(f"flow_results_hourly_avg_from_2min_{self.date_str}.parquet")
    
    def load_flow_hourly_flow(self) -> pd.DataFrame:
        """Flow 시간대별 유동인구"""
        return self._load_parquet(f"flow_results_hourly_flow_{self.date_str}.parquet")
    
    def load_flow_sward_flow(self) -> pd.DataFrame:
        """Flow S-Ward별 유동인구"""
        return self._load_parquet(f"flow_results_sward_flow_{self.date_str}.parquet")
    
    def load_flow_device_stats(self) -> pd.DataFrame:
        """Flow 디바이스별 체류 분석"""
        return self._load_parquet(f"flow_results_device_stats_{self.date_str}.parquet")
    
    def load_flow_device_type_stats(self) -> pd.DataFrame:
        """Flow 디바이스 타입별 분석"""
        return self._load_parquet(f"flow_results_device_type_stats_{self.date_str}.parquet")
    
    def load_flow_building_type_distribution(self) -> pd.DataFrame:
        """Flow Building/Floor별 디바이스 타입 분포"""
        return self._load_parquet(f"flow_results_building_type_distribution_{self.date_str}.parquet")
    
    # ========================================================================
    # 이름 매핑 (Sector/Building/Floor/Spot)
    # ========================================================================
    
    def get_name_mappings(self) -> Dict[str, Dict]:
        """Sector/Building/Floor/Spot 번호 → 이름 매핑 반환
        
        Returns:
            Dict with keys: sector, building, floor, spot
            각 값은 {no: name} 형태의 딕셔너리
        """
        if '_name_mappings' not in self._cache:
            mappings = {
                'sector': {},
                'building': {},
                'floor': {},
                'spot': {}
            }
            
            try:
                # Sector
                sector_df = pd.read_csv(config.SECTOR_CSV)
                mappings['sector'] = dict(zip(sector_df['sector_no'], sector_df['name']))
                
                # Building
                building_df = pd.read_csv(config.BUILDING_CSV)
                mappings['building'] = dict(zip(building_df['building_no'], building_df['name']))
                # 실외(0)는 명시적으로 추가
                mappings['building'][0] = '실외'
                
                # Floor
                floor_df = pd.read_csv(config.FLOOR_CSV)
                # floor_no → name 매핑 (building_no 포함하여 고유하게)
                for _, row in floor_df.iterrows():
                    floor_no = row['floor_no']
                    floor_name = row['name'] if pd.notna(row['name']) else f'Floor {floor_no}'
                    mappings['floor'][floor_no] = floor_name
                # 실외(0)는 명시적으로 추가
                mappings['floor'][0] = '실외'
                
                # Spot
                spot_df = pd.read_csv(config.SPOT_CSV)
                mappings['spot'] = dict(zip(spot_df['spot_no'], spot_df['name']))
                
            except Exception as e:
                print(f"Warning: 이름 매핑 로드 실패: {e}")
            
            self._cache['_name_mappings'] = mappings
        
        return self._cache['_name_mappings']
    
    def get_sector_names(self) -> Dict[int, str]:
        """Sector 번호 → 이름 매핑"""
        return self.get_name_mappings()['sector']
    
    def get_building_names(self) -> Dict[int, str]:
        """Building 번호 → 이름 매핑"""
        return self.get_name_mappings()['building']
    
    def get_floor_names(self) -> Dict[int, str]:
        """Floor 번호 → 이름 매핑"""
        return self.get_name_mappings()['floor']
    
    def get_spot_names(self) -> Dict[int, str]:
        """Spot 번호 → 이름 매핑"""
        return self.get_name_mappings()['spot']
    
    def get_hierarchy_structure(self) -> Dict:
        """Sector → Building → Floor 계층 구조 반환
        
        Returns:
            {
                'sectors': {'Y-Project': 22, ...},
                'buildings': {'All': None, 'WWT': 1, 'FAB': 2, ...},
                'building_floors': {
                    1: ['All', 'B1MF', 'B1F', '1F', '2F', ...],  # WWT
                    2: ['All', '1F', '4F'],  # FAB
                    ...
                },
                'floor_mapping': {
                    (1, 'B1MF'): 1,  # (building_no, floor_name) → floor_no
                    ...
                },
                'spots': {'WWT B1F (전체)': 106, 'TB16': 117, ...}
            }
        """
        if '_hierarchy' not in self._cache:
            hierarchy = {
                'sectors': {},
                'buildings': {'All': None},
                'building_floors': {},
                'floor_mapping': {},
                'spots': {'All': None}
            }
            
            try:
                # Sector
                sector_df = pd.read_csv(config.SECTOR_CSV)
                for _, row in sector_df.iterrows():
                    hierarchy['sectors'][row['name']] = row['sector_no']
                
                # Building
                building_df = pd.read_csv(config.BUILDING_CSV)
                for _, row in building_df.iterrows():
                    hierarchy['buildings'][row['name']] = row['building_no']
                    hierarchy['building_floors'][row['building_no']] = ['All']
                
                # 실외 추가
                hierarchy['buildings']['실외'] = 0
                hierarchy['building_floors'][0] = ['All', '실외']
                
                # Floor (Building별로 그룹핑)
                floor_df = pd.read_csv(config.FLOOR_CSV)
                for _, row in floor_df.iterrows():
                    bld_no = row['building_no']
                    floor_name = row['name'] if pd.notna(row['name']) else f"Floor {row['floor_no']}"
                    floor_no = row['floor_no']
                    
                    if bld_no in hierarchy['building_floors']:
                        if floor_name not in hierarchy['building_floors'][bld_no]:
                            hierarchy['building_floors'][bld_no].append(floor_name)
                    
                    hierarchy['floor_mapping'][(bld_no, floor_name)] = floor_no
                
                # Spot
                spot_df = pd.read_csv(config.SPOT_CSV)
                for _, row in spot_df.iterrows():
                    hierarchy['spots'][row['name']] = row['spot_no']
                
            except Exception as e:
                print(f"Warning: 계층 구조 로드 실패: {e}")
            
            self._cache['_hierarchy'] = hierarchy
        
        return self._cache['_hierarchy']
    
    def get_building_list(self) -> List[str]:
        """Building 목록 반환 (All 포함)"""
        return list(self.get_hierarchy_structure()['buildings'].keys())
    
    def get_floor_list(self, building_name: str = None) -> List[str]:
        """Floor 목록 반환 (Building에 따라 필터링)"""
        hierarchy = self.get_hierarchy_structure()
        
        if building_name is None or building_name == 'All':
            # 모든 Floor 반환
            all_floors = ['All']
            for floors in hierarchy['building_floors'].values():
                for f in floors:
                    if f not in all_floors:
                        all_floors.append(f)
            return all_floors
        
        building_no = hierarchy['buildings'].get(building_name)
        if building_no is not None and building_no in hierarchy['building_floors']:
            return hierarchy['building_floors'][building_no]
        
        return ['All']
    
    def get_spot_list(self) -> List[str]:
        """Spot 목록 반환 (All 포함)"""
        return list(self.get_hierarchy_structure()['spots'].keys())
    
    def filter_by_location(self, df: pd.DataFrame, building_name: str = 'All', 
                           floor_name: str = 'All', spot_name: str = 'All') -> pd.DataFrame:
        """Building/Floor/Spot 기준으로 데이터 필터링"""
        if df is None or df.empty:
            return df
        
        hierarchy = self.get_hierarchy_structure()
        result = df.copy()
        
        # Building 필터
        if building_name != 'All':
            building_no = hierarchy['buildings'].get(building_name)
            if building_no is not None and 'building_no' in result.columns:
                result = result[result['building_no'] == building_no]
        
        # Floor 필터
        if floor_name != 'All':
            building_no = hierarchy['buildings'].get(building_name) if building_name != 'All' else None
            
            if building_no is not None:
                floor_no = hierarchy['floor_mapping'].get((building_no, floor_name))
            else:
                # Building이 All인 경우, floor_name으로 직접 매칭
                floor_no = None
                for (bld, fname), fno in hierarchy['floor_mapping'].items():
                    if fname == floor_name:
                        floor_no = fno
                        break
            
            if floor_no is not None and 'floor_no' in result.columns:
                result = result[result['floor_no'] == floor_no]
        
        # Spot 필터
        if spot_name != 'All':
            spot_no = hierarchy['spots'].get(spot_name)
            if spot_no is not None and 'spot_nos' in result.columns:
                # spot_nos는 문자열 (콤마 구분)
                result = result[result['spot_nos'].str.contains(str(spot_no), na=False)]
        
        return result
    
    # ========================================================================
    # 호환성 메서드 (IRFM_demo 스타일)
    # ========================================================================
    
    def has_raw_data(self) -> Dict[str, bool]:
        """Raw 데이터 존재 여부"""
        return {
            't31': (self.cache_folder / f"flow_cache_{self.date_str}_5min.parquet").exists(),
            't41': (self.cache_folder / f"flow_cache_{self.date_str}_5min.parquet").exists(),
            'flow': (self.cache_folder / f"flow_cache_{self.date_str}_5min.parquet").exists(),
            'sward_config': False  # IRFM_demo_new에는 별도 파일 없음
        }
    
    def load_raw_t31(self) -> pd.DataFrame:
        """Raw T31 데이터 로드"""
        return self.load_t31_data()
    
    def load_raw_t41(self) -> pd.DataFrame:
        """Raw T41 데이터 로드"""
        return self.load_t41_data()
    
    def load_raw_flow(self) -> pd.DataFrame:
        """Raw Flow 데이터 로드"""
        return self.load_flow_data()
    
    # ========================================================================
    # 통합 Dashboard 캐시 (v3.0)
    # ========================================================================
    
    def load_t31_time_series(self) -> pd.DataFrame:
        """T31 5분 단위 시계열 데이터
        
        Columns: time_index, active_devices, hour, time_label, total_devices
        """
        return self._load_parquet(f"t31_time_series_{self.date_str}.parquet")
    
    def load_t31_hourly(self) -> pd.DataFrame:
        """T31 시간별 집계
        
        Columns: hour, unique_devices, max_concurrent, avg_concurrent
        """
        return self._load_parquet(f"t31_hourly_{self.date_str}.parquet")
    
    def load_t31_building(self) -> pd.DataFrame:
        """T31 Building별 장비 분포 (주 Building 기준)
        
        Columns: building_no, device_count, building_name, total_records
        """
        return self._load_parquet(f"t31_building_{self.date_str}.parquet")
    
    def load_t41_time_series(self) -> pd.DataFrame:
        """T41 5분 단위 시계열 데이터
        
        Columns: time_index, active_workers, inactive_workers, worker_count, hour, time_label
        """
        return self._load_parquet(f"t41_time_series_{self.date_str}.parquet")
    
    def load_t41_hourly(self) -> pd.DataFrame:
        """T41 시간별 집계
        
        Columns: hour, unique_workers, max_active, avg_active, max_inactive, avg_inactive
        """
        return self._load_parquet(f"t41_hourly_{self.date_str}.parquet")
    
    def load_t41_building(self) -> pd.DataFrame:
        """T41 Building별 작업자 분포 (주 Building, 주 상태 기준)
        
        Columns: building_no, active_count, inactive_count, total_count, building_name
        """
        return self._load_parquet(f"t41_building_{self.date_str}.parquet")
    
    def load_t41_building_time(self) -> pd.DataFrame:
        """T41 시간대별 Building 누적 방문
        
        Columns: building_no, active_visitors, inactive_visitors, unique_visitors,
                 time_start, time_end, period_name, building_name
        """
        return self._load_parquet(f"t41_building_time_{self.date_str}.parquet")
    
    def get_unified_summary(self) -> Dict:
        """통합 Summary (v3.0) 로드
        
        summary.json에서 t31, t41, mobile 요약 정보 반환
        - t31: total_devices, work_hour_devices, work_hour_rate, max_concurrent, avg_concurrent
        - t41: total_workers, primary_active, primary_inactive, max_active, avg_active
        - mobile: total_devices, android_devices, iphone_devices, max_concurrent
        """
        return self.get_summary()
