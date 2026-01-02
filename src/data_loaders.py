"""
Unified Data Loading Utilities
===============================
Centralized data loading with caching to eliminate duplicate extraction logic
"""
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict


# Data paths configuration
DATA_FOLDER = Path('/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010')


class DataLoader:
    """Centralized data loader to eliminate duplicate loading logic"""
    
    def __init__(self, data_folder: Path = DATA_FOLDER):
        self.data_folder = data_folder
    
    def load_spot_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load spot and spot_position data together
        
        Returns:
            Tuple of (spot_df, spot_position_df)
        """
        spot_path = self.data_folder / 'spot.csv'
        spot_pos_path = self.data_folder / 'spot_position.csv'
        
        if spot_path.exists() and spot_pos_path.exists():
            return pd.read_csv(spot_path), pd.read_csv(spot_pos_path)
        return pd.DataFrame(), pd.DataFrame()
    
    def load_spot_info(self) -> pd.DataFrame:
        """Load spot information (spot.csv only)"""
        spot_path = self.data_folder / 'spot.csv'
        if spot_path.exists():
            return pd.read_csv(spot_path)
        return pd.DataFrame()
    
    def load_spot_position(self) -> pd.DataFrame:
        """Load spot position information (spot_position.csv only)"""
        spot_pos_path = self.data_folder / 'spot_position.csv'
        if spot_pos_path.exists():
            return pd.read_csv(spot_pos_path)
        return pd.DataFrame()
    
    def load_floor_info(self) -> pd.DataFrame:
        """
        Load floor metadata from irfm.csv
        
        Returns:
            DataFrame with columns: floor_number, building_number, floor_name, length_x, length_y
        """
        irfm_path = self.data_folder / 'irfm.csv'
        if irfm_path.exists():
            df = pd.read_csv(irfm_path)
            return df[['floor_number', 'building_number', 'floor_name', 'length_x', 'length_y']].copy()
        return pd.DataFrame()
    
    def load_gateway_info(self) -> pd.DataFrame:
        """Load raw gateway information"""
        gateway_path = self.data_folder / 'gateway.csv'
        if gateway_path.exists():
            return pd.read_csv(gateway_path)
        return pd.DataFrame()
    
    def load_outdoor_gateways(self) -> pd.DataFrame:
        """
        Load outdoor gateways (floor_no is NaN, with valid coordinates)
        
        Returns:
            DataFrame with columns: gateway_no, name, location_x, location_y
        """
        gw_df = self.load_gateway_info()
        if gw_df.empty:
            return pd.DataFrame()
        
        outdoor_gw = gw_df[
            gw_df['floor_no'].isna() & 
            gw_df['location_x'].notna() & 
            gw_df['location_y'].notna()
        ][['gateway_no', 'name', 'location_x', 'location_y']].copy()
        
        return outdoor_gw
    
    def load_indoor_gateways(self, building_no: int, floor_no: int) -> pd.DataFrame:
        """
        Load indoor gateways for specific building and floor
        
        Args:
            building_no: Building number (not used in current logic, but kept for API consistency)
            floor_no: Floor number (global ID from irfm.csv)
            
        Returns:
            DataFrame with columns: gateway_no, name, location_x, location_y
        """
        gw_df = self.load_gateway_info()
        if gw_df.empty:
            return pd.DataFrame()
        
        indoor_gw = gw_df[
            (gw_df['floor_no'] == floor_no) &
            gw_df['location_x'].notna() & 
            gw_df['location_y'].notna()
        ][['gateway_no', 'name', 'location_x', 'location_y']].copy()
        
        return indoor_gw
    
    def load_flow_cache(self, cache_folder: str, date_str: str, resolution: str = '5min') -> pd.DataFrame:
        """
        Load flow cache from CachedDataLoader
        
        Args:
            cache_folder: Path to cache folder
            date_str: Date string (YYYYMMDD format)
            resolution: '5min' or '1min' aggregation level
            
        Returns:
            Flow cache DataFrame
        """
        from src.cached_data_loader import CachedDataLoader
        loader = CachedDataLoader(cache_folder, date_str)
        return loader.load_flow_cache(resolution)
    
    def load_location_cache(
        self, 
        cache_path: str, 
        date_str: str, 
        building_no: int, 
        floor_no: Optional[int] = None
    ) -> Dict:
        """
        Load optimized split location cache
        
        Args:
            cache_path: Base cache path
            date_str: Date string (YYYYMMDD format)
            building_no: 0 for outdoor, else building number
            floor_no: Floor number (required if building_no != 0)
            
        Returns:
            Dictionary mapping worker IDs to location data
        """
        try:
            # Cache 루트의 공통 location_maps 폴더 사용
            base_dir = Path(cache_path) / "location_maps"
            
            if building_no == 0:
                fname = "outdoor.json"
            else:
                if floor_no is None:
                    raise ValueError("floor_no is required when building_no != 0")
                fname = f"{building_no}_{floor_no}.json"
            
            fpath = base_dir / fname
            if fpath.exists():
                with open(fpath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading location cache for {building_no}_{floor_no}: {e}")
            return {}


# Global singleton instance
_default_loader = None


def get_data_loader(data_folder: Optional[Path] = None) -> DataLoader:
    """
    Get global DataLoader instance (singleton pattern)
    
    Args:
        data_folder: Custom data folder path (optional)
        
    Returns:
        DataLoader instance
    """
    global _default_loader
    if _default_loader is None or data_folder is not None:
        _default_loader = DataLoader(data_folder or DATA_FOLDER)
    return _default_loader


# Convenience functions for backward compatibility
def load_spot_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load spot and spot_position data"""
    return get_data_loader().load_spot_data()


def load_spot_info() -> pd.DataFrame:
    """Load spot information"""
    return get_data_loader().load_spot_info()


def load_spot_position() -> pd.DataFrame:
    """Load spot position information"""
    return get_data_loader().load_spot_position()


def load_floor_info() -> pd.DataFrame:
    """Load floor metadata"""
    return get_data_loader().load_floor_info()


def load_gateway_info() -> pd.DataFrame:
    """Load gateway information"""
    return get_data_loader().load_gateway_info()


def load_outdoor_gateways() -> pd.DataFrame:
    """Load outdoor gateways"""
    return get_data_loader().load_outdoor_gateways()


def load_indoor_gateways(building_no: int, floor_no: int) -> pd.DataFrame:
    """Load indoor gateways for specific floor"""
    return get_data_loader().load_indoor_gateways(building_no, floor_no)


def load_flow_cache(cache_folder: str, date_str: str, resolution: str = '5min') -> pd.DataFrame:
    """Load flow cache"""
    return get_data_loader().load_flow_cache(cache_folder, date_str, resolution)


def load_location_cache(
    cache_path: str, 
    date_str: str, 
    building_no: int, 
    floor_no: Optional[int] = None
) -> Dict:
    """Load location cache"""
    return get_data_loader().load_location_cache(cache_path, date_str, building_no, floor_no)
