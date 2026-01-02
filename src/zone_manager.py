
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Try to import config
try:
    from src import config
except ImportError:
    import config

class ZoneManager:
    """
    Manages mapping between physical spots (Zone) and Tensor indices ($Z$).
    Also handles static risk factors (Prior) based on spot attributes.
    """
    def __init__(self, data_context_dir=None, sort_by: str = 'spot_no'):
        """
        Args:
            data_context_dir: Directory containing spot.csv. 
                              If None, uses config.SPOT_CSV.parent.
        """
        self.spot_df = None
        self.zone_to_idx = {}
        self.idx_to_zone = {}
        self.prior_risk = None
        # sort_by: 'spot_no' | 'name' | 'risk'
        self.sort_by = sort_by if sort_by in ('spot_no', 'name', 'risk') else 'spot_no'
        
        # Determine data directory
        if data_context_dir:
            self.data_dir = Path(data_context_dir)
        elif hasattr(config, 'SPOT_CSV'):
            self.data_dir = config.SPOT_CSV.parent
        else:
            raise ValueError("No data_context_dir provided and config.SPOT_CSV not found.")
            
        self._load_data()

    def _load_data(self):
        spot_path = self.data_dir / "spot.csv"
        if not spot_path.exists():
            raise FileNotFoundError(f"spot.csv not found in {self.data_dir}")
            
        self.spot_df = pd.read_csv(spot_path)
        # Validate required column
        if 'spot_no' not in self.spot_df.columns:
            raise ValueError("spot.csv must contain 'spot_no' column")

        # Compute static risk per row first (keep in a column for sorting)
        def _compute_risk_score(row):
            risk_score = 0.0
            div = str(row.get('div', '')).lower()
            draw_div = str(row.get('draw_div', '')).lower()
            name = str(row.get('name', '')).lower()

            if 'confinedspace' in div or 'confinedspace' in draw_div or '밀폐' in name:
                return 1.0
            if 'hoist' in div or 'hoist' in draw_div or '호이스트' in name:
                return 0.8
            if 'workfloor' in div:
                return 0.2
            if 'restspace' in div or '휴게' in name:
                return 0.0
            return 0.0

        self.spot_df['prior_risk'] = self.spot_df.apply(_compute_risk_score, axis=1)

        # Sorting according to requested key
        if self.sort_by == 'spot_no':
            # numeric sort if possible
            try:
                self.spot_df['spot_no_sort'] = pd.to_numeric(self.spot_df['spot_no'], errors='coerce')
                self.spot_df = self.spot_df.sort_values(['spot_no_sort']).drop(columns=['spot_no_sort']).reset_index(drop=True)
            except Exception:
                self.spot_df = self.spot_df.sort_values('spot_no').reset_index(drop=True)
        elif self.sort_by == 'name':
            self.spot_df['name_sort'] = self.spot_df['name'].astype(str).str.lower()
            self.spot_df = self.spot_df.sort_values(['name_sort']).drop(columns=['name_sort']).reset_index(drop=True)
        elif self.sort_by == 'risk':
            # sort by prior_risk descending, tie-breaker by name
            self.spot_df['name_sort'] = self.spot_df['name'].astype(str).str.lower()
            self.spot_df = self.spot_df.sort_values(['prior_risk', 'name_sort'], ascending=[False, True]).drop(columns=['name_sort']).reset_index(drop=True)

        # Create bidirectional mappings in the chosen order
        self.zone_to_idx = {row['spot_no']: idx for idx, row in self.spot_df.iterrows()}
        self.idx_to_zone = {idx: row['spot_no'] for idx, row in self.spot_df.iterrows()}

        # Load Area Data (from spot_position.csv) - respects current spot_df order
        self.zone_areas = self._calculate_areas()

        # Initialize prior risk vector (static risk) in index order
        num_zones = len(self.spot_df)
        self.prior_risk = np.zeros(num_zones, dtype=np.float32)
        for idx, row in self.spot_df.iterrows():
            try:
                self.prior_risk[idx] = float(row.get('prior_risk', 0.0))
            except Exception:
                self.prior_risk[idx] = 0.0
            
    def _calculate_areas(self):
        """Calculate area (m2) for each spot using Shoelace formula."""
        pos_path = self.data_dir / "spot_position.csv"
        num_zones = len(self.spot_df)
        areas = np.full(num_zones, 100.0, dtype=np.float32) # Default 100m2
        
        if not pos_path.exists():
            return areas
            
        pos_df = pd.read_csv(pos_path)
        
        for idx, row in self.spot_df.iterrows():
            spot_no = row['spot_no']
            # Get points for this spot
            pts = pos_df[pos_df['spot_no'] == spot_no].sort_values("point_no")
            if len(pts) < 3:
                continue
                
            x = pts['x'].values
            y = pts['y'].values
            
            # Shoelace Formula
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            # Area in Construction Map units (assuming 1 unit = 0.1m or similar)
            # For now, let's assume raw units are roughly proportional. 
            # If map is 2500x1200, and real world is 250m x 120m, then 1 unit = 0.1m -> 1 unit^2 = 0.01 m2
            actual_area_m2 = area * 0.01 # Adjust scale factor as needed
            areas[idx] = max(10.0, actual_area_m2) # Min 10m2
            
        return areas

    def get_num_zones(self):
        """Return total number of zones (N)"""
        return len(self.zone_to_idx)
        
    def get_idx(self, spot_no):
        """Map spot_no to tensor index (0~N-1)"""
        return self.zone_to_idx.get(spot_no, -1)
        
    def get_spot_no(self, idx):
        """Map tensor index to spot_no"""
        return self.idx_to_zone.get(idx, None)

    def get_prior_risk_vector(self):
        """Return (N,) vector of static risk scores"""
        return self.prior_risk
    
    def get_zone_areas(self):
        """Return (N,) vector of zone areas in m2"""
        return self.zone_areas
    
    def get_zone_names(self):
        """Return list of zone names in index order"""
        return self.spot_df['name'].tolist()

if __name__ == "__main__":
    # Simple test
    try:
        zm = ZoneManager()
        print(f"Loaded {zm.get_num_zones()} zones.")
        print(f"Sample mapping: Spot {zm.get_spot_no(0)} -> Index 0")
        print(f"Prior risk mean: {zm.get_prior_risk_vector().mean():.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
