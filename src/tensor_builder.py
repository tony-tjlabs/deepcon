
import pandas as pd
import numpy as np
import torch
from typing import Tuple

try:
    from src import config
    from src.zone_manager import ZoneManager
except ImportError:
    import config
    from zone_manager import ZoneManager

class TensorBuilder:
    """
    Constructs (Time, Zone, Channel) tensors from 1-minute flow cache data.
    """
    def __init__(self, zone_manager: ZoneManager):
        self.zm = zone_manager
        self.num_zones = self.zm.get_num_zones()
        self.prior_risk = self.zm.get_prior_risk_vector() # (Z,)

    def build_tensor(self, df: pd.DataFrame, T: int = 12, relative_time: bool = False) -> torch.Tensor:
        """
        Builds the input tensor for DeepCon-STAT from 5-minute aggregated flow data.
        
        Args:
            df: DataFrame containing 5-minute flow cache data.
                - time_index: 1-based 5-minute interval index (1~288 for 24h)
                - Each interval represents 5 minutes
            T: Total time steps in the output tensor (default 12 = 1 hour window)
            relative_time: If True, t_idx is relative to the minimum time_index in df.
                          If False, t_idx uses absolute time_index modulo arithmetic.
        
        Returns:
            Tensor of shape (T, Z, D) where:
            - T: time steps (e.g., 12 for 60-min window)
            - Z: number of zones
            - D: 4 channels (workers, equipment, prior_risk, inactive_alert)
        """
        Z = self.num_zones
        D = 4
        
        # Initialize tensor
        tensor_np = np.zeros((T, Z, D), dtype=np.float32)
        
        # Fill Prior Risk (Static) - Broadcasted
        for t in range(T):
            tensor_np[t, :, 2] = self.prior_risk
            
        if df is None or df.empty:
            return torch.from_numpy(tensor_np)

        needed_cols = ['time_index', 'type', 'spot_nos', 'spot_no', 'mac_address', 'position_confidence', 'status']
        available_cols = [c for c in needed_cols if c in df.columns]
        sub_df = df[available_cols].copy()
        
        # Ensure status exists for filtering
        if 'status' not in sub_df.columns:
            sub_df['status'] = 1 
        
        # Drop rows with no spot
        sub_df = sub_df.dropna(subset=['spot_nos'])
        sub_df = sub_df[sub_df['spot_nos'] != '']
        
        # Explode logic: support both 'spot_nos' (comma list) and single 'spot_no'
        if 'spot_nos' in sub_df.columns:
            df_expl = sub_df.assign(spot_no=sub_df['spot_nos'].str.split(',')).explode('spot_no')
            df_expl['spot_no'] = pd.to_numeric(df_expl['spot_no'], errors='coerce')
            df_expl = df_expl.dropna(subset=['spot_no'])
            df_expl['spot_no'] = df_expl['spot_no'].astype(int)
        elif 'spot_no' in sub_df.columns:
            df_expl = sub_df.copy()
            df_expl['spot_no'] = pd.to_numeric(df_expl['spot_no'], errors='coerce')
            df_expl = df_expl.dropna(subset=['spot_no'])
            df_expl['spot_no'] = df_expl['spot_no'].astype(int)
        else:
            return torch.from_numpy(tensor_np)
        
        # Map to z_idx
        df_expl['z_idx'] = df_expl['spot_no'].map(self.zm.zone_to_idx).fillna(-1).astype(int)
        df_expl = df_expl[df_expl['z_idx'] != -1]
        
        # Time index logic for 5-minute intervals
        if relative_time:
            # Relative indexing: map time_index to sequential 0, 1, 2, ...
            # Since time_index are 5-min intervals, no conversion needed
            min_t = df_expl['time_index'].min()
            df_expl['t_idx'] = (df_expl['time_index'] - min_t).astype(int)
        else:
            # Absolute indexing: for rolling windows
            # time_index is 1~288 (5-min intervals), map to window positions
            # Example: for 1-hour window (T=12), use modulo 12
            df_expl['t_idx'] = ((df_expl['time_index'] - 1) % T).astype(int)
            
        df_expl = df_expl[(df_expl['t_idx'] >= 0) & (df_expl['t_idx'] < T)]
        
        if df_expl.empty:
            return torch.from_numpy(tensor_np)

        # Aggregation with Status Filtering
        # 1. Active Worker counts (status == 1)
        worker_active = df_expl[(df_expl['type'] == config.TYPE_41_WORKER) & (df_expl['status'] == 1)]
        if not worker_active.empty and 'mac_address' in worker_active.columns:
            worker_counts = worker_active.groupby(['t_idx', 'z_idx'])['mac_address'].nunique()
        else:
            worker_counts = worker_active.groupby(['t_idx', 'z_idx']).size()
        
        # 2. Inactive Worker Alert (status == 0) in Confined Spaces
        # Channel 3 (Index 3) was taking average confidence. 
        # We'll repurpose it or add logic to Prior Risk if strictly needed, 
        # but for now, let's put "Inactive Tag Presence" in Channel 3 (replaces confidence).
        worker_inactive = df_expl[(df_expl['type'] == config.TYPE_41_WORKER) & (df_expl['status'] == 0)]
        if not worker_inactive.empty:
            # Mask for confined spaces
            confined_mask = self.prior_risk > 0.5 # ConfinedSpace threshold
            inact_counts = worker_inactive.groupby(['t_idx', 'z_idx']).size()
            
            t_in, z_in = zip(*inact_counts.index)
            # Only alert if the zone is confined space
            for ti, zi in zip(t_in, z_in):
                if confined_mask[zi]:
                    tensor_np[ti, zi, 3] = 1.0 # High risk alert in Channel 3 (Confidence -> Alert)
        
        # 3. Active Equipment counts (status == 1) - Only count active T31
        equip_active = df_expl[(df_expl['type'] == config.TYPE_31_EQUIPMENT) & (df_expl['status'] == 1)]
        if not equip_active.empty and 'mac_address' in equip_active.columns:
            equip_counts = equip_active.groupby(['t_idx', 'z_idx'])['mac_address'].nunique()
        else:
            equip_counts = equip_active.groupby(['t_idx', 'z_idx']).size()
        
        # Fill tensor
        if not worker_counts.empty:
            t_w, z_w = zip(*worker_counts.index)
            tensor_np[list(t_w), list(z_w), 0] = worker_counts.values
        
        if not equip_counts.empty:
            t_e, z_e = zip(*equip_counts.index)
            tensor_np[list(t_e), list(z_e), 1] = equip_counts.values
            
        # Normalize counts by Area
        areas = self.zm.get_zone_areas()
        tensor_np[:, :, 0] = (tensor_np[:, :, 0] / areas) * 100.0
        tensor_np[:, :, 1] = (tensor_np[:, :, 1] / areas) * 100.0
        
        # Log-normalization
        tensor_np[:, :, 0] = np.log1p(tensor_np[:, :, 0])
        tensor_np[:, :, 1] = np.log1p(tensor_np[:, :, 1])
        
        return torch.from_numpy(tensor_np)

if __name__ == "__main__":
    # Test
    zm = ZoneManager()
    tb = TensorBuilder(zm)
    print(f"TensorBuilder initialized. Zones: {tb.num_zones}")
