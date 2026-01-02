
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import sys
import numpy as np
import time

try:
    from src import config
    from src.cached_data_loader import CachedDataLoader
    from src.zone_manager import ZoneManager
    from src.tensor_builder import TensorBuilder
    from src.model.deepcon_stat import DeepConSTAT
except ImportError:
    import config
    from cached_data_loader import CachedDataLoader
    from zone_manager import ZoneManager
    from tensor_builder import TensorBuilder
    from model.deepcon_stat import DeepConSTAT

class ForecastEngine:
    def __init__(self, sort_by: str = 'spot_no'):
        # sort_by: 'spot_no' | 'name' | 'risk' passed to ZoneManager
        self.sort_by = sort_by
        self.zm = ZoneManager(sort_by=sort_by)
        self.tb = TensorBuilder(self.zm)
        
        # Initialize Model
        self.num_zones = self.zm.get_num_zones()
        self.model = DeepConSTAT(num_zones=self.num_zones)
        
        # Load pre-trained weights
        weights_path = Path("src/model/weights/best_model.pth")
        if weights_path.exists():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                print(f"✅ Loaded trained model weights from {weights_path}")
            except Exception as e:
                print(f"⚠️ Failed to load weights: {e}")
        else:
            print("⚠️ No trained weights found. Using random initialization.")
            
        self.model.eval() # Inference mode

    def predict_step(self, df: pd.DataFrame, t_point: int):
        """
        Runs a single-step inference for the given time point.
        Used by the Simulator for real-time-like playback.
        """
        # Context window: 60 minutes [t_point-60, t_point)
        start_idx = max(0, t_point - 60)
        df_chunk = df[(df['time_index'] >= start_idx) & (df['time_index'] < t_point)]

        if df_chunk.empty:
            return None, None, None

        # Build Tensor using absolute time indexing so features align to absolute bins
        tensor_chunk = self.tb.build_tensor(df_chunk, T=12, relative_time=True) # (12, Z, D) for 5-min x 1h (relative window)
        features = tensor_chunk.cpu().numpy()[-1, :, :] # (Z, D)
        
        with torch.no_grad():
            chunk_in = tensor_chunk.permute(1, 0, 2).unsqueeze(0) # (1, Z, 60, D)
            scores = self.model(chunk_in).squeeze() # (Z,)
            raw_scores = scores.cpu().numpy()
            
        # Apply Scaling (same as run_cycle for consistency)
        prior_multipliers = self.zm.get_prior_risk_vector()
        scaled_scores = np.zeros_like(raw_scores)
        
        for z in range(self.num_zones):
            p_weight = prior_multipliers[z]
            multiplier = 1.0 + (p_weight * 1.0) 
            raw_val = raw_scores[z]
            if raw_val > 0.001:
                scaled = (raw_val ** 0.5) * multiplier
            else:
                scaled = 0.0
            
            if scaled < 0.05:
                scaled = 0.0
            scaled_scores[z] = scaled
            
        # Global calibration isn't easily possible for a single step without daily context,
        # but we can apply a reasonable default scale or use the daily max if known.
        # For now, let's keep it normalized relative to a "Significant Activity" baseline of 0.7.
        # In a real simulator, we'd probably want to keep the same scale as the 24h forecast.
        
        return scaled_scores, features, df_chunk

    def run_cycle(self, target_date: str = None):
        # 요일 판별 (주중/주말)
        import datetime
        if target_date:
            try:
                dt = datetime.datetime.strptime(target_date, '%Y%m%d')
                weekday = dt.weekday() # 0=월, ..., 5=토, 6=일
                is_weekend = weekday >= 5
            except Exception:
                is_weekend = False
        else:
            is_weekend = False
        print(f"[Forecast] {target_date} is_weekend={is_weekend}")
        """
        Executes the full forecasting cycle:
        1. Load latest cache data
        2. Build Tensor
        3. Run Inference
        4. Save Forecast
        """
        import time
        start_time = time.time()
        print(f"🚀 Starting Forecast Cycle" + (f" for {target_date}" if target_date else ""))
        output_dir = Path("Cache")
        output_dir.mkdir(exist_ok=True)
        date_for_file = target_date if target_date else None

        # 1. Load Data
        t0 = time.time()
        loader = CachedDataLoader(config.CACHE_DIR, date_str=target_date)
        if not loader.is_valid():
            print("❌ No valid cache found.")
            return

        # Load 5-min flow data - SAME AS T31/T41 TABS (Optimized with column pruning)
        try:
            # We only need these columns for TensorBuilder
            # CRITICAL: mac_address is required for unique worker/equipment counting
            # USE 5-MIN RESOLUTION to match T31/T41 tabs (already aggregated, no duplicates)
            needed_cols = ['time_index', 'type', 'spot_nos', 'mac_address', 'position_confidence', 'status']
            df = loader.load_flow_cache(resolution='5min', columns=needed_cols)
        except Exception as e:
            print(f"⚠️ Failed to load 5-min cache: {e}")
            return
            
        if df.empty:
            print("⚠️ Data is empty.")
            return
        t1 = time.time()
        print(f"⏱️ Data Load: {t1-t0:.2f}s")
        # data loaded

        # Load dwell time (체류시간) per zone (T41 작업자) & anomaly 기준 계산
        try:
            dwell_df = loader.load_t41_worker_dwell()
            # Normalize dwell_df to a DataFrame when loader returns dict/list
            if isinstance(dwell_df, dict) or isinstance(dwell_df, list):
                try:
                    dwell_df = pd.DataFrame(dwell_df)
                except Exception:
                    dwell_df = pd.DataFrame()
            if not isinstance(dwell_df, pd.DataFrame):
                dwell_df = pd.DataFrame()
            zone_dwell = np.zeros(self.num_zones)
            dwell_anomaly = np.zeros(self.num_zones)
            # 기준값: 주중/주말별 zone 평균 dwell_minutes (캐시에서 과거 데이터 활용)
            # 여기서는 예시로 Cache/zone_dwell_stats_{weekday|weekend}.json에서 불러온다고 가정
            import json
            stats_path = Path("Cache") / f"zone_dwell_stats_{'weekend' if is_weekend else 'weekday'}.json"
            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as f:
                    dwell_stats = json.load(f) # {zone_name: avg_dwell_minutes}
            else:
                dwell_stats = {}
            if not dwell_df.empty:
                # mac별로 zone 정보가 있다면 groupby('zone') 사용
                if 'zone' in dwell_df.columns:
                    zone_group = dwell_df.groupby('zone')['dwell_minutes'].mean()
                    for idx, zname in enumerate(self.zm.get_zone_names()):
                        cur_dwell = zone_group.get(zname, 0)
                        base_dwell = dwell_stats.get(zname, 0)
                        zone_dwell[idx] = cur_dwell
                        # anomaly: 기준값 대비 1.3배 이상이면 비정상
                        dwell_anomaly[idx] = 1 if base_dwell > 0 and cur_dwell > base_dwell * 1.3 else 0
                elif 'spot_no' in dwell_df.columns:
                    spot_group = dwell_df.groupby('spot_no')['dwell_minutes'].mean()
                    for idx in range(self.num_zones):
                        spot_no = self.zm.get_spot_no(idx)
                        cur_dwell = spot_group.get(spot_no, 0)
                        zname = self.zm.get_zone_names()[idx]
                        base_dwell = dwell_stats.get(zname, 0)
                        zone_dwell[idx] = cur_dwell
                        dwell_anomaly[idx] = 1 if base_dwell > 0 and cur_dwell > base_dwell * 1.3 else 0
                else:
                    avg_dwell = dwell_df['dwell_minutes'].mean()
                    zone_dwell[:] = avg_dwell
                    dwell_anomaly[:] = 0
            else:
                zone_dwell[:] = 0
                dwell_anomaly[:] = 0
        except Exception as e:
            import traceback
            print(f"⚠️ Failed to load dwell time: {e}")
            # Diagnostic: inspect dwell loader return and column value types
            try:
                import types
                print("[DWELL DIAG] Inspecting dwell_df variable...")
                # If dwell_df exists in locals, report its type and sample
                if 'dwell_df' in locals():
                    dd = dwell_df
                    try:
                        print("[DWELL DIAG] dwell_df type:", type(dd))
                        if hasattr(dd, 'shape'):
                            print(f"[DWELL DIAG] shape: {getattr(dd, 'shape', None)}")
                        print("[DWELL DIAG] dtypes:")
                        try:
                            print(dd.dtypes)
                        except Exception:
                            print("[DWELL DIAG] cannot print dtypes")
                        # For each column, show whether any cell is dict or list and sample
                        for col in getattr(dd, 'columns', [])[:10]:
                            try:
                                col_vals = dd[col].head(20).tolist()
                                types_present = {type(x) for x in col_vals}
                                print(f"[DWELL DIAG] col={col}, sample_types={types_present}")
                                # show first offending cell if dict found
                                for x in col_vals:
                                    if isinstance(x, dict):
                                        print(f"[DWELL DIAG] first dict in col={col}: {x}")
                                        break
                            except Exception:
                                print(f"[DWELL DIAG] could not inspect col={col}")
                    except Exception:
                        print("[DWELL DIAG] failed to introspect dwell_df")
                else:
                    print("[DWELL DIAG] dwell_df not defined in locals")
            except Exception:
                pass
            traceback.print_exc()
            zone_dwell = np.zeros(self.num_zones)
            dwell_anomaly = np.zeros(self.num_zones)
        
        # 2. Build Tensors & Run Sliding Inference (5-min intervals for Simulator fidelity)
        # 1440 mins / 5 mins = 288 intervals
        interval = 5
        time_points = list(range(interval, 1441, interval))  # 5, 10, 15, ..., 1440 (minutes)
        
        num_zones = len(self.zm.get_zone_names())
        num_steps = len(time_points)
        
        # Matrix to store ALL risk scores (Zone x TimeSteps)
        risk_matrix = np.zeros((num_zones, num_steps))
        # Store features for EVERY step for Simulator reasoning (T, Z, D)
        all_step_features = np.zeros((num_steps, num_zones, 4)) 
        
        print(f"⚡ Processing 24h Risk Evolution ({num_steps} intervals, 5-min res)...")
        
        t2 = time.time()
        for i, t_point in enumerate(time_points):
            # Context window: 12 intervals (60 minutes) ending at current time_index
            # Convert minute-based t_point to 5-min time_index (1~288)
            # t_point is in minutes (5, 10, 15...), time_index is 1-based 5-min intervals
            current_idx = t_point // 5  # e.g., t_point=60 -> idx=12, t_point=1440 -> idx=288
            start_idx = max(1, current_idx - 12 + 1)  # 12 intervals = 1 hour window
            df_chunk = df[(df['time_index'] >= start_idx) & (df['time_index'] <= current_idx)]
            
            # Build Tensor using RELATIVE time indexing (maps chunk to 0..11)
            # relative_time=True ensures sparse data maps correctly to tensor positions
            tensor_chunk = self.tb.build_tensor(df_chunk, T=12, relative_time=True) # (12, Z, D) for 12 x 5-min intervals
            
            # Capture features for this step (last 5-min interval of the window)
            features = tensor_chunk.cpu().numpy()[-1, :, :]
            all_step_features[i] = features
            
            with torch.no_grad():
                chunk_in = tensor_chunk.permute(1, 0, 2).unsqueeze(0) # (1, Z, 12, D) - batch_size=1, zones, time_steps=12, channels=4
                scores = self.model(chunk_in).squeeze() # (Z,)
                risk_matrix[:, i] = scores.cpu().numpy()
            # periodic progress update (no-op in sync mode)

        t3 = time.time()
        print(f"⏱️ Total 24h Inference: {t3-t2:.2f}s")
        
        # 3. Integrated Risk Scoring: combine model output with interpretable signals
        # Components:
        #  - S_density: normalized worker density (0..1)
        #  - S_mix: equipment-to-worker mix proxy (0..1)
        #  - S_dwell: normalized dwell time (0..1)
        #  - M_zone: multiplicative zone prior multiplier
        #  - B_anomaly: binary anomaly booster (from dwell anomaly)
        norm_matrix = np.zeros_like(risk_matrix)
        prior_multipliers = self.zm.get_prior_risk_vector() # (Z,)
        zone_areas = self.zm.get_zone_areas()
        # weights for interpretable components (tuned to reduce over-sensitivity)
        w_density, w_mix, w_dwell = 0.4, 0.2, 0.4
        print("[DIAG] zone, time, raw_score, density, scaled_score (최대 10개)")
        diag_count = 0
        # baseline dwell lookup (may be empty)
        # dwell_stats loaded earlier may be {} if missing
        for z in range(num_zones):
            p_weight = prior_multipliers[z]
            M_zone = 1.0 + (p_weight * 0.5)
            B_anomaly = 1.0 + (0.5 if dwell_anomaly[z] else 0.0)
            for t in range(num_steps):
                raw_score = float(risk_matrix[z, t])
                p_feat = all_step_features[t, z, :] if all_step_features is not None else np.zeros(4)
                # p_feat channels store log1p((count/area)*100)
                feat_worker = float(np.expm1(p_feat[0]))  # equals (count/area)*100
                feat_equip = float(np.expm1(p_feat[1]))   # equals (equip_count/area)*100

                area = float(zone_areas[z]) if z < len(zone_areas) else 100.0
                # Recover density in persons/m2: (count/area) = feat_worker / 100
                density_ppm2 = feat_worker / 100.0
                # Smooth saturation: use a bounded non-linear mapping to avoid sharp jumps
                # S_density = density / (density + k) with k ~ 1.0 person/m2 (comfortable)
                S_density = float(min(1.0, density_ppm2 / (density_ppm2 + 1.0)))

                # S_mix: equipment per worker proxy. If workers=0, use equip/10 as proxy
                # Compute approximate counts to form equipment-to-worker mix
                worker_count = density_ppm2 * area
                equip_count = (feat_equip / 100.0) * area
                if worker_count > 0:
                    mix = equip_count / (worker_count + 1e-6)
                    S_mix = float(min(1.0, mix / 0.5))
                else:
                    S_mix = float(min(1.0, equip_count / 10.0))

                # S_dwell: zone-level dwell normalized by baseline from stats (if available)
                zname = self.zm.get_zone_names()[z]
                baseline = dwell_stats.get(zname, None) if isinstance(dwell_stats, dict) else None
                if baseline and baseline > 0:
                    S_dwell = float(min(1.0, zone_dwell[z] / (baseline + 1e-6)))
                else:
                    S_dwell = float(min(1.0, zone_dwell[z] / 10.0))

                # Feature-based combined score
                feature_score = w_density * S_density + w_mix * S_mix + w_dwell * S_dwell
                feature_score = float(min(1.0, feature_score))

                # Model component (use as-is when confident)
                if raw_score > 0.02:
                    model_comp = raw_score ** (0.75 if is_weekend else 0.8)
                    final_score = 0.6 * model_comp + 0.4 * feature_score
                else:
                    final_score = 0.95 * feature_score

                # Apply zone multiplier and anomaly booster
                final_score = final_score * M_zone * B_anomaly
                # small threshold to remove numerical noise
                if final_score < 0.01:
                    final_score = 0.0
                norm_matrix[z, t] = float(final_score)
                if diag_count < 10:
                    print(f"zone={z}, t={t}, raw={raw_score:.4f}, density={S_density:.4f}, scaled={final_score:.4f}")
                    diag_count += 1

        # 4. Global Peak Calibration (Target 0.7)
        # Cap amplification to avoid blowing up tiny model outputs into identical
        # mid-range scores. Allow modest downscaling if necessary.
        global_max = norm_matrix.max()
        if global_max > 0:
            desired = 0.7
            scale_factor = desired / global_max
            # prevent extreme amplification or complete collapse
            scale_factor = max(0.5, min(scale_factor, 2.0))
        else:
            scale_factor = 1.0

        norm_matrix = norm_matrix * scale_factor
        norm_matrix = np.clip(norm_matrix, 0, 1.0)
        
        # Calculate global statistics for anomaly detection
        global_mean = norm_matrix.mean()
        global_std = norm_matrix.std()
        
        # 5. Advanced Global AI Analysis
        peak_idx = np.unravel_index(np.argmax(norm_matrix, axis=None), norm_matrix.shape)
        peak_zone_idx, peak_time_idx = peak_idx
        peak_score = norm_matrix[peak_zone_idx, peak_time_idx]
        peak_zone_name = self.zm.get_zone_names()[peak_zone_idx]
        
        peak_hh = time_points[peak_time_idx] // 60
        peak_mm = time_points[peak_time_idx] % 60
        peak_time_str = f"{peak_hh:02d}:{peak_mm:02d}"
        
        # Global Insight generation (Peak reasoning)
        reason_msg = ""
        p_feat = all_step_features[peak_time_idx, peak_zone_idx]
        worker_den = np.expm1(p_feat[0])
        equip_den = np.expm1(p_feat[1])
        static_risk = p_feat[2]
        
        if worker_den > 30:
            reason_msg = f"해당 구역의 **인원 밀집도(Worker Density: {worker_den:.1f})가 매우 높게** 감지되었습니다. "
        elif equip_den > 5:
            reason_msg = f"구역 내 **중장비 가동(Equipment Activity: {equip_den:.1f})**으로 인한 충돌 위험이 감지되었습니다. "
        elif static_risk > 0.5:
            reason_msg = f"해당 구역의 **고유 위험성(Static Risk: {static_risk:.1f}, 밀폐/고소)**이 주원인입니다. "
        else:
            reason_msg = "복합적인 현장 활동 패턴 분석 결과 주의가 필요합니다. "

        # 전체 평균 대비 피크 위험도 평가
        peak_z_score = (peak_score - global_mean) / global_std if global_std > 0 else 0
        
        if peak_z_score > 2.5 and peak_score > 0.6:
            global_insight = f"🚨 **DeepCon AI 분석 - CRITICAL**: 금일 **{peak_time_str}**에 **'{peak_zone_name}'** 구역에서 **평소와 다른 비정상적인 고위험 상황**이 예측됩니다. {reason_msg}긴급 안전 점검 및 작업 조정이 필요합니다."
        elif peak_score > 0.4:
            global_insight = f"⚠️ **DeepCon AI 분석 - CAUTION**: 금일 가장 주의가 필요한 시점은 **{peak_time_str}**이며, **'{peak_zone_name}'** 구역에서 피크 활동이 예상됩니다. {reason_msg}현장 안전 관리에 유의해 주시기 바랍니다."
        else:
            global_insight = f"✅ **DeepCon AI 분석 - SAFE**: 금일 현장의 전반적인 리스크는 **안전한(Safe)** 수준으로 예측됩니다. 일상적인 안전 수칙을 준수해 주세요."
 
        # 6. Process Results (Daily Max Snapshot)
        daily_max_scores = np.max(norm_matrix, axis=1) # (Z,)
        
        # 평소 대비 비정상 감지를 위한 통계 계산
        # 하루 전체 평균과 표준편차를 사용하여 이상치 탐지
        daily_mean_scores = np.mean(norm_matrix, axis=1)  # 각 구역의 하루 평균
        daily_std_scores = np.std(norm_matrix, axis=1)    # 각 구역의 표준편차
        global_mean = np.mean(daily_max_scores)            # 전체 평균
        global_std = np.std(daily_max_scores)              # 전체 표준편차
        
        results = []
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for idx, score in enumerate(daily_max_scores):
            spot_no = self.zm.get_spot_no(idx)
            peak_t_idx = np.argmax(norm_matrix[idx, :])
            p_hh = time_points[peak_t_idx] // 60
            p_mm = time_points[peak_t_idx] % 60
            peak_time_local = f"{p_hh:02d}:{p_mm:02d}"

            p_val = prior_multipliers[idx]
            dwell_val = zone_dwell[idx]
            anomaly_flag = dwell_anomaly[idx]
            
            # 해당 구역의 평균 및 표준편차
            zone_mean = daily_mean_scores[idx]
            zone_std = daily_std_scores[idx] if daily_std_scores[idx] > 0 else 0.1
            
            # Z-score 계산: 평소 대비 얼마나 벗어났는지
            z_score = (score - zone_mean) / zone_std if zone_std > 0 else 0
            
            # 전체 평균 대비 편차
            global_z = (score - global_mean) / global_std if global_std > 0 else 0
            
            # 위험도 레벨 판단 (개선된 로직)
            # 1. Critical: 평소 대비 2 표준편차 이상 벗어나거나, 절대값이 매우 높고 이상치일 때
            # 2. Caution: 하루 중 최고점이면서 일정 수준 이상일 때
            # 3. Safe: 나머지 대부분의 경우
            
            if (z_score > 2.0 or global_z > 2.5) and score > 0.6:
                # 평소와 다른 비정상적인 상황
                severity = "CRITICAL"
            elif score >= daily_max_scores.max() * 0.85 and score > 0.35:
                # 하루 중 최고 수준의 위험도 (상위 15% 이내)
                severity = "CAUTION"
            elif anomaly_flag and z_score > 1.5:
                # 체류시간 이상 + 위험도 증가
                severity = "CAUTION"
            else:
                # 일반적인 안전한 상황
                severity = "SAFE"

            # 상세한 원인 분석
            reasoning = f"[{severity}] 금일 최대 위험 도달 시간: {peak_time_local}. "
            
            # 피크 시점의 상세 데이터 추출
            p_feat = all_step_features[peak_t_idx, idx]
            worker_density = float(np.expm1(p_feat[0]))  # 인원 밀집도
            equip_density = float(np.expm1(p_feat[1]))   # 장비 밀도
            static_risk = float(p_feat[2])                # 고유 위험도
            
            area = float(zone_areas[idx]) if idx < len(zone_areas) else 100.0
            worker_count = (worker_density / 100.0) * area
            density_per_m2 = worker_density / 100.0
            
            # 원인 상세 분석
            reasons = []
            if severity == "CRITICAL":
                reasons.append(f"⚠️ **평소 대비 {z_score:.1f}배 높은 위험도 감지** (이상치)")
            
            if worker_density > 50:
                reasons.append(f"**과밀 상태**: 면적 대비 인원 밀집도 {density_per_m2:.2f}명/m² (약 {worker_count:.0f}명, 권장 기준 초과)")
            elif worker_density > 30:
                reasons.append(f"**인원 집중**: 면적 대비 {density_per_m2:.2f}명/m² (약 {worker_count:.0f}명)")
            elif worker_density > 10:
                reasons.append(f"**일반 작업**: 면적 대비 {density_per_m2:.2f}명/m² (약 {worker_count:.0f}명)")
            
            if equip_density > 8:
                reasons.append(f"**중장비 밀집**: 장비 가동률 높음 (충돌/협착 위험 증가)")
            elif equip_density > 3:
                reasons.append(f"**장비 운용 중**: 작업자-장비 간 안전거리 유지 필요")
            
            if static_risk > 0.7:
                reasons.append(f"**고위험 구역**: 밀폐공간/고소작업 등 상시 위험요소 존재")
            elif static_risk > 0.4:
                reasons.append(f"**주의 구역**: 구조적 위험요소 있음")
            
            if anomaly_flag:
                reasons.append(f"**비정상 체류**: 평소 대비 체류시간 길어짐 (평균 {dwell_val:.0f}분, 작업 지연 또는 문제 발생 가능성)")
            
            if reasons:
                reasoning += " ".join(reasons)
            else:
                reasoning += "정상적인 작업 패턴 유지 중. 일상적인 안전 수칙을 준수하세요."

            results.append({
                "spot_id": int(spot_no),
                "zone_name": self.zm.get_zone_names()[idx],
                "risk_score": float(score),
                "severity": severity,
                "reasoning": reasoning,
                "timestamp": now_str
            })
            
        time_labels = [f"{tp//60:02d}:{tp%60:02d}" for tp in time_points]
            
        output_data = {
            "forecasts": results,
            "heatmap": {
                "z": norm_matrix.tolist(),
                "y": self.zm.get_zone_names(),
                "x": time_labels
            },
            "step_features": all_step_features.tolist(), # (T, Z, D)
            "global_analysis": {
                "peak_time": peak_time_str,
                "peak_zone": peak_zone_name,
                "peak_score": float(peak_score),
                "insight": global_insight
            }
        }
        # Add explicit zone_name -> spot_id mapping for downstream consumers
        zone_names = self.zm.get_zone_names()
        name_to_spot = {}
        for idx, zname in enumerate(zone_names):
            try:
                sid = int(self.zm.get_spot_no(idx))
            except Exception:
                sid = None
            if sid is not None:
                name_to_spot[zname] = sid
        output_data['zone_mapping'] = name_to_spot
            
        # Save to JSON (use date tag)
        date_for_file = target_date if target_date else loader.date_str
        output_file = output_dir / f"forecast_{date_for_file}.json"
        
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Also keep 'latest_forecast.json' as a pointer to the newest one
        latest_file = output_dir / "latest_forecast.json"
        with open(latest_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Also save a dedicated zone mapping file for quick lookup
        try:
            mapping_file = output_dir / f"zone_mapping_{date_for_file}.json"
            with open(mapping_file, 'w', encoding='utf-8') as mf:
                json.dump(output_data.get('zone_mapping', {}), mf, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # completed

        total_time = time.time() - start_time
        print(f"✅ Refined 24h Forecast saved to {output_file} (Total: {total_time:.2f}s)")

        # 개선된 이상 감지 메시지 (평소 대비 비정상만 보고)
        anomalies = []
        
        # 전역 피크가 평소 대비 이상치인 경우만
        peak_z_score = (peak_score - global_mean) / global_std if global_std > 0 else 0
        if peak_z_score > 2.5 and peak_score > 0.6:
            anomalies.append({
                'level': 'CRITICAL',
                'message': f'[비정상 감지] {peak_zone_name} 구역 {peak_time_str} 시점에서 평소 대비 {peak_z_score:.1f}σ 높은 위험도({peak_score:.2f}) 감지. 긴급 점검 필요.'
            })
        elif peak_score > 0.5:
            anomalies.append({
                'level': 'WARNING',
                'message': f'[주의] {peak_zone_name} 구역 {peak_time_str} 시점 피크 활동({peak_score:.2f}). 안전 관리 강화 권장.'
            })

        # 구역별 CRITICAL만 보고 (평소 대비 이상치)
        for r in results:
            if r.get('severity') == 'CRITICAL':
                anomalies.append({
                    'level': 'CRITICAL',
                    'message': f"[이상치] {r['zone_name']} (spot {r['spot_id']}) 평소 대비 높은 위험도({r['risk_score']:.2f}) - 즉시 확인 필요"
                })

        output_data['anomalies'] = anomalies

        return output_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date (YYYYMMDD)")
    args = parser.parse_args()
    
    engine = ForecastEngine()
    engine.run_cycle(target_date=args.date)
