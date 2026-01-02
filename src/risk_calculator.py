"""
DeepCon 위험도 계산 모듈
========================

5가지 요소 기반 위험도 계산:
1. 면적 대비 인원수 (밀집도)
2. 구역 고유 위험도
3. 작업자+장비 혼재
4. 과거 패턴 대비 편차
5. 밀폐공간 예외 처리

최종 위험도는 0~1 범위로 정규화
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class RiskCalculator:
    """위험도 계산기"""
    
    def __init__(self):
        """초기화"""
        # 가중치 설정 (합이 1.0)
        self.weight_density = 0.25        # 면적 대비 인원수
        self.weight_base_risk = 0.20      # 구역 고유 위험도
        self.weight_混재 = 0.15            # 작업자+장비 혼재
        self.weight_pattern_dev = 0.35    # 과거 패턴 대비 편차 (가장 중요!)
        self.weight_confined_adj = 0.05   # 밀폐공간 조정
        
        # 밀집도 기준 (명/m²)
        self.density_low = 0.05      # 낮음: 0.05명/m² 미만
        self.density_medium = 0.10   # 중간: 0.10명/m² 미만
        self.density_high = 0.15     # 높음: 0.15명/m² 이상
        
        # 패턴 편차 기준 (표준편차 배수)
        self.pattern_dev_threshold_low = 1.0    # 1σ 미만: 정상
        self.pattern_dev_threshold_med = 2.0    # 2σ 미만: 주의
        self.pattern_dev_threshold_high = 3.0   # 3σ 이상: 위험
        
        # 스케일링 팩터 설정
        # 체류시간 기준 (분)
        self.dwell_short = 10      # 10분 미만: 정상
        self.dwell_medium = 30     # 30분 미만: 약간 증가
        self.dwell_long = 60       # 60분 이상: 위험 증가
        
        # 시간대 위험 구간 (시)
        self.night_start = 22      # 야간 시작 (22시)
        self.night_end = 6         # 야간 종료 (6시)
        self.dawn_start = 0        # 새벽 시작 (0시)
        self.dawn_end = 5          # 새벽 종료 (5시)
    
    def calculate_dwell_time_factor(self, avg_dwell_minutes: float) -> float:
        """
        체류시간 기반 스케일링 팩터 계산
        
        Args:
            avg_dwell_minutes: 평균 체류시간 (분)
            
        Returns:
            스케일링 팩터 (1.0~1.3)
        """
        if avg_dwell_minutes < self.dwell_short:
            return 1.0  # 정상
        elif avg_dwell_minutes < self.dwell_medium:
            # 10~30분: 선형 증가 (1.0 → 1.1)
            ratio = (avg_dwell_minutes - self.dwell_short) / (self.dwell_medium - self.dwell_short)
            return 1.0 + ratio * 0.1
        elif avg_dwell_minutes < self.dwell_long:
            # 30~60분: 선형 증가 (1.1 → 1.2)
            ratio = (avg_dwell_minutes - self.dwell_medium) / (self.dwell_long - self.dwell_medium)
            return 1.1 + ratio * 0.1
        else:
            # 60분 이상: 1.3 고정 (큰 증가)
            return 1.3
    
    def calculate_time_of_day_factor(self, hour: int) -> float:
        """
        시간대 기반 스케일링 팩터 계산
        
        Args:
            hour: 시간 (0~23)
            
        Returns:
            스케일링 팩터 (1.0~1.25)
        """
        # 새벽 시간대 (0~5시): 가장 위험 (1.25)
        if self.dawn_start <= hour < self.dawn_end:
            return 1.25
        
        # 야간 시간대 (22~23시): 위험 (1.15)
        elif hour >= self.night_start:
            return 1.15
        
        # 이른 아침 (5~7시): 약간 위험 (1.08)
        elif 5 <= hour < 7:
            return 1.08
        
        # 정상 근무 시간 (7~22시): 정상 (1.0)
        else:
            return 1.0
        
    def calculate_density_risk(self, worker_count: int, area: float) -> float:
        """
        면적 대비 인원수 위험도 계산
        
        Args:
            worker_count: 작업자 수
            area: 구역 면적 (m²)
            
        Returns:
            밀집도 위험도 (0~1)
        """
        if area <= 0:
            return 0.0
        
        density = worker_count / area
        
        if density < self.density_low:
            return 0.1  # 낮음
        elif density < self.density_medium:
            return 0.3  # 중간
        elif density < self.density_high:
            return 0.6  # 높음
        else:
            return 0.9  # 매우 높음
    
    def calculate_coexistence_risk(self, worker_count: int, equipment_count: int) -> float:
        """
        작업자+장비 혼재 위험도 계산
        
        Args:
            worker_count: 작업자 수
            equipment_count: 장비 수
            
        Returns:
            혼재 위험도 (0~1)
        """
        if worker_count == 0 or equipment_count == 0:
            return 0.0  # 혼재 없음
        
        # 작업자와 장비가 모두 있으면 위험도 상승
        # 비율에 따라 위험도 계산
        ratio = min(worker_count, equipment_count) / max(worker_count, equipment_count)
        
        if equipment_count >= 3 and worker_count >= 5:
            return 0.7 * ratio  # 고위험
        elif equipment_count >= 2 and worker_count >= 3:
            return 0.5 * ratio  # 중위험
        elif equipment_count >= 1 and worker_count >= 1:
            return 0.3 * ratio  # 저위험
        
        return 0.0
    
    def calculate_pattern_deviation_risk(
        self, 
        current_value: float, 
        avg_value: float, 
        std_value: float
    ) -> Tuple[float, float]:
        """
        과거 패턴 대비 편차 위험도 계산 (가장 중요!)
        
        Args:
            current_value: 현재 값 (작업자 수 또는 장비 수)
            avg_value: 과거 평균
            std_value: 과거 표준편차
            
        Returns:
            (편차 위험도, 표준편차 배수)
        """
        if std_value <= 0:
            std_value = max(avg_value * 0.1, 1.0)  # 최소 표준편차
        
        # 표준편차 배수 계산
        z_score = abs(current_value - avg_value) / std_value
        
        # z_score에 따른 위험도
        if z_score < self.pattern_dev_threshold_low:
            risk = 0.1  # 정상 범위
        elif z_score < self.pattern_dev_threshold_med:
            risk = 0.4  # 약간 비정상
        elif z_score < self.pattern_dev_threshold_high:
            risk = 0.7  # 비정상
        else:
            risk = 0.95  # 매우 비정상 (Critical!)
        
        return risk, z_score
    
    def calculate_confined_space_adjustment(
        self, 
        is_confined: bool, 
        worker_count: int
    ) -> float:
        """
        밀폐공간 예외 처리
        
        Args:
            is_confined: 밀폐공간 여부
            worker_count: 작업자 수
            
        Returns:
            조정 계수 (-0.5 ~ 0.0)
        """
        if not is_confined:
            return 0.0
        
        # 밀폐공간이지만 작업자가 없으면 위험도 감소
        if worker_count == 0:
            return -0.5  # 큰 감소
        elif worker_count <= 2:
            return -0.2  # 소폭 감소
        
        return 0.0  # 조정 없음
    
    def calculate_total_risk(
        self,
        worker_count: int,
        equipment_count: int,
        area: float,
        base_risk: float,
        is_confined: bool,
        avg_worker: float,
        std_worker: float,
        avg_equipment: float = 0.0,
        std_equipment: float = 0.0,
        avg_dwell_minutes: float = 15.0,
        hour_of_day: int = 12
    ) -> Dict:
        """
        종합 위험도 계산
        
        Args:
            worker_count: 현재 작업자 수
            equipment_count: 현재 장비 수
            area: 구역 면적
            base_risk: 구역 고유 위험도
            is_confined: 밀폐공간 여부
            avg_worker: 과거 평균 작업자 수
            std_worker: 과거 작업자 수 표준편차
            avg_equipment: 과거 평균 장비 수
            std_equipment: 과거 장비 수 표준편차
            avg_dwell_minutes: 평균 체류시간 (분, 기본값 15분)
            hour_of_day: 시간대 (0~23, 기본값 12시)
            
        Returns:
            위험도 상세 정보 딕셔너리
        """
        # 1. 밀집도 위험도
        density_risk = self.calculate_density_risk(worker_count, area)
        
        # 2. 구역 고유 위험도 (그대로 사용)
        zone_risk = base_risk
        
        # 3. 혼재 위험도
        coexist_risk = self.calculate_coexistence_risk(worker_count, equipment_count)
        
        # 4. 패턴 편차 위험도 (가장 중요!)
        pattern_risk_worker, z_worker = self.calculate_pattern_deviation_risk(
            worker_count, avg_worker, std_worker
        )
        
        if equipment_count > 0 and avg_equipment > 0:
            pattern_risk_equip, z_equip = self.calculate_pattern_deviation_risk(
                equipment_count, avg_equipment, std_equipment
            )
            pattern_risk = max(pattern_risk_worker, pattern_risk_equip)
            z_score_max = max(z_worker, z_equip)
        else:
            pattern_risk = pattern_risk_worker
            z_score_max = z_worker
            z_equip = 0.0
        
        # 5. 밀폐공간 조정
        confined_adj = self.calculate_confined_space_adjustment(is_confined, worker_count)
        
        # 가중 평균 계산 (기본 위험도)
        base_total_risk = (
            density_risk * self.weight_density +
            zone_risk * self.weight_base_risk +
            coexist_risk * self.weight_混재 +
            pattern_risk * self.weight_pattern_dev +
            confined_adj * self.weight_confined_adj
        )
        
        # 6. 스케일링 팩터 적용 (곱셈)
        dwell_factor = self.calculate_dwell_time_factor(avg_dwell_minutes)
        time_factor = self.calculate_time_of_day_factor(hour_of_day)
        
        # 최종 위험도 = 기본 위험도 × 체류시간 팩터 × 시간대 팩터
        total_risk = base_total_risk * dwell_factor * time_factor
        
        # 0~1 범위로 클리핑
        total_risk = np.clip(total_risk, 0.0, 1.0)
        
        # 위험 등급 결정
        if total_risk < 0.3:
            risk_level = 'Safe'
            risk_color = '#00C853'  # 녹색
        elif total_risk < 0.6:
            risk_level = 'Caution'
            risk_color = '#FFA726'  # 주황색
        else:
            risk_level = 'Critical'
            risk_color = '#D32F2F'  # 빨간색
        
        return {
            'total_risk': total_risk,
            'risk_level': risk_level,
            'risk_color': risk_color,
            # 개별 요소
            'density_risk': density_risk,
            'zone_base_risk': zone_risk,
            'coexistence_risk': coexist_risk,
            'pattern_deviation_risk': pattern_risk,
            'confined_adjustment': confined_adj,
            # 스케일링 팩터
            'dwell_time_factor': dwell_factor,
            'time_of_day_factor': time_factor,
            'base_risk_before_scaling': base_total_risk,
            # 추가 정보
            'z_score_worker': z_worker,
            'z_score_equipment': z_equip,
            'z_score_max': z_score_max,
            'worker_count': worker_count,
            'equipment_count': equipment_count,
            'area': area,
            'density': worker_count / area if area > 0 else 0.0,
            'avg_dwell_minutes': avg_dwell_minutes,
            'hour_of_day': hour_of_day
        }
    
    def get_risk_explanation(self, risk_info: Dict) -> str:
        """
        위험도 판단 근거 텍스트 생성
        
        Args:
            risk_info: calculate_total_risk() 결과
            
        Returns:
            위험도 설명 텍스트
        """
        explanations = []
        
        # 총 위험도
        total = risk_info['total_risk']
        level = risk_info['risk_level']
        explanations.append(f"**종합 위험도: {total:.2f} ({level})**")
        
        # 주요 요인 분석
        explanations.append("\n**주요 위험 요인:**")
        
        # 1. 밀집도
        density_risk = risk_info['density_risk']
        if density_risk > 0.5:
            explanations.append(f"• 밀집도: {risk_info['density']:.3f}명/m² (위험도 {density_risk:.2f}) ⚠️ 높음")
        elif density_risk > 0.3:
            explanations.append(f"• 밀집도: {risk_info['density']:.3f}명/m² (위험도 {density_risk:.2f}) 주의")
        
        # 2. 구역 고유 위험도
        zone_risk = risk_info['zone_base_risk']
        if zone_risk > 0.3:
            explanations.append(f"• 구역 특성: 기본 위험도 {zone_risk:.2f}")
        
        # 3. 혼재
        coexist = risk_info['coexistence_risk']
        if coexist > 0.3:
            explanations.append(
                f"• 작업자-장비 혼재: 작업자 {risk_info['worker_count']}명, "
                f"장비 {risk_info['equipment_count']}대 (위험도 {coexist:.2f}) ⚠️"
            )
        
        # 4. 패턴 편차 (가장 중요!)
        pattern_risk = risk_info['pattern_deviation_risk']
        z_max = risk_info['z_score_max']
        if pattern_risk > 0.6:
            explanations.append(
                f"• **비정상 패턴 감지**: 평균 대비 {z_max:.1f}σ 편차 "
                f"(위험도 {pattern_risk:.2f}) 🚨 매우 위험"
            )
        elif pattern_risk > 0.4:
            explanations.append(
                f"• 패턴 편차: 평균 대비 {z_max:.1f}σ 편차 (위험도 {pattern_risk:.2f}) ⚠️"
            )
        
        # 5. 밀폐공간 조정
        confined_adj = risk_info['confined_adjustment']
        if confined_adj < -0.1:
            explanations.append(f"• 밀폐공간이지만 작업자 없음 → 위험도 감소")
        
        return "\n".join(explanations)
