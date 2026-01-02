"""
IRFM Gateway Structure Module
==============================

Gateway(GW) 공간 계층 구조 관리
GW → Floor → Building → Sector

케이스 1: 실외 GW (floor_no=0, building_no=0)
    - Sector 좌표계 사용 (sector_coord)
    - sector_layout.png 지도에 표시

케이스 2: 실내 GW (floor_no>0, building_no>0)
    - Floor 좌표계 사용 (floor_coord)
    - floor_layout_{building}_{floor}.png 지도에 표시
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class GatewayStructure:
    """Gateway 공간 구조 관리 클래스"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Raw 데이터 폴더 경로
        """
        self.data_dir = Path(data_dir)
        
        # 데이터 로드
        self.gateway_df = pd.read_csv(self.data_dir / 'gateway.csv')
        self.sector_df = pd.read_csv(self.data_dir / 'sector.csv')
        self.building_df = pd.read_csv(self.data_dir / 'building.csv')
        self.floor_df = pd.read_csv(self.data_dir / 'floor.csv')
        self.irfm_df = pd.read_csv(self.data_dir / 'irfm.csv')
        self.spot_df = pd.read_csv(self.data_dir / 'spot.csv')
        self.spot_position_df = pd.read_csv(self.data_dir / 'spot_position.csv')
        
        # 좌표가 있는 GW만 사용 (미설치 GW 제외)
        self.gateway_df_valid = self.gateway_df[
            self.gateway_df['location_x'].notna() & 
            self.gateway_df['location_y'].notna()
        ].copy()
        
        # Gateway 구조 분석
        self._analyze_gateway_structure()
    
    def _analyze_gateway_structure(self):
        """Gateway 구조 분석 및 분류"""
        print("=" * 60)
        print("🔍 Gateway 구조 분석")
        print("=" * 60)
        
        # 전체 GW 수
        total_gw = len(self.gateway_df)
        valid_gw = len(self.gateway_df_valid)
        print(f"\n총 Gateway 수: {total_gw}개")
        print(f"설치된 GW (좌표 있음): {valid_gw}개")
        print(f"미설치 GW (좌표 없음): {total_gw - valid_gw}개 ⚠️ 제외됨")
        
        # 실외 GW (floor_no가 NaN 또는 0)
        outdoor_gw = self.gateway_df_valid[
            self.gateway_df_valid['floor_no'].isna() | 
            (self.gateway_df_valid['floor_no'] == 0)
        ]
        print(f"\n실외 GW (floor_no 없음): {len(outdoor_gw)}개")
        
        # 실내 GW (floor_no > 0)
        indoor_gw = self.gateway_df_valid[self.gateway_df_valid['floor_no'] > 0]
        print(f"실내 GW (floor_no 있음): {len(indoor_gw)}개")
        
        # GW 타입별 분류 (유효한 GW만)
        print("\n📊 GW Type 분류 (설치된 것만):")
        for gw_type in sorted(self.gateway_df_valid['type'].unique()):
            count = len(self.gateway_df_valid[self.gateway_df_valid['type'] == gw_type])
            type_name = {1: '일반용', 2: '밀폐공간용', 3: '야외용'}.get(gw_type, '기타')
            print(f"  Type {gw_type} ({type_name}): {count}개")
        
        # Floor별 GW 분포
        print("\n🏢 Floor별 GW 분포:")
        floor_counts = indoor_gw.groupby('floor_no').size().sort_index()
        for floor_no, count in floor_counts.items():
            # Floor 정보 가져오기
            floor_info = self.floor_df[self.floor_df['floor_no'] == floor_no]
            if len(floor_info) > 0:
                floor_name = floor_info.iloc[0]['name']
                building_no = floor_info.iloc[0]['building_no']
                building_info = self.building_df[self.building_df['building_no'] == building_no]
                if len(building_info) > 0:
                    building_name = building_info.iloc[0]['name']
                    print(f"  Floor {floor_no} ({building_name} - {floor_name}): {count}개")
                else:
                    print(f"  Floor {floor_no} ({floor_name}): {count}개")
        
        # 실외 GW 상세 정보
        if len(outdoor_gw) > 0:
            print("\n📍 실외 GW 목록:")
            for _, gw in outdoor_gw.iterrows():
                print(f"  - GW {gw['gateway_no']} ({gw['code']}): "
                      f"({gw['location_x']}, {gw['location_y']})")
    
    def classify_gateway(self, gateway_no: int) -> Dict:
        """Gateway 위치 분류
        
        Args:
            gateway_no: Gateway 번호
            
        Returns:
            dict: {
                'location': 'indoor' or 'outdoor',
                'coord_system': 'sector_coord' or 'floor_coord',
                'sector_no': int,
                'building_no': int (실내인 경우),
                'floor_no': int (실내인 경우),
                'building_name': str (실내인 경우),
                'floor_name': str (실내인 경우),
                'map_file': str,
                'x': float,
                'y': float
            }
        """
        gw = self.gateway_df[self.gateway_df['gateway_no'] == gateway_no]
        
        if len(gw) == 0:
            raise ValueError(f"Gateway {gateway_no} not found")
        
        gw = gw.iloc[0]
        floor_no = gw['floor_no']
        sector_no = gw['sector_no']
        location_x = gw['location_x']
        location_y = gw['location_y']
        
        result = {
            'gateway_no': gateway_no,
            'code': gw['code'],
            'name': gw['name'],
            'type': gw['type'],
            'sector_no': sector_no,
            'x': location_x,
            'y': location_y
        }
        
        # 케이스 1: 실외 GW (floor_no == 0)
        if floor_no == 0:
            result.update({
                'location': 'outdoor',
                'coord_system': 'sector_coord',
                'map_file': 'sector_layout.png'
            })
        
        # 케이스 2: 실내 GW (floor_no > 0)
        else:
            # Floor 정보 가져오기
            floor_info = self.floor_df[self.floor_df['floor_no'] == floor_no]
            if len(floor_info) == 0:
                raise ValueError(f"Floor {floor_no} not found")
            
            floor_info = floor_info.iloc[0]
            building_no = floor_info['building_no']
            floor_name = floor_info['name']
            
            # Building 정보 가져오기
            building_info = self.building_df[self.building_df['building_no'] == building_no]
            if len(building_info) == 0:
                raise ValueError(f"Building {building_no} not found")
            
            building_name = building_info.iloc[0]['name']
            
            result.update({
                'location': 'indoor',
                'coord_system': 'floor_coord',
                'building_no': building_no,
                'building_name': building_name,
                'floor_no': floor_no,
                'floor_name': floor_name,
                'map_file': f'floor_layout_{building_name}_{floor_name}.png'
            })
        
        return result
    
    def get_gateways_by_location(self, location: str = None, 
                                  building_no: int = None, 
                                  floor_no: int = None) -> pd.DataFrame:
        """특정 위치의 Gateway 목록 가져오기
        
        Args:
            location: 'indoor' or 'outdoor'
            building_no: Building 번호 (실내인 경우)
            floor_no: Floor 번호 (실내인 경우)
            
        Returns:
            DataFrame: Gateway 목록
        """
        result = self.gateway_df.copy()
        
        if location == 'outdoor':
            result = result[result['floor_no'] == 0]
        elif location == 'indoor':
            result = result[result['floor_no'] > 0]
        
        if building_no is not None:
            # Floor 정보에서 해당 building의 floor 찾기
            floors = self.floor_df[self.floor_df['building_no'] == building_no]['floor_no']
            result = result[result['floor_no'].isin(floors)]
        
        if floor_no is not None:
            result = result[result['floor_no'] == floor_no]
        
        return result
    
    def get_gateway_summary(self) -> pd.DataFrame:
        """설치된 Gateway 요약 정보 (좌표 있는 것만, 실외 포함)
        
        Returns:
            DataFrame: Gateway 요약 (location, building, floor 정보 포함)
        """
        summary_list = []
        
        for _, gw in self.gateway_df_valid.iterrows():
            try:
                info = self.classify_gateway(gw['gateway_no'])
                summary_list.append(info)
            except Exception as e:
                # 실외 Gateway는 floor_no가 NaN일 수 있으므로 직접 처리
                if pd.isna(gw['floor_no']) or gw['floor_no'] == 0:
                    summary_list.append({
                        'gateway_no': gw['gateway_no'],
                        'code': gw['code'],
                        'name': gw['name'],
                        'type': gw['type'],
                        'location': 'outdoor',
                        'coord_system': 'sector_coord',
                        'sector_no': gw['sector_no'],
                        'building_no': 0,
                        'floor_no': 0,
                        'x': gw['location_x'],
                        'y': gw['location_y']
                    })
                else:
                    print(f"Warning: Gateway {gw['gateway_no']} 처리 오류: {e}")
        
        return pd.DataFrame(summary_list)
    
    def validate_coordinates(self) -> dict:
        """좌표 유효성 검증
        
        Returns:
            dict: 검증 결과 통계
        """
        print("\n" + "=" * 60)
        print("🔍 좌표 유효성 검증")
        print("=" * 60)
        
        issues = {
            'missing_coords': [],
            'outdoor_out_of_sector': [],
            'indoor_out_of_floor': []
        }
        
        # Sector 크기 가져오기
        sector_info = self.irfm_df[self.irfm_df['building_number'] == 0].iloc[0]
        sector_width = sector_info['length_x']
        sector_height = sector_info['length_y']
        
        for _, gw in self.gateway_df.iterrows():
            gw_no = gw['gateway_no']
            x = gw['location_x']
            y = gw['location_y']
            
            # 좌표 결측 체크
            if pd.isna(x) or pd.isna(y):
                issues['missing_coords'].append(gw_no)
                continue
            
            # 실외 GW: Sector 범위 체크
            if gw['floor_no'] == 0:
                if x < 0 or x > sector_width or y < 0 or y > sector_height:
                    issues['outdoor_out_of_sector'].append({
                        'gateway_no': gw_no,
                        'x': x,
                        'y': y,
                        'sector_bounds': f"(0-{sector_width}, 0-{sector_height})"
                    })
            
            # 실내 GW: Floor 범위 체크
            else:
                floor_info = self.irfm_df[self.irfm_df['floor_number'] == gw['floor_no']]
                if len(floor_info) > 0:
                    floor_info = floor_info.iloc[0]
                    floor_width = floor_info['length_x']
                    floor_height = floor_info['length_y']
                    
                    if x < 0 or x > floor_width or y < 0 or y > floor_height:
                        issues['indoor_out_of_floor'].append({
                            'gateway_no': gw_no,
                            'floor_no': gw['floor_no'],
                            'x': x,
                            'y': y,
                            'floor_bounds': f"(0-{floor_width}, 0-{floor_height})"
                        })
        
        # 결과 출력
        print(f"\n❌ 좌표 없음: {len(issues['missing_coords'])}개")
        if issues['missing_coords']:
            print(f"   GW 번호: {issues['missing_coords']}")
        
        print(f"\n⚠️ Sector 범위 벗어남: {len(issues['outdoor_out_of_sector'])}개")
        for item in issues['outdoor_out_of_sector'][:5]:  # 최대 5개만 표시
            print(f"   GW {item['gateway_no']}: ({item['x']}, {item['y']}) "
                  f"범위: {item['sector_bounds']}")
        
        print(f"\n⚠️ Floor 범위 벗어남: {len(issues['indoor_out_of_floor'])}개")
        for item in issues['indoor_out_of_floor'][:5]:  # 최대 5개만 표시
            print(f"   GW {item['gateway_no']} (Floor {item['floor_no']}): "
                  f"({item['x']}, {item['y']}) 범위: {item['floor_bounds']}")
        
        return issues
    
    def visualize_gateways(self, output_dir: str = '../Gateway'):
        """Gateway 위치를 지도 위에 시각화
        
        Args:
            output_dir: 이미지 저장 폴더 (기본값: ../Gateway)
        """
        output_path = Path(self.data_dir).parent.parent / output_dir.lstrip('../')
        output_path.mkdir(exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        print("\n" + "=" * 60)
        print("🎨 Gateway 지도 생성 중...")
        print("=" * 60)
        
        # 1. Sector 전체 지도 (실외 GW)
        self._draw_sector_gateways(output_path)
        
        # 2. Floor별 지도 (실내 GW)
        self._draw_floor_gateways(output_path)
        
        print(f"\n✅ 모든 Gateway 지도 생성 완료: {output_path}")
    
    def _draw_sector_gateways(self, output_path: Path):
        """Sector 지도에 실외 Gateway와 Spot 표시"""
        # Sector 크기 가져오기
        sector_info = self.irfm_df[self.irfm_df['building_number'] == 0].iloc[0]
        sector_width = sector_info['length_x']
        sector_height = sector_info['length_y']
        
        # 실외 GW (floor_no가 NaN 또는 0)
        outdoor_gw = self.gateway_df_valid[
            self.gateway_df_valid['floor_no'].isna() | 
            (self.gateway_df_valid['floor_no'] == 0)
        ]
        
        if len(outdoor_gw) == 0:
            print("  ⚠️ 실외 Gateway 없음 - sector_layout 건너뜀")
            return
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_xlim(0, sector_width)
        ax.set_ylim(0, sector_height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Sector Layout - Spots & Gateways ({len(outdoor_gw)}개 GW)', 
                     fontsize=14, fontweight='bold')
        
        # Building 경계선 그리기
        for _, building in self.building_df.iterrows():
            building_info = self.irfm_df[
                self.irfm_df['building_number'] == building['building_no']
            ].iloc[0]
            
            # irfm.csv는 Sector_coord_x1, Sector_coord_y1 등을 사용
            x = building_info['Sector_coord_x1']
            y = building_info['Sector_coord_y2']  # y2가 bottom
            w = building_info['length_x']
            h = building_info['length_y']
            
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                      edgecolor='gray', facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, building['name'], 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Spot 그리기 (실외 Spot만)
        outdoor_spots = self.spot_df[
            self.spot_df['floor_no'].isna() | (self.spot_df['floor_no'] == '')
        ]
        
        for _, spot in outdoor_spots.iterrows():
            spot_no = spot['spot_no']
            spot_name = spot['name']
            spot_color = spot['color']
            
            # 해당 spot의 모든 position 가져오기
            positions = self.spot_position_df[
                self.spot_position_df['spot_no'] == spot_no
            ].sort_values('point_no')
            
            if len(positions) < 3:
                continue
            
            # x, y 좌표 추출
            coords = positions[['x', 'y']].dropna()
            if len(coords) < 3:
                continue
            
            # 다각형 그리기
            try:
                polygon = patches.Polygon(
                    coords.values,
                    linewidth=1,
                    edgecolor='gray',
                    facecolor=f'#{spot_color}' if pd.notna(spot_color) and spot_color != '' else '#CCCCCC',
                    alpha=0.3,
                    zorder=1
                )
                ax.add_patch(polygon)
                
                # Spot 이름 표시 (중심점)
                center_x = coords['x'].mean()
                center_y = coords['y'].mean()
                ax.text(center_x, center_y, spot_name,
                       fontsize=7, ha='center', va='center', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.7, edgecolor='none'),
                       zorder=2)
            except:
                pass
        
        # Gateway 표시
        gw_types = outdoor_gw['type'].unique()
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
        markers = {0: 's', 1: 'o', 2: '^', 3: 'D'}
        type_names = {0: '야외용(type 0)', 1: '일반용(type 1)', 
                     2: '밀폐공간용(type 2)', 3: '야외용(type 3)'}
        
        for gw_type in gw_types:
            gw_subset = outdoor_gw[outdoor_gw['type'] == gw_type]
            ax.scatter(gw_subset['location_x'], gw_subset['location_y'], 
                      c=colors.get(gw_type, 'black'), 
                      marker=markers.get(gw_type, 'o'), 
                      s=100, alpha=0.7, 
                      label=f'{type_names.get(gw_type, f"Type {gw_type}")} ({len(gw_subset)}개)')
        
        # Gateway 번호 표시 (일부만)
        for _, gw in outdoor_gw.head(20).iterrows():  # 처음 20개만
            ax.annotate(f"{int(gw['gateway_no'])}", 
                       (gw['location_x'], gw['location_y']),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=7, alpha=0.7)
        
        ax.legend(loc='upper right', fontsize=10)
        
        # 저장
        output_file = output_path / 'sector_gateways.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Sector Gateway 지도: {output_file.name} ({len(outdoor_gw)}개 GW)")
    
    def _draw_floor_gateways(self, output_path: Path):
        """Floor별 지도에 실내 Gateway와 Spot 표시"""
        # 실내 GW (floor_no > 0)
        indoor_gw = self.gateway_df_valid[self.gateway_df_valid['floor_no'] > 0]
        
        if len(indoor_gw) == 0:
            print("  ⚠️ 실내 Gateway 없음 - floor layout 건너뜀")
            return
        
        # Floor별로 그룹화
        floor_groups = indoor_gw.groupby('floor_no')
        
        for floor_no, gw_group in floor_groups:
            # Floor 정보 가져오기
            floor_info_df = self.floor_df[self.floor_df['floor_no'] == floor_no]
            if len(floor_info_df) == 0:
                continue
            
            floor_info_row = floor_info_df.iloc[0]
            building_no = floor_info_row['building_no']
            floor_name = floor_info_row['name']
            
            # Building 정보
            building_info = self.building_df[self.building_df['building_no'] == building_no]
            if len(building_info) == 0:
                continue
            building_name = building_info.iloc[0]['name']
            
            # Floor 크기 (irfm.csv에서)
            floor_irfm = self.irfm_df[self.irfm_df['floor_number'] == floor_no]
            if len(floor_irfm) == 0:
                continue
            
            floor_irfm = floor_irfm.iloc[0]
            floor_width = floor_irfm['length_x']
            floor_height = floor_irfm['length_y']
            
            # 그림 생성
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, floor_width)
            ax.set_ylim(0, floor_height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title(f'{building_name} - {floor_name} Spots & Gateways ({len(gw_group)}개 GW)', 
                        fontsize=14, fontweight='bold')
            
            # Floor 경계선
            rect = patches.Rectangle((0, 0), floor_width, floor_height, 
                                     linewidth=2, edgecolor='black', 
                                     facecolor='lightgray', alpha=0.2)
            ax.add_patch(rect)
            
            # 해당 floor의 Spot 그리기
            floor_spots = self.spot_df[self.spot_df['floor_no'] == floor_no]
            
            for _, spot in floor_spots.iterrows():
                spot_no = spot['spot_no']
                spot_name = spot['name']
                spot_color = spot['color']
                
                # 해당 spot의 모든 position 가져오기
                positions = self.spot_position_df[
                    self.spot_position_df['spot_no'] == spot_no
                ].sort_values('point_no')
                
                if len(positions) < 3:
                    continue
                
                # x, y 좌표 추출
                coords = positions[['x', 'y']].dropna()
                if len(coords) < 3:
                    continue
                
                # 다각형 그리기
                try:
                    polygon = patches.Polygon(
                        coords.values,
                        linewidth=1.5,
                        edgecolor='darkgray',
                        facecolor=f'#{spot_color}' if pd.notna(spot_color) and spot_color != '' else '#CCCCCC',
                        alpha=0.4,
                        zorder=1
                    )
                    ax.add_patch(polygon)
                    
                    # Spot 이름 표시
                    center_x = coords['x'].mean()
                    center_y = coords['y'].mean()
                    ax.text(center_x, center_y, spot_name,
                           fontsize=8, ha='center', va='center', 
                           color='black', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.8, edgecolor='gray', linewidth=0.5),
                           zorder=2)
                except:
                    pass
            
            # Gateway 표시 (타입별 색상)
            gw_types = gw_group['type'].unique()
            colors = {1: 'blue', 2: 'green', 3: 'orange'}
            markers = {1: 'o', 2: '^', 3: 'D'}
            type_names = {1: '일반용', 2: '밀폐공간용', 3: '야외용'}
            
            for gw_type in gw_types:
                gw_subset = gw_group[gw_group['type'] == gw_type]
                ax.scatter(gw_subset['location_x'], gw_subset['location_y'], 
                          c=colors.get(gw_type, 'blue'), 
                          marker=markers.get(gw_type, 'o'), 
                          s=120, alpha=0.7, 
                          label=f'{type_names.get(gw_type, f"Type {gw_type}")} ({len(gw_subset)}개)')
            
            # Gateway 번호 표시
            for _, gw in gw_group.iterrows():
                ax.annotate(f"{int(gw['gateway_no'])}", 
                           (gw['location_x'], gw['location_y']),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            ax.legend(loc='upper right', fontsize=10)
            
            # 저장
            output_file = output_path / f'floor_gateways_{building_name}_{floor_name}.png'
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ {building_name} {floor_name}: {output_file.name} ({len(gw_group)}개 GW)")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # 데이터 폴더 경로
    data_dir = '/Users/Tony_mac/Desktop/TJLABS/TJLABS_Research/Project/SKEP/IRFM_demo_new/Datafile/Yongin_Cluster_202512010'
    
    # Gateway 구조 분석
    gw_structure = GatewayStructure(data_dir)
    
    # 요약 정보 생성
    summary_df = gw_structure.get_gateway_summary()
    
    # 요약 저장
    output_path = Path(data_dir).parent.parent / 'src' / 'gateway_summary.csv'
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Gateway 요약 저장: {output_path}")
    
    # 좌표 유효성 검증
    validation_result = gw_structure.validate_coordinates()
    
    # Gateway 지도 생성
    gw_structure.visualize_gateways()
    
    # 샘플 출력
    print("\n" + "=" * 60)
    print("📋 Gateway 분류 샘플 (처음 5개)")
    print("=" * 60)
    for i, (_, row) in enumerate(summary_df.head().iterrows()):
        print(f"\n{i+1}. GW {row['gateway_no']} ({row['code']})")
        print(f"   위치: {row['location']}")
        print(f"   좌표계: {row['coord_system']}")
        if row['location'] == 'indoor':
            print(f"   건물/층: {row['building_name']} - {row['floor_name']}")
        print(f"   좌표: ({row['x']}, {row['y']})")
        print(f"   지도: {row['map_file']}")
