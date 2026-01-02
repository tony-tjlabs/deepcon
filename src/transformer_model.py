"""
위험도 예측용 Transformer 모델
================================

Zone 시퀀스 → 30분 후 위험도
"""

import torch
import torch.nn as nn
import math
from pathlib import Path
import pandas as pd

# Zone 수 동적 로드
def get_num_zones():
    """spot.csv에서 zone 수 가져오기"""
    data_dir = Path(__file__).parent.parent / 'Datafile' / 'Yongin_Cluster_202512010'
    spot_df = pd.read_csv(data_dir / 'spot.csv')
    return len(spot_df['spot_no'].unique())

NUM_ZONES = get_num_zones()

class PositionalEncoding(nn.Module):
    """시간 정보 인코딩"""
    def __init__(self, d_model: int, max_len: int = 288):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: [seq_len, batch, d_model]"""
        return x + self.pe[:x.size(0)]

class RiskTransformer(nn.Module):
    """Zone 위험도 예측 Transformer"""
    
    def __init__(
        self,
        num_zones: int = None,  # None이면 자동으로 spot.csv에서 로드
        input_features: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if num_zones is None:
            num_zones = NUM_ZONES
        
        self.num_zones = num_zones
        self.d_model = d_model
        
        # 1. 입력 임베딩: [33 zones × 7 features] → [d_model]
        self.input_projection = nn.Linear(num_zones * input_features, d_model)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq, batch, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 4. 출력 헤드: [d_model] → [33 zones]
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_zones),
            nn.Sigmoid()  # 위험도는 0~1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        입력: [batch, seq_len, zones, features]
        출력: [batch, zones]
        """
        batch_size, seq_len, zones, features = x.shape
        
        # Reshape: [batch, seq, zones, feat] → [batch, seq, zones*feat]
        x = x.reshape(batch_size, seq_len, -1)
        
        # 입력 projection: [batch, seq, zones*feat] → [batch, seq, d_model]
        x = self.input_projection(x)
        
        # Transpose for transformer: [batch, seq, d_model] → [seq, batch, d_model]
        x = x.transpose(0, 1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # [seq, batch, d_model]
        
        # 마지막 타임스텝 사용
        last_output = encoded[-1]  # [batch, d_model]
        
        # 위험도 예측
        risk_pred = self.output_head(last_output)  # [batch, zones]
        
        return risk_pred

class RiskLoss(nn.Module):
    """위험도 예측용 손실 함수
    
    MSE + Zone별 가중치 (밀폐공간에 더 높은 가중치)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target, zone_weights=None):
        """
        pred: [batch, zones]
        target: [batch, zones]
        zone_weights: [zones] (optional)
        """
        loss = self.mse(pred, target)  # [batch, zones]
        
        if zone_weights is not None:
            loss = loss * zone_weights.unsqueeze(0)
        
        return loss.mean()

def create_model(device='cpu'):
    """모델 생성"""
    model = RiskTransformer(
        num_zones=None,  # 자동으로 spot.csv에서 로드
        input_features=7,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    )
    return model.to(device)

if __name__ == '__main__':
    # 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Zone 수: {NUM_ZONES}")
    
    model = create_model(device)
    
    # 더미 입력
    batch_size = 8
    x = torch.randn(batch_size, 6, NUM_ZONES, 7).to(device)
    
    # Forward
    y_pred = model(x)
    print(f"입력: {x.shape}")
    print(f"출력: {y_pred.shape}")
    
    # 파라미터 수
    num_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터: {num_params:,}개")
