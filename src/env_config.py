"""
Environment Configuration
=========================

환경 변수 및 설정 관리
"""

import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class AppConfig:
    """애플리케이션 설정"""
    
    # Application
    title: str = "SK Hynix Y1 Cluster - IRFM Dashboard"
    env: str = "production"
    
    # Security
    password: str = "admin"  # Default - 프로덕션에서는 반드시 환경변수 사용
    
    # Paths
    data_dir: Path = Path("Datafile/Yongin_Cluster_202512010")
    cache_dir: Path = Path("Cache")
    
    # Server
    server_port: int = 8501
    server_address: str = "localhost"
    server_headless: bool = True
    
    # Performance
    cache_ttl: int = 3600
    max_workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/deepcon.log"
    
    # Features
    enable_transformer: bool = True
    enable_forecast: bool = True
    enable_simulator: bool = True
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """환경 변수에서 설정 로드"""
        
        # .env 파일 로드 시도
        env_file = Path(".env")
        if env_file.exists():
            cls._load_env_file(env_file)
        
        return cls(
            title=os.getenv("APP_TITLE", cls.title),
            env=os.getenv("APP_ENV", cls.env),
            password=os.getenv("APP_PASSWORD", cls.password),
            data_dir=Path(os.getenv("DATA_DIR", str(cls.data_dir))),
            cache_dir=Path(os.getenv("CACHE_DIR", str(cls.cache_dir))),
            server_port=int(os.getenv("STREAMLIT_SERVER_PORT", str(cls.server_port))),
            server_address=os.getenv("STREAMLIT_SERVER_ADDRESS", cls.server_address),
            server_headless=os.getenv("STREAMLIT_SERVER_HEADLESS", str(cls.server_headless)).lower() == "true",
            cache_ttl=int(os.getenv("CACHE_TTL", str(cls.cache_ttl))),
            max_workers=int(os.getenv("MAX_WORKERS", str(cls.max_workers))),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            log_file=os.getenv("LOG_FILE", cls.log_file),
            enable_transformer=os.getenv("ENABLE_TRANSFORMER", str(cls.enable_transformer)).lower() == "true",
            enable_forecast=os.getenv("ENABLE_FORECAST", str(cls.enable_forecast)).lower() == "true",
            enable_simulator=os.getenv("ENABLE_SIMULATOR", str(cls.enable_simulator)).lower() == "true",
        )
    
    @staticmethod
    def _load_env_file(env_file: Path):
        """간단한 .env 파일 파서"""
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # 따옴표 제거
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.env.lower() == "production"
    
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.env.lower() == "development"

# 전역 설정 인스턴스
app_config = AppConfig.from_env()
