"""
Logging Configuration for DeepCon
==================================

중앙 집중식 로깅 설정
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class DeepConLogger:
    """DeepCon 애플리케이션을 위한 로거"""
    
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = "deepcon") -> logging.Logger:
        """싱글톤 로거 인스턴스 반환"""
        if cls._instance is None:
            cls._setup_logger(name)
        return cls._instance
    
    @classmethod
    def _setup_logger(cls, name: str):
        """로거 초기 설정"""
        # 로그 디렉토리 생성
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 로거 생성
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 (중복 방지)
        logger.handlers.clear()
        
        # 파일 핸들러 (날짜별 로그 파일)
        log_file = log_dir / f"deepcon_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        cls._instance = logger
        logger.info("=" * 80)
        logger.info("DeepCon Logger Initialized")
        logger.info("=" * 80)
    
    @classmethod
    def log_error(cls, error: Exception, context: str = ""):
        """에러 로깅 헬퍼"""
        logger = cls.get_logger()
        logger.error(f"{context}: {type(error).__name__}: {str(error)}", exc_info=True)
    
    @classmethod
    def log_performance(cls, operation: str, duration: float):
        """성능 로깅 헬퍼"""
        logger = cls.get_logger()
        logger.info(f"[PERFORMANCE] {operation}: {duration:.2f}s")

# 편의를 위한 전역 로거
logger = DeepConLogger.get_logger()
