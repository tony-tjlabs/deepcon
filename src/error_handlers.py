"""
Error Handling Utilities
=========================

일관된 에러 처리를 위한 유틸리티
"""

import streamlit as st
import traceback
from functools import wraps
from typing import Callable, Any, Optional
from src.logging_config import DeepConLogger

logger = DeepConLogger.get_logger()

def handle_errors(
    default_return: Any = None,
    error_message: str = "오류가 발생했습니다.",
    show_details: bool = False
):
    """
    에러 핸들링 데코레이터
    
    Args:
        default_return: 에러 발생 시 반환할 기본값
        error_message: 사용자에게 표시할 에러 메시지
        show_details: 개발 환경에서 상세 에러 표시 여부
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 로그에 에러 기록
                logger.error(
                    f"Error in {func.__name__}: {type(e).__name__}: {str(e)}",
                    exc_info=True
                )
                
                # 사용자에게 에러 표시
                st.error(f"❌ {error_message}")
                
                # 개발 환경에서는 상세 정보 표시
                if show_details:
                    with st.expander("상세 에러 정보 (개발용)"):
                        st.code(traceback.format_exc())
                
                return default_return
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, **kwargs) -> Optional[Any]:
    """
    함수를 안전하게 실행하고 에러 발생 시 None 반환
    
    Args:
        func: 실행할 함수
        *args: 함수 인자
        **kwargs: 함수 키워드 인자
    
    Returns:
        함수 실행 결과 또는 None
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in safe_execute({func.__name__}): {e}", exc_info=True)
        return None

def validate_dataframe(df, required_columns: list, context: str = "") -> bool:
    """
    DataFrame 유효성 검사
    
    Args:
        df: 검사할 DataFrame
        required_columns: 필수 컬럼 리스트
        context: 에러 메시지에 포함할 컨텍스트
    
    Returns:
        유효성 여부
    """
    if df is None or df.empty:
        logger.warning(f"{context}: DataFrame is None or empty")
        return False
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"{context}: Missing columns: {missing_cols}")
        return False
    
    return True

class DataValidationError(Exception):
    """데이터 유효성 검사 실패 시 발생하는 예외"""
    pass

class CacheError(Exception):
    """캐시 관련 에러"""
    pass

class ConfigurationError(Exception):
    """설정 관련 에러"""
    pass
