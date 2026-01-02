"""
Chart Utilities
===============
Common chart layout and styling utilities for Plotly visualizations
"""
from src.components.styles import THEME


def get_chart_layout(title='', height=400, show_legend=True):
    """
    Plotly 차트 기본 레이아웃 반환 - 모든 텍스트 색상 명시적 설정
    
    Args:
        title: Chart title
        height: Chart height in pixels
        show_legend: Whether to show legend
        
    Returns:
        Dictionary of Plotly layout parameters
    """
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=THEME['text_primary'])
        ),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=THEME['text_primary'], size=12),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.08)',
            linecolor=THEME['border'],
            tickfont=dict(color=THEME['text_secondary']),
            title_font=dict(color=THEME['text_secondary'])
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.08)',
            linecolor=THEME['border'],
            tickfont=dict(color=THEME['text_secondary']),
            title_font=dict(color=THEME['text_secondary'])
        ),
        legend=dict(
            font=dict(color=THEME['text_primary']),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ) if show_legend else dict(visible=False),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode='x unified'
    )
