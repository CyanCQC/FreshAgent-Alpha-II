"""
配置中心模块
--------------------
集中管理系统所有光谱处理参数

包含配置组：
- ACQUISITION: 光谱采集参数
- PROCESSING: 光谱预处理参数
- ANALYSIS: 化学计量学参数
"""

# 显式导入所有配置参数
from .settings import (
    ACQUISITION,
    PROCESSING,
    ANALYSIS
)

# 参数合法性校验
def _validate_config():
    """配置校验流程"""
    # 校验必要参数组存在性
    required_sections = {
        "ACQUISITION": ["wavelength_range", "exposure_time", "laser_power"],
        "PROCESSING": ["smoothing_window", "smoothing_order", "baseline_order"],
        "ANALYSIS": ["pls_components", "test_size"]
    }

    # 参数组存在性检查
    for section, params in required_sections.items():
        if section not in globals():
            raise ValueError(f"关键配置组缺失: {section}")
        # 参数项完整性检查
        missing_params = [p for p in params if p not in globals()[section]]
        if missing_params:
            raise ValueError(f"{section}组缺失必要参数: {', '.join(missing_params)}")

    # 参数值范围校验
    if not (100 <= ACQUISITION["laser_power"] <= 500):
        raise ValueError("激光功率需在100-500mW范围内")
    
    if PROCESSING["smoothing_window"] % 2 == 0:
        raise ValueError("平滑窗口必须为奇数")
    
    if not 0 < ANALYSIS["test_size"] < 1:
        raise ValueError("测试集比例需在0-1之间")

# 执行自动校验（当模块被导入时）
_validate_config()

# 定义模块导出白名单
__all__ = [
    "ACQUISITION",
    "PROCESSING",
    "ANALYSIS"
]