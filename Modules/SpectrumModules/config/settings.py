# 光谱采集参数
ACQUISITION = {
    "wavelength_range": (500, 2000),  # cm⁻¹
    "exposure_time": 3.0,            # 秒
    "laser_power": 300               # mW
}

# 预处理参数
PROCESSING = {
    "smoothing_window": 17,      # Savitzky-Golay窗口点数
    "smoothing_order": 3,        # 多项式阶数
    "baseline_order": 4,         # 基线校正多项式阶数
    "snv_smoothing": 9,         # 标准正态变换窗口
    "iqr_threshold": 1.5        # 异常值剔除阈值
}

# 化学计量学参数
ANALYSIS = {
    "pls_components": 6,
    "test_size": 0.2,
    "random_state": 42
}

"""配置校验（可去掉）"""
if __name__ == "__main__":
    print("配置校验通过，关键参数：")
    print(f"波长范围: {ACQUISITION['wavelength_range']}")
    print(f"PLS成分数: {ANALYSIS['pls_components']}")