import numpy as np

class QualityChecker:
    @staticmethod
    def validate_ranges(y_pred, ranges):
        """验证预测值是否在合理范围内"""
        alerts = {}
        for i, (low, high) in enumerate(ranges):
            outliers = np.where((y_pred[:,i] < low) | (y_pred[:,i] > high))[0]
            alerts[ANALYSIS["target_names"][i]] = {  
                "outliers": outliers,
                "message": f"超出范围[{low}, {high}]的样本数: {len(outliers)}"
            }
        return alerts

# 合理范围配置
VALIDATION_RANGES = [
    (0, 100),    # 水分（Moisture）
    (5, 25),     # 糖度（Brix）
    (1, 10)      # 硬度（Hardness）
]               