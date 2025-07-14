import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    confusion_matrix
)
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-whitegrid')

# --------------------------
# 指标计算模块
# --------------------------
class RegressionMetrics:
    @staticmethod
    def calculate_all(y_true, y_pred):
        """计算全部回归指标
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            
        Returns:
            dict: 包含R²、RMSE等指标的字典
        """
        return {
            "R²": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "Error_Distribution": y_pred - y_true
        }

class ClassificationMetrics:
    @staticmethod
    def calculate_all(y_true, y_prob, threshold=0.5):
        """计算全部分类指标
        Args:
            y_true: 真实标签数组
            y_prob: 预测概率数组
            threshold: 分类阈值 (默认0.5)
            
        Returns:
            dict: 包含AUC、F1等指标的字典
        """
        y_pred = (y_prob > threshold).astype(int)
        return {
            "AUC": roc_auc_score(y_true, y_prob),
            "F1": f1_score(y_true, y_pred),
            "Confusion_Matrix": confusion_matrix(y_true, y_pred)
        }

# --------------------------
# 可视化模块
# --------------------------
class SpectralVisualizer:
    @staticmethod
    def plot_spectral_comparison(raw, processed, wavelength):
        """原始与处理光谱对比可视化
        Args:
            raw: 原始光谱数据 (样本数 x 波长数)
            processed: 处理后的光谱数据
            wavelength: 波长数组
        """
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)
        
        # 光谱曲线对比
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(wavelength, raw.T, alpha=0.3, color='blue', label='Raw')
        ax1.plot(wavelength, processed.T, alpha=0.3, color='red', label='Processed')
        ax1.set_title("Spectral Profile Comparison")
        ax1.set_xlabel("Wavenumber (cm⁻¹)")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.legend()
        
        # 数据分布直方图
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(raw.flatten(), bins=50, color='blue', kde=True, ax=ax2)
        ax2.set_title("Raw Data Distribution")
        
        ax3 = fig.add_subplot(gs[1, 1])
        sns.histplot(processed.flatten(), bins=50, color='red', kde=True, ax=ax3)
        ax3.set_title("Processed Data Distribution")
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_chemometrics(model, wavelength):
        """PLS回归系数可视化
        Args:
            model: 训练好的PLS模型
            wavelength: 波长数组
        """
        plt.figure(figsize=(10, 6))
        plt.bar(wavelength, model.coefficients_.flatten(), width=1.5)
        plt.title("PLS Regression Coefficients")
        plt.xlabel("Wavelength (cm⁻¹)")
        plt.ylabel("Coefficient Value")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()

    @staticmethod
    def plot_qc_metrics(qc_report):
        """质量控制系统可视化面板
        Args:
            qc_report: 包含以下键的字典:
                - snr_values: 信噪比数组
                - peak_shift: 峰位漂移数据
                - peak_shift_threshold: 漂移阈值
                - laser_power: 激光功率记录
                - valid_samples: 有效样本数
                - invalid_samples: 无效样本数
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # SNR分布箱线图
        sns.boxplot(data=qc_report['snr_values'], ax=axes[0])
        axes[0].set_title("SNR Distribution")
        
        # 峰位漂移趋势图
        axes[1].plot(qc_report['peak_shift'], 'o-')
        axes[1].axhline(qc_report['peak_shift_threshold'], color='r', linestyle='--')
        axes[1].set_title("Characteristic Peak Shift")
        
        # 激光功率稳定性
        axes[2].plot(qc_report['laser_power'], 's-', markersize=8)
        axes[2].set_title("Laser Power Stability")
        
        # 数据完整性柱状图
        axes[3].bar(['Valid', 'Invalid'], 
                  [qc_report['valid_samples'], qc_report['invalid_samples']])
        axes[3].set_title("Data Integrity Check")
        
        plt.tight_layout()

# --------------------------
# 综合报告模块
# --------------------------
class QualityEvaluator:
    @staticmethod
    def hybrid_report(y_reg_true, y_reg_pred, y_cls_true, y_cls_prob):
        """生成综合质量评估报告
        Args:
            y_reg_true: 回归真实值
            y_reg_pred: 回归预测值
            y_cls_true: 分类真实标签
            y_cls_prob: 分类预测概率
            
        Returns:
            dict: 包含糖度水分和农药风险的综合报告
        """
        return {
            "Sugar_Quality": RegressionMetrics.calculate_all(y_reg_true, y_reg_pred),
            "Pesticide_Risk": ClassificationMetrics.calculate_all(y_cls_true, y_cls_prob)
        }