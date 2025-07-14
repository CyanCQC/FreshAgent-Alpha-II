import numpy as np
from core.preprocessing import SpectralProcessor
from core.analysis import ChemometricsAnalyzer
from utils.visualization import plot_spectra

"""模拟数据加载"""
"""举例
wavelengths = np.linspace(500, 2000, 1000)
raw_spectra = np.random.randn(100, 1000)  # 100个样本
labels = np.random.uniform(8, 15, 100)     # 糖度值
"""

"""光谱预处理"""
processor = SpectralProcessor(wavelengths)
processed = processor.full_pipeline(raw_spectra)

"""化学计量学建模"""
analyzer = ChemometricsAnalyzer()
results = analyzer.train(processed, labels)

"""输出结果"""
print("模型性能：")
for metric in results:
    print(f"{metric}:")
    print(f"  RMSE: {results[metric]['RMSE']:.3f}")
    print(f"  R²: {results[metric]['R²']:.2f}")
    print(f"  Bias: {results[metric]['Bias']:.3f}\n")
print("关键波长：", analyzer.get_important_wavelengths())

"""可视化处理效果"""
plot_spectra(wavelengths, raw_spectra, processed)