# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
import joblib
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class CitrusQualityModel:
    def __init__(self, wave_list=np.arange(410, 941, 25)):
        self.bands = wave_list
        self.models = {}
        self.scalers = {}
        self.y_stats = {}

    def _sg_derivative(self, X, window_length=5, polyorder=2, deriv=1):
        window_length = min(window_length, X.shape[1] - 1)
        return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)

    def preprocess(self, file_paths):
        spectra, targets = [], []
        for path in file_paths:
            try:
                df = pd.read_excel(path, header=None, engine='openpyxl')
                if df.shape[0] < 4 or df.shape[1] < 18:
                    continue

                spec_data = (
                    df.iloc[:3, :18]
                    .replace('[^0-9.]', '', regex=True)
                    .astype(float)
                )

                quality = (
                    df.iloc[3, :3]
                    .replace('[^0-9.]', '', regex=True)
                    .astype(float)
                )
                if quality.isnull().any() or not (0 <= quality[0] <= 1):
                    continue

                quality_values = quality.values.copy()
                quality_values[0] *= 100

                for i in range(3):
                    spectra.append(spec_data.iloc[i].values)
                    targets.append(quality_values)
            except:
                continue

        X = np.array(spectra)
        y = np.array(targets)

        if len(X) == 0 or len(y) == 0:
            raise ValueError("没有有效样本，请检查数据文件。")

        print("\n数据统计信息：")
        print(f"样本数量：{X.shape[0]}")
        print(f"水分范围：{y[:, 0].min():.1f}% - {y[:, 0].max():.1f}%")
        print(f"硬度范围：{y[:, 1].min():.2f} - {y[:, 1].max():.2f}")
        print(f"糖度范围：{y[:, 2].min():.2f} - {y[:, 2].max():.2f}")
        return X, y

    def train(self, X, y):
        X_sg = self._sg_derivative(X)
        y_names = ['水分', '硬度', '糖度']

        for i in range(3):
            y_single = y[:, i]
            scaler_x = StandardScaler()
            X_scaled = scaler_x.fit_transform(X_sg)
            y_mean, y_std = y_single.mean(), y_single.std()

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y_single)

            y_pred = model.predict(X_scaled)
            r2 = r2_score(y_single, y_pred)
            rmse = mean_squared_error(y_single, y_pred, squared=False)

            print(f"\n[{y_names[i]}] 模型性能：R² = {r2:.3f}, RMSE = {rmse:.3f}")
            plt.figure()
            plt.scatter(y_single, y_pred, alpha=0.7)
            plt.plot([y_single.min(), y_single.max()], [y_single.min(), y_single.max()], '--', color='red')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{y_names[i]} 预测效果')
            plt.grid(True)
            plt.show()

            self.models[y_names[i]] = model
            self.scalers[y_names[i]] = scaler_x
            self.y_stats[y_names[i]] = {'mean': y_mean, 'std': y_std}

    def predict(self, X_new):
        X_sg = self._sg_derivative(X_new)
        results = []
        for name in ['水分', '硬度', '糖度']:
            X_scaled = self.scalers[name].transform(X_sg)
            y_pred = self.models[name].predict(X_scaled)
            results.append(y_pred.reshape(-1, 1))
        return np.hstack(results)

    def save_model(self, path='models/citrus_model.pkl'):
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'y_stats': self.y_stats
        }, path)

    @classmethod
    def load_model(cls, path='models/citrus_model.pkl'):
        data = joblib.load(path)
        model = cls()
        model.models = data['models']
        model.scalers = data['scalers']
        model.y_stats = data['y_stats']
        return model
