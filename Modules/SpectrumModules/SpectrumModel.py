import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import joblib


class SpectralPredictor:
    def __init__(self, model_path='C:/Users/Cyan/PycharmProjects/FruitExamine/Agent/spectrum/model/citrus_model.pkl'):
        """Initialize the predictor by loading the trained model."""
        data = joblib.load(model_path)
        self.models = data['models']
        self.scalers = data['scalers']
        self.y_stats = data['y_stats']
        self.bands = np.arange(410, 941, 25)  # Spectral bands, consistent with training

    def _sg_derivative(self, X, window_length=5, polyorder=2, deriv=1):
        """Apply Savitzky-Golay derivative to the spectral data."""
        window_length = min(window_length, X.shape[1] - 1)
        return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)

    def preprocess(self, file_path):
        """Read and preprocess the new spectral data from an Excel file."""
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
        if df.shape[0] < 1 or df.shape[1] < 18:
            raise ValueError("数据文件格式不正确，请确保至少有1行和18列数据")

        # Extract spectral data (assuming first 18 columns)
        spec_data = df.iloc[0, :18].values.reshape(1, -1)
        return spec_data

    def predict(self, file_path):
        """Predict moisture, hardness, and sugar content from the spectral data."""
        # Preprocess the new data
        X_new = self.preprocess(file_path)
        X_sg = self._sg_derivative(X_new)

        # Perform predictions
        results = {}
        for tp in [(' %','水分'), (' N/cm²', '硬度'), (' °Brix', '糖度')]:
            suffix, name = tp
            scaler = self.scalers[name]
            model = self.models[name]

            X_scaled = scaler.transform(X_sg)
            y_pred = model.predict(X_scaled)
            results[name] = f"{y_pred[0]:.2f}"+suffix

        return results


if __name__ == "__main__":
    # Example usage
    predictor = SpectralPredictor()
    test_file = Path("data/031.xlsx")  # Update this path to your test file
    predictions = predictor.predict(test_file)

    print("\n预测结果：")
    print(f"水分含量：{predictions['水分']:.1f}%")
    print(f"果实硬度：{predictions['硬度']:.2f} N/cm²")
    print(f"可溶性糖：{predictions['糖度']:.2f} °Brix")