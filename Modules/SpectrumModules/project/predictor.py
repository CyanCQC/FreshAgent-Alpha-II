# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import StandardScaler
import warnings

class CitrusQualityModel:
    def __init__(self, wave_list=np.arange(410, 941, 25)):
        self.bands = wave_list
        self.spa_features = None
        self.pls_model = PLSRegression(n_components=5)
        self.scaler = StandardScaler()
        self.y_stats = None

    def _adaptive_wavelet_denoise(self, spectra, wavelet='sym8'):
        """改进型自适应小波去噪"""
        denoised = []
        max_level = pywt.dwt_max_level(len(spectra[0]), wavelet)
        level = min(3, max_level)
        
        for spec in spectra:
            coeffs = pywt.wavedec(spec, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-level])) / 0.6745
            threshold = sigma * np.sqrt(2*np.log(len(spec)))
            
            coeffs = [coeffs[0]] + [
                pywt.threshold(c, threshold, mode='soft') if i >= 1 else c 
                for i, c in enumerate(coeffs[1:])
            ]
            denoised_spec = pywt.waverec(coeffs, wavelet)[:len(spec)]
            denoised.append(denoised_spec)
        return np.array(denoised)

    def _robust_spa_selection(self, X, y, max_features=14):
        """鲁棒连续投影算法"""
        X_centered = X - X.mean(axis=0)
        valid_cols = list(range(X.shape[1]))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = np.corrcoef(X_centered.T, y.reshape(-1,1).T)
        init_col = np.nanargmax(np.abs(corr_matrix[0,1:]))
        selected = [init_col]
        valid_cols.remove(init_col)
        
        max_features = min(max_features, X.shape[1]-1)
        
        for _ in range(max_features-1):
            X_sub = X_centered[:, selected]
            try:
                proj = X_sub @ np.linalg.pinv(X_sub)
            except np.linalg.LinAlgError:
                reg = 1e-6 * np.eye(X_sub.shape[1])
                proj = X_sub @ np.linalg.inv(X_sub.T @ X_sub + reg) @ X_sub.T
                
            residuals = X_centered - proj @ X_centered
            norm_residuals = np.linalg.norm(residuals[:, valid_cols], axis=0)
            
            next_col = valid_cols[np.argmax(norm_residuals)]
            selected.append(next_col)
            valid_cols.remove(next_col)
            
            if len(selected) >=2 and (norm_residuals.max() < 1.05 * np.max(norm_residuals[:-1])):
                break
                
        self.spa_features = sorted(selected)
        return X[:, self.spa_features]

    def _determine_pls_components(self, X, y):
        """动态确定最优成分数"""
        rmse_scores = []
        max_components = min(10, X.shape[1])
        
        for n in range(1, max_components+1):
            model = PLSRegression(n_components=n)
            scores = cross_val_score(
                model, X, y,
                scoring='neg_root_mean_squared_error',
                cv=5
            )
            rmse_scores.append(-scores.mean())
        
        optimal_n = np.argmin(np.diff(rmse_scores)) + 1
        return min(optimal_n, 6)

    def preprocess(self, file_paths):
        """增强型数据预处理"""
        spectra, targets = [], []
        error_files = []
        
        for path in file_paths:
            try:
                # 读取Excel文件
                df = pd.read_excel(path, header=None, engine='openpyxl')
                
                # 数据维度验证
                if df.shape[0] < 4 or df.shape[1] < 18:
                    raise ValueError("行列数不满足要求")
                    
                # 解析光谱数据
                spec_data = (
                    df.iloc[:3, :18]
                    .replace('[^0-9.]', '', regex=True)
                    .astype(float)
                )
                if spec_data.isnull().any().any():
                    raise ValueError("光谱数据包含无效值")
                    
                # 解析品质指标
                quality = (
                    df.iloc[3, :3]
                    .replace('[^0-9.]', '', regex=True)
                    .astype(float)
                )
                if quality.isnull().any() or not (0 <= quality[0] <= 1):
                    raise ValueError("品质指标格式错误")
                
                # 数据转换
                quality_values = quality.values.copy()
                quality_values[0] *= 100  # 转换水分值
                
                # 存储有效数据
                spectra.append(spec_data.mean(axis=0))
                targets.append(quality_values)
                
            except Exception as e:
                error_files.append(f"{path.name}: {str(e)}")
                continue
                
        # 有效性检查
        if len(spectra) == 0:
            error_msg = "无有效数据，请检查以下文件：\n" + "\n".join(error_files)
            raise ValueError(error_msg)
            
        X = self._adaptive_wavelet_denoise(np.array(spectra))
        y = np.array(targets)
        self.y_stats = {
            'mean': y.mean(axis=0),
            'std': y.std(axis=0)
        }
        
        print(f"成功处理 {len(spectra)}/{len(file_paths)} 个文件")
        if error_files:
            print("\n以下文件被跳过：")
            print("\n".join(error_files))
            
        return X, y

    def train(self, X, y, test_size=0.2):
        """增强型训练流程"""
        # 特征选择
        X_selected = self._robust_spa_selection(X, y[:,0])
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X_selected)
        y_scaled = (y - self.y_stats['mean']) / self.y_stats['std']
        
        # 确定最优成分
        n_components = self._determine_pls_components(X_scaled, y_scaled)
        self.pls_model = PLSRegression(n_components=n_components)
        
        # 交叉验证
        cv_scores = cross_val_score(
            self.pls_model, X_scaled, y_scaled,
            scoring='neg_root_mean_squared_error',
            cv=5
        )
        print(f"\n交叉验证结果：")
        print(f"平均RMSE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # 模型训练
        self.pls_model.fit(X_scaled, y_scaled)
        
        # 测试集评估
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42)
        y_pred = self.pls_model.predict(X_test)
        
        print("\n模型性能评估：")
        print(f"决定系数 R²: {r2_score(y_test, y_pred):.3f}")
        print(f"测试集RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")

    def predict(self, X_new):
        """稳健预测方法"""
        # 特征选择
        if not self.spa_features:
            raise ValueError("请先进行特征选择训练")
            
        X_selected = X_new[:, self.spa_features]
        
        # 标准化
        X_scaled = self.scaler.transform(X_selected)
        
        # 预测
        y_pred = self.pls_model.predict(X_scaled)
        
        # 反标准化
        result = y_pred * self.y_stats['std'] + self.y_stats['mean']
        
        # 数值约束
        result[:, 0] = np.clip(result[:, 0], 0, 100)  # 水分限制在0-100%
        return result

    def save_model(self, path='citrus_model.pkl'):
        """保存完整模型"""
        joblib.dump({
            'model': self.pls_model,
            'scaler': self.scaler,
            'features': self.spa_features,
            'y_stats': self.y_stats
        }, path)

    @classmethod
    def load_model(cls, path='citrus_model.pkl'):
        """加载预训练模型"""
        data = joblib.load(path)
        model = cls()
        model.pls_model = data['model']
        model.scaler = data['scaler']
        model.spa_features = data['features']
        model.y_stats = data['y_stats']
        return model

if __name__ == "__main__":
    # 配置数据路径
    data_dir = Path(r"F:\桌面\文件\学习\竞赛\国创赛\光谱\predict\project\data")
    
    try:
        # 动态文件加载
        files = list(data_dir.glob("[0-9][0-9][0-9].xlsx")) 
        if not files:
            raise FileNotFoundError("未找到符合命名规则的三位数.xlsx文件")
            
        print(f"发现 {len(files)} 个数据文件")
        
        # 初始化模型
        model = CitrusQualityModel()
        
        # 数据预处理
        X, y = model.preprocess(files)
        
        # 数据统计
        print("\n数据统计信息：")
        print(f"样本数量：{X.shape[0]}")
        print(f"水分范围：{y[:,0].min():.1f}% - {y[:,0].max():.1f}%")
        print(f"硬度范围：{y[:,1].min():.2f} - {y[:,1].max():.2f}")
        print(f"糖度范围：{y[:,2].min():.2f} - {y[:,2].max():.2f}")
        
        # 模型训练
        model.train(X, y)
        model.save_model()
        
        # 示例预测
        test_sample = np.random.randn(len(model.bands))
        prediction = model.predict(test_sample.reshape(1, -1))
        
        print("\n示例预测结果：")
        print(f"水分含量：{prediction[0,0]:.1f}%")
        print(f"果实硬度：{prediction[0,1]:.2f}N/cm²")
        print(f"可溶性糖：{prediction[0,2]:.2f}°Brix")
        
    except Exception as e:
        print(f"\n程序运行异常：{str(e)}")
        print("排查建议：")
        print("1. 确认数据文件符合命名规范（如031.xlsx）")
        print("2. 检查Excel文件前4行格式：")
        print("   - 前3行：18列光谱数值")
        print("   - 第4行前3列：水分(0-1)、硬度、糖度")
        print("3. 确保至少包含5个有效数据文件")