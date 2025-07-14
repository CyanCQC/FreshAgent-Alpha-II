# -*- coding: utf-8 -*-
import numpy as np
from core import CitrusQualityModel

model = CitrusQualityModel.load_model("models/citrus_model.pkl")
test_sample = np.random.rand(1, 18)
pred = model.predict(test_sample)

print("预测结果：")
print(f"水分：{pred[0, 0]:.1f}%")
print(f"硬度：{pred[0, 1]:.2f}")
print(f"糖度：{pred[0, 2]:.2f}")
