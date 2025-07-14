# -*- coding: utf-8 -*-

from pathlib import Path
from core import CitrusQualityModel

data_dir = Path("data")
files = list(data_dir.glob("[0-9][0-9][0-9].xlsx"))
if not files:
    raise FileNotFoundError("未找到符合命名规则的三位数.xlsx文件")

model = CitrusQualityModel()
X, y = model.preprocess(files)
model.train(X, y)
model.save_model("models/citrus_model.pkl")
