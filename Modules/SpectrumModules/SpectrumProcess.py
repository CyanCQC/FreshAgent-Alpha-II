import os
from Agent.spectrum.SpectrumModel import SpectralPredictor

import numpy as np
import logging

model = SpectralPredictor()
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def get_spectrum_dict(directory: str):
    """Return spectrum prediction dictionary for a directory."""
    img_path = os.path.join(directory, "img")
    spec_path = os.path.join(directory, "spec")
    img_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 初始化结果字典
    results = {}

    # 遍历每个图像文件
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        file_name = os.path.basename(img_file)
        xlsx_file = os.path.join(spec_path, base_name + '.xlsx')

        # 检查 .xlsx 文件是否存在
        if os.path.exists(xlsx_file):
            try:
                predictions = model.predict(xlsx_file)
                results[file_name] = predictions
                logging.info(f"{file_name} spectrum result: {predictions}")
            except Exception as exc:
                logging.error(f"Failed to process {xlsx_file}: {exc}")
                results[file_name] = None
        else:
            logging.warning(f"Missing spectrum file for {img_file}")
            results[file_name] = None

    return results
