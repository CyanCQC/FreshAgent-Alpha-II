import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import FastSAM


def visualize_multicolor(image_path, masks, output_path, alpha=0.5):
    """
    对给定的 N x H x W 布尔掩码，用随机颜色叠加到原图上，
    并保存到 output_path。alpha 控制不透明度（0.0～1.0）。
    """
    # 原图转 RGBA
    # print(f"IMAGE_PATH: {image_path}")
    img = Image.open(image_path).convert("RGBA")
    base = img.copy()
    _, H, W = masks.shape

    for mask in masks:
        # 随机颜色（RGB）
        r, g, b = [random.randint(0, 255) for _ in range(3)]
        # 生成图层，初始 alpha 设置为 0
        layer = Image.new("RGBA", (W, H), (r, g, b, 0))

        # 将布尔掩码转成 [0, int(255*alpha)] 的灰度图
        mask_alpha = (mask.astype(np.uint8) * int(255 * alpha))
        alpha_img = Image.fromarray(mask_alpha, mode="L")

        # 把调整好的 alpha 作为图层的不透明度
        layer.putalpha(alpha_img)

        # 叠加到 base 上
        base = Image.alpha_composite(base, layer)

    # 保存（转换回 RGB）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base.convert("RGB").save(output_path)


def image_process_to_ratio(image_path, output_path, device='cpu', def_m="DefectSAM.pt", fru_m="FruitSAM.pt",
                           model_directory="./model", filename=""):
    # os.makedirs(output_path, exist_ok=True)

    # print("→ Opening:", p)
    # print("   Exists?", p.exists(), " Size:", p.stat().st_size if p.exists() else "N/A")
    img = Image.open(image_path).convert("RGBA")

    def_p, fru_p = os.path.join(model_directory, def_m), os.path.join(model_directory, fru_m)
    print(def_p, fru_p)

    model_defects = FastSAM(def_p)
    model_general = FastSAM(fru_p)

    # 获取缺陷区域
    results_defects = model_defects.predict(
        source=image_path,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9
    )
    # 获取水果面积
    results_general = model_general.predict(
        source=image_path,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9
    )
    res_d = results_defects[0]
    res_g = results_general[0]

    result_dict = {}

    # 获取掩码
    masks_defects = res_d.masks.data.cpu().numpy() > 0.5
    masks_general = res_g.masks.data.cpu().numpy() > 0.5

    # 可视化并保存
    # visualize_multicolor(image_path, masks_defects, os.path.join(output_path, "defects.jpg"))
    # visualize_multicolor(image_path, masks_general, os.path.join(output_path, "general.jpg"))

    print(os.path.join(output_path, f"defects_{filename}.jpg"))

    visualize_multicolor(image_path, masks_defects, output_path+f"defects.jpg")
    visualize_multicolor(image_path, masks_general, output_path+f"general.jpg")

    result_dict["save_path"] = output_path

    print(f"可视化已存储，路径 {output_path}")

    # 计算像素面积
    pixel_areas_d = [mask.sum() for mask in masks_defects]
    pixel_areas_g = [mask.sum() for mask in masks_general]

    result_dict["def_pix_area"] = pixel_areas_d
    result_dict["fru_pix_area"] = pixel_areas_g

    print("缺陷检测")
    for i, pa in enumerate(pixel_areas_d):
        print(f"实例 {i} 的像素面积: {pa}")

    print("水果检测")
    for i, pa in enumerate(pixel_areas_g):
        print(f"实例 {i} 的像素面积: {pa}")

    ratio = sum(pixel_areas_d) / sum(pixel_areas_g)
    result_dict["def_ratio"] = ratio
    print(f"缺陷面积占比 {ratio * 100 :.2f} %")

    return result_dict


if __name__ == "__main__":

    image_path = "005-01-1.jpg"
    device = "cpu"
    output_path = "./output"
    image_process_to_ratio(image_path, output_path, device='cpu', def_m="DefectSAM.pt", fru_m="FruitSAM.pt",
                           model_directory="./model")

    # 模型定义与参数设置
    model_defects = FastSAM("./model/DefectSAM.pt")
    model_general = FastSAM("./model/FruitSAM.pt")

    IMAGE_PATH = "90.jpg"
    DEVICE     = "cpu"

    # 获取缺陷区域
    results_defects = model_defects.predict(
        source=IMAGE_PATH,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9
    )
    # 获取水果面积
    results_general = model_general.predict(
        source=IMAGE_PATH,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        texts=["an orange"],
        iou=0.9
    )
    res_d = results_defects[0]
    res_g = results_general[0]

    # 获取掩码
    masks_defects = res_d.masks.data.cpu().numpy() > 0.5
    masks_general = res_g.masks.data.cpu().numpy() > 0.5

    # 4. 多彩可视化并保存
    OUTPUT_PATH = "./output/pipeline/"
    visualize_multicolor(IMAGE_PATH, masks_defects, OUTPUT_PATH+"defects.jpg")
    visualize_multicolor(IMAGE_PATH, masks_general, OUTPUT_PATH+"general.jpg")

    print(f"可视化已存储，路径 {OUTPUT_PATH}")

    # 计算像素面积
    pixel_areas_d = [mask.sum() for mask in masks_defects]
    pixel_areas_g = [mask.sum() for mask in masks_general]
    
    print("缺陷检测")
    for i, pa in enumerate(pixel_areas_d):
        print(f"实例 {i} 的像素面积: {pa}")

    print("水果检测")
    for i, pa in enumerate(pixel_areas_g):
        print(f"实例 {i} 的像素面积: {pa}")

    ratio = sum(pixel_areas_d) / sum(pixel_areas_g)
    print(f"缺陷面积占比 {ratio * 100 :.2f} %")
    
    
