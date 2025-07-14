import os
import json
import base64
import logging
import re
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def base_encode(path):
    with open(path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_data


def get_img_json(path_1, path_2):
    prompt = "直接输出图中水果的种类。不要有任何多余输出。"
    raw_url = base_encode(path_1)
    red_url = base_encode(path_2)
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[{"role": "user","content": [
                {"type": "text","text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{raw_url}"}}
                ]}]
        )
    category =completion.choices[0].message.content
    # print(category)
    nir_prompt = get_nir_prompt(category)
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": nir_prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{red_url}"}}
        ]}]
    )

    json_part = re.search(r'\{[\s\S]*\}', completion.choices[0].message.content).group()
    fixed_json = json_part.replace('\n', '')  # 地址字段保留换行语义
    data = json.loads(fixed_json)

    logging.info(data)
    return data


def get_nir_prompt(category):
    prompt_nir = f"""
    你是一名专业农产质量检测员。请根据该{category}的远红外光谱图像分析以下特征，并按照严格规范的JSON格式输出结果：
    
    要求：
    1. 外观特征分析需包含：
       - 表皮颜色分布（主色调及异常色斑）
       - 表面纹理描述（光滑度/粗糙度/褶皱）
       - 几何形状（球形/椭球/不规则形变）
    
    2. 整体形态评估：
       - 三维对称性（轴向对称度百分比）
       - 轮廓完整度（是否存在凹陷/凸起/畸形）
    
    3. 成熟度判断标准：
       - 糖度分布梯度（低/中/高）
       - 细胞活性指数（0-100%）
       - 成熟阶段（生长期/转色期/完熟期/过熟期）
    
    4. 病虫害检测：
       - 感染迹象（是/否）
       - 异常区域定位（顶部/中部/底部）
       - 病灶类型预测（真菌/虫蛀/机械损伤）
    
    5. 品质均匀性评估：
       - 糖度一致性（标准差系数）
       - 密度均匀度（核心与表皮差异比）
    
    6. 质量预测模型：
       - 综合评级（A+/A/B/C）
       - 货架期预测（单位：天）
       - 商业价值评估（高端/常规/次级）
    
    输出要求：
    - 数值型数据保留两位小数
    - 枚举值使用中文表述
    - 缺失数据字段保持null值
    - 使用规范的JSON结构（禁用注释）
    
    示例：
    {{
      "水果品种": "砂糖桔"
      "外观特征": {{
        "表皮颜色": {{"主色调": "深红色", "异常色斑": "底部浅色区域"}},
        "表面纹理": "微粗糙带蜡质层",
        "几何形状": "椭球形（长径比1:1.2）"
      }},
      "整体形态": {{
        "对称性": 82.35,
        "轮廓完整度": "顶端轻微凹陷"
      }},
      "成熟度": {{
        "糖度梯度": "高",
        "细胞活性": 76.40,
        "成熟阶段": "完熟期"
      }},
      "病虫害": {{
        "感染迹象": true,
        "异常区域": "中部",
        "病灶类型": "真菌感染"
      }},
      "品质均匀性": {{
        "糖度一致性": 0.15,
        "密度均匀度": 0.08
      }},
      "质量预测": {{
        "综合评级": "A",
        "货架期": 6.5,
        "商业价值": "高端"
      }}
    }}
    """

    return prompt_nir

if __name__ == '__main__':

    get_img_json("C:/Users/Cyan/Desktop/010-02-0.jpg", "C:/Users/Cyan/Desktop/010-02-1.jpg")

