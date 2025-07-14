import json
import os

from Agent.spectrum.SpectrumProcess import get_spectrum_dict
from .analysis import create_image_task, query_and_get_result, get_img_jsonl
import re
from zhipuai import ZhipuAI
import time
from openai import OpenAI

client = ZhipuAI(api_key="04ac6f4a2fa34264acc3b0c1ac691d97.sCXt5ScbILJOyLmf")  # 请填写您自己的APIKey

client_Q = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-8f0775132fdc4a5db3bbfeb335ac8452",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def execute_agent_task(dir_path, output_csv, category):
    """
    自动化质检任务执行流程：
    1. 创建图像分析任务
    2. 周期性查询结果
    3. 返回最终检测报告
    """
    # try:
    # 初始化检测任务
    task_id = create_image_task(dir_path, output_csv, category)
    if not task_id:
        raise ValueError("任务创建失败")

    # 轮询结果（最多等待5分钟）
    max_retries = 30
    for _ in range(max_retries):
        result = query_and_get_result(task_id)

        if result and result.get("status") == "completed":
            print("end")
            return result

        time.sleep(10)

    raise TimeoutError("任务处理超时")

    # except Exception as e:
    #     return {"error": str(e), "task_id": task_id}


def merge_json(json1: dict, json2: dict) -> dict:
    """高效合并两个无重复键的JSON对象

    :param json1: 第一个字典对象
    :param json2: 第二个字典对象
    :return: 合并后的新字典
    """
    print(json1, json2)
    return {**json1, **json2}

def get_report_prompt(user_role):
    report_prompt = f"""
    请根据提供的JSON水果数据，生成一份适合普通{user_role}消费者阅读的购买指南。具体要求：
    1. **结构清晰**：包含"省钱推荐"、"营养冠军"、"应季优选"三个板块
    2. **口语化表达**：使用"每天一个苹果"这类生活化建议，避免"膳食纤维"等专业术语
    3. **价格可视化**：用"¥15/斤≈3杯奶茶钱"的类比帮助理解价格
    4. **突出亮点**：用⭐符号标记性价比超过90%的产品
    5. **风险提示**：标注"谨慎购买"项（如榴莲标注核重占比）
    6. **数据关联**：当推荐苹果时，关联显示"冷藏存放时间：5-7天"等存储信息
    
    示例期待输出：
    "【省钱推荐】当前香蕉（⭐⭐⭐）正值上市季，批发市场单价较超市低40%，适合家庭整箱采购..."
    """
    return report_prompt


def construct_structured_data(dir_path, category, fastsam_output, fastsam_model, reasoner=False):
    jsonl_img = get_img_jsonl(dir_path, category, fastsam_output, fastsam_model)
    jsonl_nir = get_spectrum_dict(dir_path)

    json_merge = []
    for tp in jsonl_img:
        if tp[1] is None and tp[0] is not None:
            json_merge.append(jsonl_nir[tp[0]])
        elif jsonl_nir.get(tp[0]) is None and tp[1] is not None:
            json_merge.append(tp[1])
        else:
            json_merge.append(merge_json(tp[1], jsonl_nir[tp[0]]))
        # json_merge = [merge_json(tp[1], jsonl_nir[tp[0]]) for tp in jsonl_img]

    prompt = """
    处理水果质检数据，规则如下：

    1. **输入**：JSON列表。若为空，直接告诉用户无法处理空json。
    
    2. **质量等级**：
    - 优品
    - 良品
    - 合格品
    - 不合格品
    - 异常数据标记`数据异常`
    
    3. **输出**：
    ```markdown
    | 序号 | 质量等级 | 水分 (%) | 糖度 (Brix) | 硬度 (N/cm²) | 水果描述 |
|------|----------|----------|-------------|--------------|----------|
| 1    | 优等品   | 85.0     | 14.5        | 5.2          | 该果表面完整，糖分较高，无机械损伤，具有较高商业价值 |
| 2    | 合格品   | 82.3     | 12.8        | 4.8          | 表面有轻微擦伤，糖分适中，整体品质良好 |
| 3    | 数据异常 | 78.6     | 10.2        | 3.5          | 水分偏低，糖度不足，硬度过低，可能存在品质问题 |
    请输出严格符合格式的两列表格，保持原始数据顺序。
    不要有任何解释及多余输出，直接开始回答！
    """

    if reasoner:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "根据用户提供的json列表，判断水果品质。将结果输出为检测结果的markdown表格。"},
                {"role": "user", "content": f"{json_merge}。直接给出结果表格，不要有任何多余的输出！"},
            ],
            stream=False
        )
    else:
        response = client_Q.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{json_merge}"},
            ],
            stream=False
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content



if __name__ == '__main__':
    construct_structured_data("C:/Users/Cyan/Desktop/images", "砂糖桔")