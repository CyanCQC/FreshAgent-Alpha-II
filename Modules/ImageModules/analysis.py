import csv
import json
from pathlib import Path
from openai import OpenAI
import os
import base64
from .ImageProcess.pipeline import image_process_to_ratio

client = OpenAI(
    # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
    api_key="sk-7312141a7efd48f3acbd3957aecf8390",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 百炼服务的base_url
)


# 企业场景，需要人工给定水果种类
def jsonl_generate(csv_name, category):
    prompt = f"""
    作为农业质检系统，分析{category}的远红外光谱图像，输出精简质检报告：

    关键指标：
    1. 外观特征（表皮色斑/纹理/形状异常）
    2. 成熟度（糖度梯度/细胞活性/成熟阶段）
    3. 病虫害（是否感染/病灶类型）
    4. 品质预测（综合评级/货架期/商品等级）

    输出要求：
    - 仅保留核心字段
    - 数值保留两位小数
    - 异常状态标注中文类型
    - 无异常字段输出"正常"

    示例：
    {{
      \"产品类型\": \"砂糖桔\",
      \"外观检测\": {{\"色斑\": \"底部浅斑\", \"纹理\": \"粗糙\", \"形状\": \"正常\"}},
      \"成熟度\": {{\"糖度\": \"高\", \"活性指数\": 76.40, \"阶段\": \"完熟期\"}},
      \"病虫害\": {{\"感染\": true, \"类型\": \"真菌\"}},
      \"品质预测\": {{\"评级\": \"A\", \"保质期\": 6.5, \"等级\": \"高端\"}}
    }}
    """

    def messages_builder_example(content):
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{content}"}}
            ]}
        ]

    with open(f"{csv_name}.csv", "r") as fin:
        with open(f"{csv_name}.jsonl", 'w', encoding='utf-8') as fout:
            csvreader = csv.reader(fin)
            for row in csvreader:
                body = {"model": "qwen-vl-max", "messages": messages_builder_example(row[1])}
                request = {"custom_id": row[0], "method": "POST", "url": "/v1/chat/completions", "body": body}
                fout.write(json.dumps(request, separators=(',', ':'), ensure_ascii=False) + "\n", )


import re
def extract_json_list(filename):
    """从结果文件中提取结构化JSON列表"""
    result = []

    with open(f"{filename}.jsonl", 'r') as f:
        for line in f:
            # 解析基础数据
            print(line)
            entry = json.loads(line)
            content_str = entry.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get(
                "content", "")

            # 提取并修复JSON
            json_str = re.search(r'\{.*\}', content_str, re.DOTALL)
            clean_json = json_str.replace('\n', '')
            data = json.loads(clean_json)

            result.append(data)


    return result


def result_generate(filename):
    columns = ["custom_id",
               "model",
               "request_id",
               "status_code",
               "error_code",
               "error_message",
               "created",
               "content",
               "usage"]

    def dict_get_string(dict_obj, path):
        obj = dict_obj
        try:
            for element in path:
                obj = obj[element]
            return obj
        except:
            return None

    with open(f"{filename}.jsonl", "r") as fin:
        with open(f"{filename}.csv", 'w', encoding='utf-8') as fout:
            rows = [columns]
            for line in fin:
                request_result = json.loads(line)
                row = [dict_get_string(request_result, ["custom_id"]),
                       dict_get_string(request_result, ["response", "body", "model"]),
                       dict_get_string(request_result, ["response", "request_id"]),
                       dict_get_string(request_result, ["response", "status_code"]),
                       dict_get_string(request_result, ["error", "error_code"]),
                       dict_get_string(request_result, ["error", "error_message"]),
                       dict_get_string(request_result, ["response", "body", "created"]),
                       dict_get_string(request_result, ["response", "body", "choices", 0, "message", "content"]),
                       dict_get_string(request_result, ["response", "body", "usage"])]
                rows.append(row)
            writer = csv.writer(fout)
            writer.writerows(rows)

    return extract_json_list(filename)


def task_create(file_path):
    file_object = client.files.create(file=Path(file_path), purpose="batch")
    file_id = json.loads(file_object.model_dump_json())["id"]

    batch = client.batches.create(
        input_file_id=file_id,  # 上传文件返回的 id
        endpoint="/v1/chat/completions",  # 大语言模型固定填写，/v1/chat/completions，embedding文本向量模型填写"/v1/embeddings"
        completion_window="24h"
    )
    print(batch)

    return batch.id


def task_query(file_id):
    batch = client.batches.retrieve(file_id)  # 将batch_id替换为Batch任务的id
    return batch


def task_cancel(file_id):
    batch = client.batches.retrieve(file_id)  # 将batch_id替换为Batch任务的id
    print(batch)


def task_result(file_id):
    content = client.files.content(file_id=file_id)
    print(content.text)
    content.write_to_file(f"{file_id}.jsonl")


def get_file_id(dir_path, output_csv):
    raw_files = os.listdir(dir_path)
    results = []
    for filename in raw_files:
        path = os.path.join(dir_path, filename)
        with open(path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        results.append((filename, base64_data))

    with open(f"{output_csv}.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        # writer.writerow(['id', 'file_id'])
        # 批量写入数据
        writer.writerows(results)

    return results


def create_image_task(dir_path, output_csv, category):
    get_file_id(dir_path, output_csv)
    jsonl_generate(output_csv, category)
    file_id = task_create(output_csv+".csv")
    return file_id


def query_and_get_result(file_id):
    batch = task_query(file_id)
    if batch.status == "completed":
        print(f"Completed! File was saved to {file_id}.csv")
        return result_generate(f"{file_id}")
    elif batch.status == "failed":
        print("Task Failed. Cancelling...")
        task_cancel(file_id)
    elif batch.status == "in_progress":
        print("Processing...")


def base_encode(path):
    with open(path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_data

def get_img_jsonl(dir_path, category, fastsam_output, fastsam_model):
    img_path = os.path.join(dir_path, "img")
    filenames = os.listdir(img_path)
    data_list = []
    for filename in filenames:
        print(f"### 正在分析 {filename}...")
        path = os.path.join(img_path, filename)

        base, _ = os.path.splitext(filename)

        ### FastSAM先处理
        sam_result = image_process_to_ratio(
            path,
            fastsam_output+base+'/',
            model_directory=fastsam_model,
            filename=filename.split(".")[0]
        )

        ### 大语言模型再处理
        red_url = base_encode(path)
        def_url = base_encode(os.path.join(sam_result["save_path"], "defects.jpg"))
        # print(category)
        nir_prompt = f"""
        作为农业质检系统，分析{category}的原图图像与缺陷分割结果图像（缺陷占比{sam_result["def_ratio"]}），输出JSON格式的精简质检报告：
    
        关键指标：
        1. 产品类型（准确的水果种类）
        2. 外观特征（表皮色斑/纹理/形状异常）
        3. 病虫害（是否感染/病灶类型）
    
        输出要求：
        - 仅保留核心字段
        - 数值保留两位小数
        - 异常状态标注中文类型
        - 无异常字段输出"正常"
    
        示例：
        {{
          \"产品类型\": \"砂糖桔\",
          \"外观检测\": {{\"色斑\": \"底部浅斑\", \"纹理\": \"粗糙\", \"形状\": \"正常\", \"概述\": \"砂糖橘表面整体呈橙黄色，底部有轻微浅斑，纹理略显粗糙，形状标准，果型均匀。\"}},
          \"病虫害\": {{\"感染\": true, \"类型\": \"真菌\"}},
        }}
        """
        completion = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": nir_prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{red_url}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{def_url}"}}
            ]}]
        )

        json_part = re.search(r'\{[\s\S]*\}', completion.choices[0].message.content).group()
        # print(f"JSON PART: {json_part}")
        fixed_json = json_part.replace('\n', '')  # 地址字段保留换行语义
        # print(f"FIXED-JSON: {fixed_json}")

        print(f"< {filename} > 外观分析结果：\n {fixed_json}")
        data = json.loads(fixed_json)

        data["外观检测"]["缺陷面积比例"] = sam_result["def_ratio"]

        data_list.append((filename, data))

    print(data_list)
    return data_list