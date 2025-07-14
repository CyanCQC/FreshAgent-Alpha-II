import json
from analyze import get_img_json
# from spectrum import get_spectrum_json

import logging
from openai import OpenAI

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
client_Q = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def merge_json(json1: dict, json2: dict) -> dict:
    """高效合并两个无重复键的JSON对象

    :param json1: 第一个字典对象
    :param json2: 第二个字典对象
    :return: 合并后的新字典
    """
    return {**json1, **json2}


def get_report_prompt(user_role):
    report_prompt = f"""
    请根据用户提供的JSON水果数据，针对当前水果生成一份适合{user_role}消费者阅读的购买指南。具体要求：
    1. **结构清晰**：包含"省钱推荐"、"营养冠军"、"应季优选"三个板块
    2. **口语化表达**：使用"每天一个苹果"这类生活化建议，避免"膳食纤维"等专业术语
    3. **价格可视化**：用"¥15/斤≈3杯奶茶钱"的类比帮助理解价格
    4. **突出亮点**：用⭐符号标记性价比超过90%的产品
    5. **风险提示**：标注"谨慎购买"项（如榴莲标注核重占比）
    6. **数据关联**：当推荐苹果时，关联显示"冷藏存放时间：5-7天"等存储信息
    7. **联网搜索**：生成报告时，结合联网搜索内容，确保报告时效性
    
    示例期待输出：
    "【省钱推荐】当前香蕉（⭐⭐⭐）正值上市季，批发市场单价较超市低40%，适合家庭整箱采购..."
    """
    return report_prompt


def construct_structured_data(path_1, path_2, user_role="19岁健身男性", reasoner=False):
    json_img = get_img_json(path_1, path_2)
    # json_nir = get_spectrum_json()
    # json_merge = merge_json(json_img, json_nir)
    json_merge = json_img


    if reasoner:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": get_report_prompt(user_role)},
                {"role": "user", "content": f"{json_merge}"},
            ],
            stream=False
        )
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": get_report_prompt(user_role)},
                {"role": "user", "content": f"{json_merge}"},
            ],
            stream=False,
            # extra_body={
            #     "enable_search": True
            # }
        )
    return response.choices[0].message.content


# def construct_structured_data_C(path_1, path_2, user_role="20岁女性", reasoner=False):
#     json_img = get_img_json(path_1, path_2)
#     json_nir = get_spectrum_json()
#     json_merge = merge_json(json_img, json_nir)
#
#     if reasoner:
#         response = client.chat.completions.create(
#             model="deepseek-reasoner",
#             messages=[
#                 {"role": "system", "content": get_report_prompt(user_role)},
#                 {"role": "user", "content": f"{json_merge}"},
#             ],
#             stream=False
#         )
#     else:
#         response = client.chat.completions.create(
#             model="qwen-max",
#             messages=[
#                 {"role": "system", "content": get_report_prompt(user_role)},
#                 {"role": "user", "content": f"{json_merge}"},
#             ],
#             stream=False,
#             extra_body={
#                 "enable_search": True
#             }
#         )
#     return response.choices[0].message.content


if __name__ == '__main__':
    logging.info(
        construct_structured_data(
            "C:/Users/Cyan/Desktop/010-02-0.jpg",
            "C:/Users/Cyan/Desktop/010-02-1.jpg",
        )
    )

