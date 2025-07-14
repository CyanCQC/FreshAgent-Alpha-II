import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


def get_upload_policy(api_key, model_name):
    """获取文件上传凭证"""
    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "action": "getPolicy",
        "model": model_name
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to get upload policy: {response.text}")

    return response.json()['data']


def upload_file_to_oss(policy_data, file_path):
    """将文件上传到临时存储OSS"""
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"

    with open(file_path, 'rb') as file_handle:
        files = {
            'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
            'Signature': (None, policy_data['signature']),
            'policy': (None, policy_data['policy']),
            'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
            'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
            'key': (None, key),
            'success_action_status': (None, '200'),
            'file': (file_name, file_handle)
        }

        response = requests.post(policy_data['upload_host'], files=files, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to upload file: {response.text}")

    return f"oss://{key}"


def upload_file_and_get_url(api_key, model_name, file_path):
    """上传文件并获取公网URL"""
    # 1. 获取上传凭证
    policy_data = get_upload_policy(api_key, model_name)
    oss_url = upload_file_to_oss(policy_data, file_path)
    logging.info(oss_url)
    return oss_url


# 使用示例
def get_url(file_path, api_key=None, model_name="qwen-vl-plus"):
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")

    public_url = upload_file_and_get_url(api_key, model_name, file_path)
    expire_time = datetime.now() + timedelta(hours=48)
    print(f"文件上传成功，有效期为48小时，过期时间: {expire_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"公网URL: {public_url}")

    return public_url

if __name__ == '__main__':
    get_url("C:/Users/Cyan/PycharmProjects/FruitExamine/Agent/images/8da80e34-adde-4a14-a506-5877ab17182d.jpg")
