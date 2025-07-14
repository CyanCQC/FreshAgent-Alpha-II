import os
import logging
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymysql
from pdfminer.high_level import extract_text
from pymysql.constants import CLIENT
import dashscope
from FreshAgentAlpha.Modules.Handlers.DBH import DBHandler

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

# 通义千问API配置
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_ENDPOINT = "https://api.tongyi.com/embeddings"


class LocalDataHandler:
    def __init__(self, db_config):
        self.DBHandler = DBHandler(db_config)

    def _get_existed_files(self):
        sql = """
            SELECT file_name FROM embeddings;
        """
        return self.DBHandler.execute(sql, fetch_all=True)

    def _check_dir(self, dir_path):
        """
        遍历目录，将尚未入库的 PDF 文件上传。
        注意：同名文件更新情况未作处理
        """
        filenames = os.listdir(dir_path)
        logging.debug(filenames)
        # print(self._get_existed_files())
        # 注意：此处根据你数据库返回字典的键名进行调整，此处示例中使用的是 'TABLE_NAME'
        existed = [f['file_name'] for f in self._get_existed_files()]
        # print("已存在文件:", existed)
        for filename in filenames:
            if filename.endswith('.pdf') and filename not in existed:
                self._upload_file(os.path.join(dir_path, filename))
            else:
                logging.info(f"{filename} 已检测到")

    def _upload_file(self, file_path):
        """
        对文件进行解析、文本划分、生成嵌入后上传到数据库
        """
        chunks, embeddings = self._parse_and_embed_file(file_path)
        file_name = os.path.basename(file_path)
        self._save_to_db(file_name, chunks, embeddings)

    def _parse_and_embed_file(self, file_path):
        """
        解析文件内容、利用文本划分器进行分块并生成每个块的嵌入。
        """
        text = self._parse_file(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        embeddings = []
        for chunk in chunks:
            embedding = self._get_embedding(chunk)
            embeddings.append(embedding)
        return chunks, np.array(embeddings)

    def _parse_file(self, file_path):
        """
        根据文件类型解析文件内容，目前支持 .txt 和 .pdf 文件
        """
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.lower().endswith('.pdf'):
            return extract_text(file_path)
        else:
            raise ValueError("不支持的文件类型，仅支持.txt和.pdf")

    def _save_to_db(self, file_name, chunks, embeddings):
        """
        将文件的文本块及其嵌入保存到数据库。
        注意：这里将 numpy 数组直接通过 tobytes() 转为二进制存储
        """
        if self.DBHandler:
            try:
                for chunk, embedding in zip(chunks, embeddings):
                    embedding_binary = embedding.tobytes()
                    row_count = self.DBHandler.execute(
                        "INSERT INTO embeddings (file_name, chunk, embedding) VALUES (%s, %s, %s)",
                        [file_name, chunk, embedding_binary]
                    )
                    if row_count != 1:
                        raise RuntimeError(f"插入失败，影响行数为 {row_count}")
                    logging.info(f"文件 {file_name} 的嵌入已保存到数据库。")
                self.DBHandler.commit()
            except RuntimeError as e:
                logging.error(f"保存文件 {file_name} 时出错: {e}")
                raise

    def _get_embedding(self, text):
        """
        使用 dashscope API 生成文本嵌入，返回 numpy 数组
        """
        try:
            response = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v1,
                input=text
            )
            if response.status_code == 200:
                return np.array(response.output['embeddings'][0]['embedding'], dtype=np.float32)
            else:
                raise Exception(f"API 调用失败: {response.status_code}, {response.message}")
        except Exception as e:
            logging.error(f"错误: {e}")
            raise

    def cosine_similarity(self, vec1, vec2):
        """
        计算两个向量之间的余弦相似度
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2 + 1e-10)

    def search_file_by_keyword(self, keyword: str, top_k=3, similarity_threshold=0.5):
        """
        根据关键词进行增强检索：
          1. 计算查询关键词的嵌入
          2. 从数据库中查询所有文本块及其嵌入
          3. 计算查询嵌入与每个文本块嵌入的余弦相似度
          4. 返回相似度超过阈值的 top_k 结果，结果中包含文件名、相似度和文本片段
        """
        # 1. 获取查询关键词的嵌入
        query_embedding = self._get_embedding(keyword)

        # 2. 查询数据库中所有文本块及其嵌入
        sql = "SELECT file_name, chunk, embedding FROM embeddings;"
        results = self.DBHandler.execute(sql, fetch_all=True)
        if not results:
            return "无检索结果。"

        similarities = []
        for row in results:
            file_name = row["file_name"]
            chunk = row["chunk"]
            embedding_blob = row["embedding"]
            # 将二进制数据转换为 numpy 数组，此处假设使用 np.float32 类型
            candidate_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            sim = self.cosine_similarity(query_embedding, candidate_embedding)
            similarities.append({
                "file_name": file_name,
                "chunk": chunk,
                "similarity": sim
            })

        # 过滤出相似度高于阈值的结果
        filtered = [item for item in similarities if item["similarity"] >= similarity_threshold]
        # 按相似度降序排序
        filtered.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = filtered[:top_k]

        if not top_results:
            return "未匹配到相关内容。"

        ret_str = ""
        for res in top_results:
            ret_str += f"文件: {res['file_name']}, 相似度: {res['similarity']:.2f}\n文本片段: {res['chunk']}\n\n"
        return ret_str


if __name__ == '__main__':
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "LocalKnowledge"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "charset": os.getenv("DB_CHARSET", "utf8mb4"),
        "autocommit": False,
    }

    LocalHandler = LocalDataHandler(db_config)
    LocalHandler.check_dir('./')

