import pymysql
from pymysql.constants import CLIENT
from typing import List, Dict, Union, Optional
import logging


class DBHandler:
    def __init__(self, db_config: dict):
        """
        初始化数据库连接
        :param db_config: 配置字典，格式示例：
            {
                "host": "localhost",
                "user": "root",
                "password": "123456",
                "database": "fruits_db",
                "port": 3306,
                "charset": "utf8mb4",
                "autocommit": False  # 是否自动提交事务
            }
        """
        self.db_config = db_config
        self.connection: Optional[pymysql.Connection] = None
        self._connect()

        # 日志
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s: %(message)s',
            level=logging.ERROR
        )

    def _connect(self) -> None:
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(
                host=self.db_config["host"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"],
                port=self.db_config.get("port", 3306),
                charset=self.db_config.get("charset", "utf8mb4"),
                cursorclass=pymysql.cursors.DictCursor,  # 返回字典格式结果
                client_flag = CLIENT.MULTI_STATEMENTS
            )
            # 设置自动提交模式
            self.connection.autocommit(self.db_config.get("autocommit", False))
        except pymysql.Error as e:
            logging.error(f"数据库连接失败: {str(e)}")
            raise RuntimeError(f"数据库连接错误: {str(e)}")

    def _ensure_connected(self) -> None:
        """确保连接有效，若断开则自动重连"""
        try:
            self.connection.ping(reconnect=True)
        except pymysql.Error:
            self._connect()

    def execute(
            self,
            sql: str,
            params: Optional[Union[List, Dict]] = None,
            fetch_all: bool = True
    ) -> Union[List[Dict], int, None]:
        """
        执行SQL语句
        :param sql: SQL语句，支持参数化查询（使用%s占位符）
        :param params: 参数列表或字典
        :param fetch_all: 是否获取所有结果（SELECT时生效）
        :return:
            - SELECT查询返回结果列表
            - INSERT/UPDATE/DELETE返回受影响行数
            - 其他操作返回None
        """
        self._ensure_connected()
        cursor = None
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)

                # 判断操作类型：通过第一个单词识别查询
                sql_command = sql.strip().upper().split()[0] if sql.strip() else ""
                if sql_command in {"SELECT", "DESC", "DESCRIBE", "SHOW"}:
                    return cursor.fetchall() if fetch_all else cursor.fetchone()
                else:
                    return cursor.rowcount

        except pymysql.Error as e:
            self.connection.rollback()
            logging.error(f"SQL执行失败: {sql}\n错误信息: {str(e)}")
            raise RuntimeError(f"数据库操作错误: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    # ------------------------- 事务控制 -------------------------
    def commit(self) -> None:
        """提交事务"""
        try:
            self.connection.commit()
        except pymysql.Error as e:
            logging.error(f"事务提交失败: {str(e)}")
            raise

    def rollback(self) -> None:
        """回滚事务"""
        try:
            self.connection.rollback()
        except pymysql.Error as e:
            logging.error(f"事务回滚失败: {str(e)}")
            raise

    # ------------------------ 连接管理 ------------------------
    def close(self) -> None:
        """手动关闭连接"""
        if self.connection:
            self.connection.close()
            self.connection = None

    def get_table_names(self) -> List[str]:
        """获取当前数据库所有表名"""
        result = self.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE'"
        )
        return [row["TABLE_NAME"] for row in result] if isinstance(result, list) else []

    def __del__(self):
        """对象销毁时自动关闭连接"""
        self.close()

if __name__ == '__main__':
    pass