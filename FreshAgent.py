import markdown
from apscheduler.events import EVENT_JOB_REMOVED
from apscheduler.schedulers.blocking import BlockingScheduler
from zhipuai import ZhipuAI
from datetime import datetime, timedelta

from typing import List, Dict, Union, Optional
import os

from FreshAgentAlpha.Modules.Handlers.DBH import DBHandler
from FreshAgentAlpha.Modules.Handlers.EMH import EmailHandler
from FreshAgentAlpha.Modules.Handlers.LDH import LocalDataHandler
from Modules.ImageModules.url_generate import get_url
from Modules.ImageModules.report import construct_structured_data
import json
from openai import OpenAI
from PyQt5.QtCore import QObject, pyqtSignal

client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

client_Qwen = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

client_KwooLa = OpenAI(
    api_key=os.getenv("KWOOLA_API_KEY"),
    base_url="https://api.tgkwai.com/api/v1/qamodel/",
)


class FruitAgent(QObject):
    output_signal = pyqtSignal(str)

    def __init__(self, location, db_config, email_config):
        super().__init__()
        self.location = location
        self.dbHandler = DBHandler(db_config)
        self.emailHandler = EmailHandler(email_config)
        self.localDataHandler = LocalDataHandler(db_config)
        self.history = []
        self.memory = []
        self.scheduler = BlockingScheduler()
        self.enhanced_retrieval = True
        self.user_target = "暂无"

        self.FASTSAM_OUTPUT = "./output/"
        self.FASTSAM_MODEL = "./ImageModules/ImageProcess/model"

        self.debug = True

    def analyze(self, pending_str):
        tpl_prompt = f"""
你是一个智能 Agent，收到用户的最新需求后，请按以下格式输出 JSON，不要多写任何多余内容，也不要写思考过程：

可用工具表：  
| 工具名           | 应用场景         | 参数规则                                                      |
| ---------------- | ---------------- | ------------------------------------------------------------- |
| analyze          | 果蔬分析          | {{"prompt":"<对于分析需求的概括>"}}     |
| query_db         | 数据库操作        | {{"sql":"<需求的SQL语句（可有多句）>"}} |
| search           | 联网搜索          | {{"query":"<检索内容>"}}                   |
| generate         | 直接生成          | <无参数>                                   |
| send_message     | 信息发送          | {{"to":"<邮箱或手机号>","subject":"<邮件主题>","content":"<消息内容>"}} |
| enhanced_search  | 增强检索          | {{"query":"<检索内容>"}}             |
| further_analyze  | 深度分析          | {{"query":"需要深度分析的内容"}}                               |

【工具类型】
1. 果蔬分析（需图像识别或质量判断）
2. 数据库操作（需查询/修改数据库）
3. 联网搜索（需实时网络信息）
4. 直接生成（无需上述三类操作）
5. 信息发送（向用户发送邮件/信息）
6. 增强检索（需要在本地知识库中检索时）
7. 深度分析（需要对部分数据进行深度分析）

【判断规则】
① 含图像/光谱分析必选1
② 需数据库交互必选2。
----凡是涉及到总结等应用数据库数据分析的任务，必须先查询、再分析！
----涉及到保存数据到表中的操作，必须先使用DESC查询对应表的字段，再进行插入！不允许直接插入！
③ 需最新网络信息必选3
④ 仅文本生成时选4
⑤ 若用户要求发送邮件或信息选5
————选6时，元任务表述必须包含邮箱地址
⑥ 有潜在的本地知识库检索需求时，选6
⑦ 有对果蔬分析结果或用户提供的数据进行深度分析时，选7

【用户目的】
{self.user_target}

{{
  "response": "<LLM 要回复给用户的文本>",
  "call": {{
    "name": "<要调用的功能标识符，例如 search、query_db、analyze_image 等>",
    "arguments": {{ /* 调用该功能所需的 JSON 参数 */ }}
  }},
  "end": <true 或 false>  /* true 表示这是整个 workflow 的最后一步且整个 workflow 已实现用户目的 */
}}

请注意：
- 仅在最后一行输出上述 JSON，其他任何解释、思考都不要输出。
- “response” 用于向用户展示，可以是提示下一步、展示结果或者最终回答。
- “call” 是上层程序用来执行的动作，不执行时可以填 `"name": null, "arguments": {{}}`。
- “end”: true 时，上层循环应停止；否则继续把本次 call 的执行结果反馈给模型。

当前用户最新需求是：
"""
        pending_list = pending_str.split('\n')
        if not pending_str.strip():
            return {
                "response": "任务已结束！",
                "call": {
                    "name": "generate",
                    "arguments": ""
                },
                "end": True
            }

        # 调用 LLM（伪代码）
        tp = client_Qwen.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": tpl_prompt},
                {"role": "user", "content": pending_list[0]}
            ],
            stream=False
        )

        output = tp.choices[0].message.content.strip()
        # 解析 JSON
        try:
            result = json.loads(output)
        except json.JSONDecodeError as e:
            # 如果解析失败，可记录日志并抛出或做默认处理
            raise ValueError(f"LLM 返回的内容不是合法 JSON：{e}\nRaw output:\n{output}")

        # 提取字段，供上层使用
        response_text = result.get("response", "")
        call_name = result.get("call", {}).get("name")
        call_args = result.get("call", {}).get("arguments", {})
        is_end = bool(result.get("end", False))

        # 把人可读部分和动作指令分别发射
        self.output_signal.emit(f"## response\n{response_text}")
        self.output_signal.emit(f"## call\nname: {call_name}\nargs: {call_args}")
        if is_end:
            self.output_signal.emit("## end of workflow")

        return {
            "response": response_text,
            "call": {
                "name": call_name,
                "arguments": call_args
            },
            "end": is_end
        }

    def _query_process(self, present_query: str) -> str:
        """
        将用户最新需求拆解为按执行顺序排列的元任务清单
        """
        tpl_prompt = """
                请将用户需求拆解为元任务链，按执行顺序输出结构化列表。元任务分类及判断规则：

                【元任务类型】
                1. 果蔬分析（需图像识别或质量判断）
                2. 数据库操作（需查询/修改数据库）
                3. 联网搜索（需实时网络信息）
                4. 直接生成（无需上述三类操作）
                5. 定时任务（需要周期性/定时进行的任务）
                6. 信息发送（向用户发送邮件/信息）
                7. 设备调用（调用传感器等质量检测设备）
                8. 增强检索（需要在本地知识库中检索时）
                9. 深度分析（细致分析“果蔬分析”任务检测得到或用户提供的数据）

                【判断规则】
                ① 含图像/光谱分析必选1
                ② 需数据库交互必选2。
                ----凡是涉及到总结等应用数据库数据分析的任务，必须先查询、再分析！
                ----涉及到保存数据到表中的操作，必须先使用DESC查询对应表的字段，再进行插入！不允许直接插入！
                ③ 需最新网络信息必选3
                ④ 仅文本生成时选4
                ⑤ 涉及到定时或周期性任务选5。注意，此时必须保留间隔时间、总时间等重要信息！无需单独列出每一次！
                ⑥ 若用户要求发送邮件或信息选6
                ————选6时，元任务表述必须包含邮箱地址
                ⑦ 涉及“质量检测”，“使用检测设备”等设备调用指令必选7
                ⑧ 有潜在的本地知识库检索需求时，选8
                ⑨ 需要对用户提供的数据或“果蔬分析”任务得到的数据进行进一步分析时，选9

                【输出格式要求】
                按执行顺序逐行输出，每行格式：
                任务序号-任务名称：简要解释
                注意：
                ① 任务序号由任务类型决定，与任务在序列中的顺序无关！
                ② 任何任务的最后，都必须有一个总结性的文本生成！

                【示例】
                输入：检测苹果并生成市场报告
                输出：
                1-果蔬检测：识别苹果质量
                2-数据库操作：查询苹果价格数据
                3-联网搜索：获取最新市场动态
                4-直接生成：综合数据生成报告

                输入：删除过期的苹果数据
                输出：
                2-数据库操作：删除过期苹果记录

                输入：每隔50分钟检测一次砂糖橘
                输出：
                5-定时任务：每隔50分钟检测一次砂糖橘


                避免解释与任何多余的输出，直接回答。当前需解析的用户需求：
                """
        tp = client_Qwen.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": tpl_prompt},
                {"role": "user", "content": present_query}
            ],
            stream=False
        )

        # 直接返回模型的原始分解输出
        output = tp.choices[0].message.content.strip()
        self.output_signal.emit(f"## TODO\n {output}")
        return output

    def _chat(self, t=4, rag_text=None):
        if rag_text is not None:
            response = client_Qwen.chat.completions.create(
                model=self._get_chat_model(t),
                messages=[
                    {"role": "system", "content": self._get_chat_prompt(t)},
                    *(
                        {"role": h["role"], "content": h["content"]}
                        for h in self.memory
                    ),
                    {"role": "user", "content": self._get_chat_prompt(t) + rag_text}
                ],
                stream=False
            )
        else:
            response = client_Qwen.chat.completions.create(
                model=self._get_chat_model(t),
                messages=[
                    {"role": "system", "content": self._get_chat_prompt(t)},
                    *(
                        {"role": h["role"], "content": h["content"]}
                        for h in self.memory
                    ),
                ],
                stream=False
            )

        output = response.choices[0].message.content
        # self.output_signal.emit(output)
        return output

    def _further_analyze(self, content, t=5):
        resp = client_KwooLa.chat.completions.create(
            model=self._get_chat_model(t),
            messages=[
                *(
                    {"role": h["role"], "content": h["content"]}
                    for h in self.memory
                ),
                {"role": "user", "content": content}
            ],
            max_tokens=4096,
        )

        output = resp.choices[0].message.content
        # self.output_signal.emit(output)
        return output

    def _use_tools(self, call: dict):
        # 解析输入
        name          = call.get("name")
        arguments     = call.get("arguments", {})

        if name.startswith("analyze"):
            report = self._fruit_examine(arguments.get("prompt"))
        elif name.startswith("query_db"):
            sql = self._extract_sql(arguments.get("sql"))
            report = self._sql_execute(sql)
        elif name.startswith("search"):
            report = self._apply_online_search()
        elif name.startswith("generate"):
            report = self._chat(t=4)
        elif name.startswith("send_message"):
            to      = arguments.get("to")
            subject = arguments.get("subject")
            self.memory.append({"role": "user", "content": "总结检测结果，汇总为可邮件发送的报告内容。落款为：鲜而易见 FreshAgent"})
            content = self._chat()
            self._send_email(to, subject, content)
            report = f"已成功发送邮件 <{subject}> 至 <{to}>"
        elif name.startswith("enhanced_search"):
            report = self._chat(t=4, rag_text=arguments.get("query"))
        elif name.startswith("further_analyze"):
            report = self._further_analyze(arguments.get("query"))
        else:
            report = "Tool calling 错误"

        return report

    def _update_query(self):
        response = client_Qwen.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system",
                 "content": f"根据用户需求，联网查询信息。直接告知结果，不要进行无关输出。避免输出代码。用户地理位置为{self.location}"},
                *(
                    {"role": h["role"], "content": h["content"]}
                    for h in self.memory
                ),
            ],
            extra_body={
                "enable_search": True
            }
        )
        return response.choices[0].message.content

    def _add_summary_task(self, chain):
        if not chain[-1].startswith("4"):
            chain.append("4-直接生成：对上述执行任务及其结果进行概括总结，形成报告")

    def _apply_adjustments(
            self,
            chain: List[str],
            adjustments: Dict[str, List],
    ) -> List[str]:
        """
        chain: 原始任务链，每项是字符串
        adjustments: {
          "keep":   [<保留的任务>, ...],
          "add":    [<新增的任务>, ...],
          "remove": [<要删除的任务>, ...],
          "update": [
              {"from": <旧描述>, "to": <新描述>},
              ...
          ]
        }
        返回：应用所有调整后的新链
        """
        # 1. 删除
        to_remove = set(adjustments.get("remove", []))
        filtered = [t for t in chain if t not in to_remove]

        # 2. 更新
        updates = adjustments.get("update", [])
        # 构建 from→to 的映射方便查找
        replace_map = {u["from"]: u["to"] for u in updates}
        updated = [
            replace_map.get(t, t)  # 如果有替换，输出替换后的；否则保原样
            for t in filtered
        ]

        # 3. 保留校验（可选）
        # keep_set = set(adjustments.get("keep", []))

        # 4. 新增（插到末尾，也可以在特定 idx 插入）
        to_add = adjustments.get("add", [])
        new_chain = to_add + updated

        return new_chain

    def _dynamic_task_schedule(self, finish, chain):
        tpl_prompt = f"""
请根据下面的“元任务分类与判断规则”，对“当前任务链”进行动态调整，满足非必要不增加、非必要不保留的高效原则
并以 JSON 格式输出结果，格式如下：

{{
  "keep":   [  /* 保留不变的子任务，字符串列表 */ ],
  "add":    [  /* 新增的子任务，按执行顺序排列的字符串列表 */ ],
  "remove": [  /* 需要删除的子任务，字符串列表 */ ],
  "update": [  /* 需要调整的子任务，列表中每项为对象 */ 
    {{
      "from": "<原子任务原描述>",
      "to":   "<原子任务新描述>"
    }}
  ]
}}

不要输出除上述 JSON 之外的任何内容、注释或解释。

【用户要求】
{self.user_target}

【已完成任务】
{finish}

【当前任务链】  
{chain}

【元任务类型】
1. 果蔬分析（需图像识别或质量判断）
2. 数据库操作（需查询/修改数据库）
3. 联网搜索（需实时网络信息）
4. 直接生成（无需上述三类操作）
5. 定时任务（需要周期性/定时进行的任务）
6. 信息发送（向用户发送邮件/信息）
7. 设备调用（调用传感器等质量检测设备）
8. 增强检索（需要在本地知识库中检索时）
9. 深度分析（细致分析“果蔬分析”任务检测得到或用户提供的数据）

【判断规则】
① 含图像/光谱分析必选1
② 需数据库交互必选2。
----凡是涉及到总结等应用数据库数据分析的任务，必须先查询、再分析！
----涉及到保存数据到表中的操作，必须先使用DESC查询对应表的字段，再进行插入！不允许直接插入！
③ 需最新网络信息必选3
④ 仅文本生成时选4
⑤ 涉及到定时或周期性任务选5。注意，此时必须保留间隔时间、总时间等重要信息！无需单独列出每一次！
⑥ 若用户要求发送邮件或信息选6
————选6时，元任务表述必须包含邮箱地址
⑦ 涉及“质量检测”，“使用检测设备”等设备调用指令必选7
⑧ 有潜在的本地知识库检索需求时，选8
⑨ 需要对用户提供的数据或“果蔬分析”任务得到的数据进行进一步分析时，选9

【输出格式要求】
按执行顺序逐行输出，每行格式：
任务序号-任务名称：简要解释
注意：
① 任务序号由任务类型决定，与任务在序列中的顺序无关！
② 任何任务的最后，都必须有一个总结性的文本生成！

【示例】
输入：检测苹果并生成市场报告
输出：
1-果蔬检测：识别苹果质量
2-数据库操作：查询苹果价格数据
3-联网搜索：获取最新市场动态
4-直接生成：综合数据生成报告

输入：删除过期的苹果数据
输出：
2-数据库操作：删除过期苹果记录

输入：每隔50分钟检测一次砂糖橘
输出：
5-定时任务：每隔50分钟检测一次砂糖橘


避免解释与任何多余的输出，直接回答。当前需解析的用户需求：
                        """
        tp = client_Qwen.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": tpl_prompt},
                *(
                    {"role": h["role"], "content": h["content"]}
                    for h in self.memory
                ),
            ],
            stream=False
        )

        # 直接返回模型的原始分解输出
        output = tp.choices[0].message.content.strip()
        output_json = json.loads(output)

        new_chain = self._apply_adjustments(chain, output_json)

        if self.debug:
            print(output)
            print(new_chain)

        self.output_signal.emit(f"## TODO\n {new_chain}")

        return new_chain

    def turn(self, user_input, enhanced_retrieval=False):
        # 1. 记录用户目的
        self.user_target = user_input
        todo = self._query_process(self.user_target)

        if self.debug: print(todo)
        chain = todo.split('\n')
        self._add_summary_task(chain)

        # 2. 进行用户目的的预拆解
        self.memory = []
        self.history.append(
            {"role": "user", "content": user_input + "\n<请同时启用增强检索>\n" if enhanced_retrieval else user_input})
        query = chain[0]
        finish = []

        # 3. 结束参数置为否，进入循环
        is_end = False
        while not is_end:
            # 3.1 获取本轮的 pending_str，取第一项处理
            pending_str  = query
            calling_dict = self.analyze(pending_str)

            if self.debug:
                print(calling_dict)

            response = calling_dict.get("response")
            call     = calling_dict.get("call")
            is_end   = calling_dict.get("end")

            if self.debug:
                print(response)
            self.output_signal.emit(response)

            # 3.2 工具效果
            report = self._use_tools(call)

            # 3.3 记忆管理
            self.memory.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": response})
            self.history.append({"role": "assistant", "content": json.dumps(call)})
            self.memory.append({"role": "assistant", "content": response})
            self.memory.append({"role": "assistant", "content": json.dumps(call)})
            self._history_check()
            self.history.append({"role": "assistant", "content": report})
            self.memory.append({"role": "assistant", "content": report})
            self._history_check()

            if self.debug:
                print(report)
            self.output_signal.emit(report)

            # 3.4 需求更新
            finish.append(chain[0])
            chain.pop(0)

            if chain:
                query = chain[0]
            else:
                break

            if is_end:
                break

            # 3.5 任务链更新
            chain = self._dynamic_task_schedule(finish, chain)


        self.output_signal.emit("## 任务结束！")

    def _enhanced_retrieval(self, user_input):
        data_dir = "LocalDataBase/Data"
        retrieval_info = ""
        if os.path.exists(data_dir):
            self.localDataHandler._check_dir(data_dir)
            try:
                existed_files = self.localDataHandler._get_existed_files()
                if existed_files:
                    file_list = ", ".join([item['file_name'] for item in existed_files])
                    retrieval_info += "本地已存在文件：" + file_list + "\n"
                else:
                    retrieval_info += "本地数据为空。\n"
            except Exception as e:
                retrieval_info += f"检索出错: {e}\n"
            match_info = self.localDataHandler.search_file_by_keyword(user_input)
            retrieval_info += f"关键词匹配结果：\n{match_info}"
        else:
            retrieval_info = "本地数据目录不存在。"
        return retrieval_info

    def _apply_online_search(self):
        response = client_Qwen.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system",
                 "content": f"根据用户需求，联网查询信息。直接告知结果，不要进行无关输出。避免输出代码。用户地理位置为{self.location}"},
                *(
                    {"role": h["role"], "content": h["content"]}
                    for h in self.memory
                ),
            ],
            extra_body={
                "enable_search": True
            }
        )
        return response.choices[0].message.content

    def _apply_alarm_task(self, cmd):
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {
                    "role": "system",
                    "content": """严格按以下规则处理：
        1. 从用户输入中提取 cmd, minutes, total_time 
           其中，cmd代表需要周期执行的内容。minutes为间隔时间（单位为分钟）。total_time为总执行时间。
        2. 必须生成合法JSON对象，包含三个字段
        3. 不要执行任何函数"""
                },
                {"role": "user", "content": cmd}
            ],
            response_format={"type": "json_object"}
        )

        paras = json.loads(response.choices[0].message.content)
        self.history.append({"role": "user", "content": paras["cmd"]})
        paras["cmd"] = self.analyze()
        self.history.pop()
        self._create_alarm_task(paras["cmd"], paras["minutes"], paras["total_time"])

    def _get_table_schema(self) -> str:
        table_sql = """
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = DATABASE()
        """
        table_result = self._sql_execute(table_sql, auto=True, form_json=False)
        column_sql = """
            SELECT 
                TABLE_NAME, 
                COLUMN_NAME, 
                DATA_TYPE, 
                COLUMN_COMMENT 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE()
        """
        column_result = self._sql_execute(column_sql, auto=True, form_json=False)
        schema = {}
        if isinstance(column_result, str):
            return json.loads("{}")
        for col in column_result:
            table_name = col["TABLE_NAME"]
            if table_name not in schema:
                schema[table_name] = []
            schema[table_name].append({
                "column_name": col["COLUMN_NAME"],
                "data_type": col["DATA_TYPE"],
                "comment": col["COLUMN_COMMENT"]
            })
        final_data = [
            {
                "table_name": table["TABLE_NAME"],
                "columns": schema.get(table["TABLE_NAME"], [])
            }
            for table in table_result
        ]
        return self._format_result_as_json(final_data)

    def _get_chat_prompt(self, t=4):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = "告诉用户\"命令解析错误，请重新尝试\""
        if t == 1:
            prompt = f"""
                你是水果质量检测助手，请精准解析用户意图并表示即将开始工作，请用户确认检测设备完好。请你遵守以下协议：
                ⚙️ 执行约束
                - 标记{current_time}时间戳
                📌 当前会话策略：
                    不要重复问题，直接开始回答！
            """
        elif t == 2:
            prompt = f"""
                你是水果检测数据库助手，请精准解析用户意图并根据数据库当前情况生成相应sql代码。请你遵守以下协议：
                1️⃣ 数据操作
                  ├─ 增：INSERT前，必须首先检验表是否存在，若不存在，则创建
                  ├─ 删：DELETE必须带WHERE条件
                  ├─ 改：UPDATE需记录修改时间戳
                  └─ 查：SELECT默认按质评等级排序
                ⚙️ 当前数据库信息
                数据库名：'{self.dbHandler.db_config["database"]}'
                表信息：'{self._get_table_schema()}'
                ⚙️ 执行约束
                - 时间敏感性：所有操作需标记'{current_time}'时间戳
                - SQL安全规范：关键操作需生成确认提示
                - 错误处理：捕获字段缺失异常并引导补充
                📌 当前会话策略：
                不要重复问题，直接开始回答！
                不要在代码中生成任何注释！
                不要进行解释及多余输出！
                确保sql代码被包裹在sql代码块中
                确保语法正确。若使用varchar，必须给定具体长度
            """
        elif t == 3:
            prompt = f"""
                你是信息查询助手，直接告知用户收到联网搜索请求，即将进行查询
                ⚙️ 执行约束
                - 时标记{current_time}时间戳
                📌 当前会话策略：
                不要重复问题，直接开始回答！
            """
        elif t == 4:
            prompt = f"""
                你是水果报告生成助手，请结合知识生成回答，使用自然语言
                   - 知识范围：水果栽培/采后处理/质量分级
                   - 引用表格：若前面的检测生成了表格，必须在报告中体现
                   - 禁用操作：涉及金钱交易的建议
                   - 不确定应答如"请提供更详细的品种信息"
                   ⚙️ 执行约束
                标记'{current_time}'时间戳
                📌 当前会话策略：
                不要重复问题，直接开始回答！
            """
        elif t == 5:
            prompt = f"""
                你是定时任务解析助手，直接告知用户收到定时任务请求，即将进行解析
                ⚙️ 执行约束
                标记'{current_time}'时间戳
                📌 当前会话策略：
                不要重复问题，直接开始回答！
            """
        elif t == 6:
            prompt = f"""
                    你是邮件发送助手，直接告知用户收到邮件发送请求，即将进行任务
                    ⚙️ 执行约束
                    标记'{current_time}'时间戳
                    📌 当前会话策略：
                    不要重复问题，直接开始回答！
                """
        elif t == 7:
            prompt = f"""
                    你是设备调用助手，直接告知用户收到设备调用请求，即将执行任务，并请用户复核设备完好性
                    ⚙️ 执行约束
                    标记'{current_time}'时间戳
                    📌 当前会话策略：
                    不要重复问题，直接开始回答！
                """
        return prompt

    def _get_chat_model(self, t=4):
        if t in [1, 3, 5, 6, 7]:
            return "qwen-turbo"
        elif t == 2:
            return "qwen-coder-plus"
        elif t == 4:
            return "qwen-max"
        else:
            return "KwooLa"


    def _capture_bluetooth(self, gap=1):
        data = self.bluetoothHandler.get_bt_response('COM3', 9600, 269)
        self.bluetoothHandler.disconnect()
        return data

    def _history_check(self):
        if len(self.history) > 10:
            self.history = self.history[:10]

    def _fruit_examine(self, user_input):
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = client_Qwen.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system",
                 "content": f"""你是一个目录名选择器。从{os.listdir("data")}中匹配与用户需求最接近的目录名并直接输出匹配的目录名。避免任何解释与多余内容，只允许输出目录名！若不存在匹配度较高的目录名，输出\"None\""""},
                {"role": "user", "content": user_input}
            ],
            max_tokens=128
        )

        if response.choices[0].message.content.startswith("None"):
            return "未找到有效目录，请确认目录已创建"

        category = client_Qwen.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system",
                 "content": "从用户输入中提取出想要检测的水果品类，并直接输出。不要有任何解释及多余输出。"},
                {"role": "user", "content": user_input}
            ],
            max_tokens=128
        ).choices[0].message.content

        dir_name = response.choices[0].message.content
        self.output_signal.emit(f"## 识别到目录名：{dir_name}")
        self.memory.append({"role": "assistant", "content": f"识别到目录名：{dir_name}"})
        if not os.path.exists("data"):
            os.makedirs("data")

        dir_path = os.path.join("data", dir_name)
        self.output_signal.emit(f"## 检测到{dir_path}文件夹。读取数据进行分析...")
        response = construct_structured_data(dir_path, category, self.FASTSAM_OUTPUT, self.FASTSAM_MODEL)
        content = f"检测完成，报告如下：\n{response}"
        return content

    def _sql_clarity_check(self, sql: str) -> str:
        valid_tables = self.dbHandler.get_table_names()
        self.output_signal.emit("## 检索数据库表名...")
        self.output_signal.emit(f"## 数据库检索到：{valid_tables}")
        if not valid_tables:
            self.output_signal.emit("## 数据库无可用表")
            return sql

        used_tables = self._extract_sql_tables(sql)
        self.output_signal.emit(f"## SQL语句检索到：{used_tables}")
        missing_tables = [t for t in used_tables if t not in valid_tables]

        if not missing_tables:
            self.output_signal.emit("## SQL通过验证")
            return sql

        corrected_sql = self._glm_correct_sql(sql, valid_tables, missing_tables)
        final_tables = self._extract_sql_tables(corrected_sql)
        if all(t in valid_tables for t in final_tables):
            self.output_signal.emit("## SQL语句已修正")
            return corrected_sql
        return sql

    def _extract_sql_tables(self, sql: str) -> List[str]:
        import re
        clean_sql = re.sub(r'--.*?\n|/\*.*?\*/', ' ', sql, flags=re.DOTALL)
        clean_sql = ' '.join(clean_sql.split()).upper()
        patterns = [
            r"(?:DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?|TRUNCATE\s+TABLE\s+|ALTER\s+TABLE\s+)(?:`?(\w+)`?\.)?`?(\w+)`?",
            r"(?:FROM|JOIN)\s+(?!\(SELECT\b)(?:`?(\w+)`?\.)?`?(\w+)`?",
        ]
        tables = []
        for pattern in patterns:
            matches = re.findall(pattern, clean_sql, re.IGNORECASE | re.VERBOSE)
            for match in matches:
                if len(match) == 2:
                    schema, table = match[0], match[1]
                    tables.append(table or schema)
                else:
                    for group in match:
                        if group:
                            tables.append(group)
        return list(set(filter(None, tables)))

    def _glm_correct_sql(self, original_sql: str, valid_tables: List[str], wrong_tables: List[str]) -> str:
        prompt = f"""
        请严格按以下要求修正SQL语句：

        # 任务
        1. 仅将错误的表名修正为可用表名中最相关的表名，保留其他所有内容
        2. 错误表名列表：{wrong_tables}
        3. 可用表名列表：{valid_tables}

        # 输入SQL
        {original_sql}

        # 输出规则
        1. 只输出修正后的SQL，不要任何解释
        2. 使用标准SQL格式，不要代码块标记
        3. 确保表名在可用列表中

        示例：
        输入：SELECT * FROM users
        可用表：employees
        输出：SELECT * FROM employees
        """
        response = client_Qwen.chat.completions.create(
            model="qwen-coder-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        corrected = response.choices[0].message.content.strip()
        if corrected.startswith("```sql"):
            corrected = corrected[6:-3].strip()
        return corrected

    def _extract_sql(self, response_text: str) -> str:
        import re
        code_blocks = re.findall(r'```sql(.*?)```', response_text, re.DOTALL)
        if code_blocks:
            sql = code_blocks[0].strip()
            sql = self._sql_clarity_check(sql)
            sql = sql.replace("\n", "")
            return sql
        raise ValueError("未找到有效SQL语句")

    def _sql_execute(self, sql: str, params: Optional[Union[List, Dict]] = None, auto=True, form_json=True) -> str:
        if not auto:
            check = input("## 请确认操作[y/n]：")
            if check.lower().startswith('y'):
                self.output_signal.emit("## 开始执行...")
            else:
                return f"已取消执行！"

        if not self._is_sql_safe(sql):
            return "## ⚠️ 安全校验失败：禁止执行危险操作"

        result = self.dbHandler.execute(sql, params=params, fetch_all=True)
        self.output_signal.emit(f"## 执行结果：{result}")
        self.output_signal.emit("## 执行完毕！")

        if isinstance(result, list):
            if form_json:
                return self._format_result_as_json(result)
            else:
                return result
        else:
            return f"## 操作成功，受影响行数：{result}"

    def _format_result_as_json(self, result: List[Dict]) -> str:
        import json
        return json.dumps(result, ensure_ascii=False, indent=4)

    def _is_sql_safe(self, sql: str) -> bool:
        return True

    def _markdown_to_html(self, md_content: str) -> str:
        html_content = markdown.markdown(md_content)
        return f"""<div style="
            font-family: 阿里巴巴普惠体 R, sans-serif;
            line-height: 1.6;
            color: TODO333;
        ">{html_content}</div>"""

    def _send_email(self, to_addr, subject, body):
        from_addr = ("鲜而易见FreshAgent", "FreshNIR@163.com")
        html_content = body.replace('\n', '<br>')
        html_body = self._markdown_to_html(html_content)
        self.emailHandler.send_email(from_addr, [to_addr], subject, html_body, is_html=True)

    def _get_email_content(self, cmd):
        example_content = json.dumps(
            {
                "to_addr": "12345@example.com",
                "subject": "检测已结束",
                "content": "您的水果检测任务已经结束！"
            },
            ensure_ascii=False
        )
        response = client_Qwen.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    "role": "system",
                    "content": f"""你是一个邮件发送助手。你的指令如下：
                    1、从命令及对话历史中提取用户的收件地址
                    2、概括邮件的标题
                    3、生成邮件的内容
                    使用json格式回答，三个字段分别为to_addr, subject, content；
                    对于content，生成html格式文本，'\\n'用<br>替代
                    不要进行任何解释及多余输出
                    示例：
                    Q: 水果检测结束后，向12345@example.com发送邮件告知我检测结束
                    A： {example_content}
                    """
                },
                *(
                    {"role": h["role"], "content": h["content"]}
                    for h in self.memory
                ),
                {
                    "role": "user",
                    "content": cmd
                }
            ],
            response_format={"type": "json_object"},
        )
        json_data = json.loads(response.choices[0].message.content)
        return json_data

    def process_image(self, user_input, image_path):
        self.history.append({"role": "user", "content": user_input})
        img_system_prompt = """
        你是专业的果蔬质量检测AI大模型，能够通过图像分析精准识别果蔬表面缺陷、成熟度、规格及品种，并结合多模态数据（如环境参数或用户描述）进行综合评估。
        你将根据用户的输入提供质量报告，包含缺陷定位、保质期预测及处理建议，并确保结果符合农业标准。支持多轮交互与可视化解释。
        """
        url = get_url(image_path)
        if self.enhanced_retrieval:
            retrieval_info = self._enhanced_retrieval(user_input)
            self.output_signal.emit(f"## 增强检索结果：{retrieval_info}")
            combined_input = f"用户输入：{user_input}\n\n增强检索信息：{retrieval_info}"
        else:
            combined_input = user_input
        messages = [
            {"role": "system", "content": img_system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": combined_input},
                {"type": "image_url", "image_url": {"url": url}}
            ]}
        ]
        completion = client_Qwen.chat.completions.create(
            model="qwen-vl-plus",
            messages=messages,
            extra_headers={"X-DashScope-OssResourceResolve": "enable"}
        )
        answer = completion.choices[0].message.content
        self.output_signal.emit(answer)
        self.history.append({"role": "assistant", "content": answer})
        return answer


if __name__ == '__main__':
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "Fruit"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "charset": os.getenv("DB_CHARSET", "utf8mb4"),
        "autocommit": False,
    }
    email_config = {
        "host": os.getenv("EMAIL_HOST", "smtp.163.com"),
        "port": int(os.getenv("EMAIL_PORT", "465")),
        "username": os.getenv("EMAIL_USERNAME", "FreshNIR@163.com"),
        "password": os.getenv("EMAIL_PASSWORD", ""),
        "use_ssl": bool(int(os.getenv("EMAIL_USE_SSL", "1"))),
    }
    agent = FruitAgent(os.getenv("AGENT_LOCATION", "成都市"), db_config, email_config)
    while True:
        user_input = input("==> 用户: ")
        agent.turn(user_input)
