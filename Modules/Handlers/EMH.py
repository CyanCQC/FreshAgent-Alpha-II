import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from typing import List, Dict, Union, Optional

class EmailHandler:
    def __init__(self, email_config: dict):
        """
        初始化邮件处理器

        :param email_config: 邮件配置字典，包含以下键：
            - host: SMTP服务器地址（必需）
            - port: SMTP端口（可选，根据加密方式自动设置默认值）
            - username: 用户名（可选）
            - password: 密码（可选）
            - use_tls: 启用TLS加密（默认False）
            - use_ssl: 启用SSL加密（默认False）
        """
        self.smtp_host = email_config['host']
        self.username = email_config.get('username')
        self.password = email_config.get('password')
        self.use_tls = email_config.get('use_tls', False)
        self.use_ssl = email_config.get('use_ssl', False)

        if self.use_ssl and self.use_tls:
            raise ValueError("use_ssl 和 use_tls 不能同时为 True")

        # 设置默认端口
        self.smtp_port = email_config.get('port')
        if self.smtp_port is None:
            if self.use_ssl:
                self.smtp_port = 465
            elif self.use_tls:
                self.smtp_port = 587
            else:
                self.smtp_port = 25

    def send_email(
            self,
            from_addr: Union[str, tuple],
            to_addrs: List[Union[str, tuple]],
            subject: str,
            body: str,
            is_html: bool = False
    ) -> None:
        """
        发送电子邮件

        :param from_addr: 发件人地址，支持格式：
            - "name@example.com"
            - ("显示名称", "name@example.com")
        :param to_addrs: 收件人列表，每个元素支持格式同from_addr
        :param subject: 邮件主题
        :param body: 邮件正文
        :param is_html: 是否为HTML格式（默认纯文本）
        """
        # 处理发件人信息
        if isinstance(from_addr, tuple):
            from_display = formataddr(from_addr)
            from_email = from_addr[1]
        else:
            from_display = from_addr
            _, from_email = parseaddr(from_addr)

        # 处理收件人信息
        to_emails = []
        to_displays = []
        for addr in to_addrs:
            if isinstance(addr, tuple):
                to_displays.append(formataddr(addr))
                to_emails.append(addr[1])
            else:
                to_displays.append(addr)
                _, email_addr = parseaddr(addr)
                to_emails.append(email_addr if email_addr else addr)

        # 创建邮件对象
        msg = MIMEText(body, 'html' if is_html else 'plain')
        msg['Subject'] = subject
        msg['From'] = from_display
        msg['To'] = ', '.join(to_displays)

        server = None
        try:
            # 建立SMTP连接
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
                server.ehlo()  # 显式调用 EHLO
            elif self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
                server.ehlo()  # 显式调用 EHLO

            # 登录认证（如果有凭证）
            if self.username and self.password:
                server.login(self.username, self.password)

            # 发送邮件
            server.sendmail(from_email, to_emails, msg.as_string())
        except smtplib.SMTPException as e:
            raise RuntimeError(f"邮件发送失败: {str(e)}")
        finally:
            if server:
                server.quit()

if __name__ == '__main__':
    pass