import sys, os, zipfile, markdown, datetime, uuid, shutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QLineEdit, QPushButton, QAction, QFileDialog,
    QSplitter, QMessageBox, QToolBar, QStyle, QSizePolicy, QLabel
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QUrl, QSize
from PyQt5.QtGui import QTextCursor, QDesktopServices, QIcon, QFont, QPixmap
from FreshAgent import FruitAgent

# ========= 主题常量 ========= #
LIGHT_STYLE = """
QWidget          { background:#F7F9FB; color:#333; font:16px "Alibaba PuHuiTi"; }
QToolBar         { background:#2E8B57; border:none; }
QToolButton      { color:white; padding:6px 12px; }
QLineEdit        { background:white; border:1px solid #CCC; border-radius:4px; padding:6px; }
QPushButton      { background:#2E8B57; color:white; border:none; border-radius:4px; padding:6px 12px; }
QPushButton:hover{ background:#25724A; }
QPushButton:disabled{ background:#AAA; }
"""
DARK_STYLE  = """
QWidget          { background:#202124; color:#E8EAED; font:16px "Alibaba PuHuiTi"; }
QToolBar         { background:#444; border:none; }
QToolButton      { color:#E8EAED; padding:6px 12px; }
QLineEdit        { background:#303134; border:1px solid #555; border-radius:4px; padding:6px; color:#E8EAED; }
QPushButton      { background:#5F6368; color:#E8EAED; border:none; border-radius:4px; padding:6px 12px; }
QPushButton:hover{ background:#757982; }
QPushButton:disabled{ background:#555; }
"""

# ========= 子线程包装 ========= #
class AgentWorker(QThread):
    finished = pyqtSignal()
    aborted  = pyqtSignal()
    def __init__(self, agent, prompt, enhanced):
        super().__init__()
        self.agent = agent
        self.prompt = prompt
        self.enhanced = enhanced
    def run(self):
        try:
            self.agent.turn(self.prompt, enhanced_retrieval=self.enhanced)
        except Exception as e:
            self.agent.output_signal.emit(f"**⛔ 发生错误：** {e}")
        finally:
            self.finished.emit()
    def stop(self):
        self.terminate()
        self.aborted.emit()

class ImageAgentWorker(QThread):
    finished = pyqtSignal()
    aborted  = pyqtSignal()
    def __init__(self, agent, prompt, image_path):
        super().__init__()
        self.agent = agent
        self.prompt = prompt
        self.image_path = image_path
    def run(self):
        try:
            self.agent.process_image(self.prompt, self.image_path)
        except Exception as e:
            self.agent.output_signal.emit(f"**⛔ 发生错误：** {e}")
        finally:
            self.finished.emit()
    def stop(self):
        self.terminate()
        self.aborted.emit()

# ========= 主窗口 ========= #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("“鲜”而易见 · FreshAgent")
        self.resize(1600, 900)

        self.current_image_path = None
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(100, 100)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)

        tb = QToolBar("MainBar")
        tb.setIconSize(QSize(20,20))
        self.addToolBar(tb)
        theme_act = QAction("💡", self)
        theme_act.triggered.connect(self.toggle_theme)
        tb.addAction(theme_act)
        tb.addSeparator()
        self.statusBar().showMessage("数据库：Fruit  |  增强检索：关闭")

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        lytL = QVBoxLayout(left)
        lytL.setContentsMargins(6,6,6,6)
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        self.chat.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lytL.addWidget(self.chat)

        inLayout = QHBoxLayout()
        self.btnUploadImage = QPushButton("🖼️")
        self.btnUploadImage.setFixedSize(40, 40)
        self.btnUploadImage.clicked.connect(self.upload_image)
        inLayout.addWidget(self.btnUploadImage)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ctrl+Enter 发送…")
        self.input.setMinimumHeight(40)
        self.input.returnPressed.connect(self.send_msg_shortcut)
        inLayout.addWidget(self.input)
        inLayout.addWidget(self.thumbnail_label)
        self.btnSend = QPushButton("发送")
        self.btnSend.setMinimumHeight(40)
        self.btnSend.clicked.connect(self.do_send)
        inLayout.addWidget(self.btnSend)
        lytL.addLayout(inLayout)
        splitter.addWidget(left)

        right = QWidget()
        rLyt = QVBoxLayout(right)
        rLyt.setContentsMargins(6,6,6,6)
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png").scaledToWidth(200)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        rLyt.addWidget(logo_label)
        btn_uploadZip = QPushButton("📦 导入数据集 (.zip)")
        btn_uploadZip.clicked.connect(self.upload_zip)
        btn_uploadFile= QPushButton("📑 上传文档到知识库")
        btn_uploadFile.clicked.connect(self.upload_kb_file)
        btn_viewData  = QPushButton("🗁 打开数据目录")
        btn_viewData.clicked.connect(lambda:_open_dir("data"))
        self.btnRetrieval = QPushButton("⚙ 增强检索  OFF")
        self.btnRetrieval.setCheckable(True)
        self.btnRetrieval.clicked.connect(self.toggle_retrieval)
        btn_clear   = QPushButton("⚡ 清屏")
        btn_clear.clicked.connect(self.chat.clear)
        for b in (btn_uploadZip, btn_uploadFile, btn_viewData, self.btnRetrieval, btn_clear):
            b.setMinimumHeight(48)
            rLyt.addWidget(b)
        rLyt.addStretch()
        splitter.addWidget(right)
        splitter.setStretchFactor(0,3)

        db_cfg = dict(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "Fruit"),
            port=int(os.getenv("DB_PORT", "3306")),
            charset=os.getenv("DB_CHARSET", "utf8mb4"),
            autocommit=False,
        )
        email_cfg = dict(
            host=os.getenv("EMAIL_HOST", "smtp.163.com"),
            port=int(os.getenv("EMAIL_PORT", "465")),
            username=os.getenv("EMAIL_USERNAME", "FreshNIR@163.com"),
            password=os.getenv("EMAIL_PASSWORD", ""),
            use_ssl=bool(int(os.getenv("EMAIL_USE_SSL", "1"))),
        )
        self.agent = FruitAgent("成都", db_cfg, email_cfg)
        self.agent.enhanced_retrieval = False
        self.agent.output_signal.connect(lambda text: self.add_message("agent", text))

        self.show_welcome()
        self.dark = False
        self.setStyleSheet(LIGHT_STYLE)

    def show_welcome(self):
        self.chat.setHtml("""
        <div style='text-align:center;padding:60px 0;font-size:22px;color:#666'>
            👋 <b>您好，我是“鲜”而易见智能体</b><br>
            试着向我提问水果质量检测、市场报告等问题吧~
        </div>""")

    def send_msg_shortcut(self):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.do_send()

    def do_send(self):
        prompt = self.input.text().strip()
        if not prompt:
            return
        if self.current_image_path:
            self.add_message("user", f"{prompt}<br><img src='file:///{self.current_image_path}' width='150'><br>")
        else:
            self.add_message("user", prompt)
        self.input.clear()
        self.btnSend.setEnabled(False)
        self.thumbnail_label.clear()

        if self.current_image_path:
            self.worker = ImageAgentWorker(self.agent, prompt, self.current_image_path)
        else:
            self.worker = AgentWorker(self.agent, prompt, self.agent.enhanced_retrieval)
        self.worker.finished.connect(lambda: self.btnSend.setEnabled(True))
        self.worker.start()
        self.current_image_path = None

    def add_message(self, role, text):
        if "👋" in self.chat.toPlainText():
            self.chat.clear()
        stamp = datetime.datetime.now().strftime("%H:%M")
        if role == "user":
            md = f"**{stamp}  🧑‍💻 你：**  \n{text}"
        else:
            md = f"**{stamp}  🤖 FreshAgent：**  \n{text}"
        html = f"<div style='margin:8px 0'>{markdown.markdown(md)}<br></div>"
        self.chat.insertHtml(html)
        self.chat.moveCursor(QTextCursor.End)
        self.chat.ensureCursorVisible()

    def upload_zip(self):
        path,_ = QFileDialog.getOpenFileName(self,"选择数据集(.zip)","","Zip Files (*.zip)")
        if not path: return
        if not zipfile.is_zipfile(path):
            QMessageBox.warning(self,"格式错误","请选择合法 zip 文件"); return
        target="data"; os.makedirs(target,exist_ok=True)
        zipfile.ZipFile(path).extractall(target)
        QMessageBox.information(self,"成功","数据集已导入并解压")

    def upload_kb_file(self):
        path,_= QFileDialog.getOpenFileName(self,"选择文档",".",
                    "PDF/TXT (*.pdf *.txt)")
        if not path: return
        try:
            self.agent.localDataHandler._upload_file(path)
            QMessageBox.information(self,"成功","知识库已更新")
        except Exception as e:
            QMessageBox.critical(self,"失败",str(e))

    def upload_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"选择图片","",
                    "Image Files (*.png *.jpg *.bmp)")
        if not path: return
        try:
            os.makedirs("images", exist_ok=True)
            ext = os.path.splitext(path)[1]
            new_filename = str(uuid.uuid4()) + ext
            new_path = os.path.join("images", new_filename)
            shutil.copy(path, new_path)
            abs_path = os.path.abspath(new_path).replace("\\", "/")
            self.current_image_path = abs_path

            pixmap = QPixmap(abs_path).scaled(100, 100, Qt.KeepAspectRatio)
            self.thumbnail_label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self,"错误",f"无法上传图片：{e}")

    def toggle_retrieval(self):
        self.agent.enhanced_retrieval = self.btnRetrieval.isChecked()
        state = "ON" if self.agent.enhanced_retrieval else "OFF"
        self.btnRetrieval.setText(f"⚙ 增强检索  {state}")
        self.statusBar().showMessage(f"数据库：Fruit  |  增强检索：{'开启' if state=='ON' else '关闭'}")

    def toggle_theme(self):
        self.dark = not self.dark
        self.setStyleSheet(DARK_STYLE if self.dark else LIGHT_STYLE)

# ------- 工具：打开目录 ------- #
def _open_dir(folder):
    if not os.path.exists(folder): os.makedirs(folder)
    QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(folder)))

# ========= 入口 ========= #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("鲜而易见 · FreshAgent Chat")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())