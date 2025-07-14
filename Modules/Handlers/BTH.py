import serial
import bluetooth
import logging
import numpy as np

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

class BluetoothHandler:
    def __init__(self):
        """初始化 BluetoothHandler，创建空的串行对象。"""
        self.ser = None
        self.data = []

    def discover_devices(self):
        """
        查询附近的蓝牙设备。
        返回值: 包含 (MAC 地址, 设备名称) 的设备列表。
        """
        logging.info("Scanning Bluetooth devices...")
        devices = bluetooth.discover_devices(lookup_names=True)
        return devices

    def _connect(self, port_name, baudrate=9600):
        """
        连接到指定的串口。
        参数:
            port_name (str): 串口名称，例如 'COM3'。
            baudrate (int): 波特率，默认为 9600。
        """
        try:
            self.ser = serial.Serial(port_name, baudrate)
            logging.info("Connected to device")
        except serial.SerialException as e:
            logging.error("Connection failed: %s", e)
            self.ser = None

    def get_data_cmd(self, command):
        """
        向设备发送命令并接收返回的数据。
        参数:
            command (str): 要发送的命令。
        返回值: 接收到的数据 (UTF-8 解码后的字符串)，或 None (未连接或出错时)。
        """
        if self.ser is None or not self.ser.is_open:
            logging.warning("Device not connected")
            return None
        try:
            self.ser.write(command.encode('utf-8'))
            data = self.ser.read(269)
            return data.decode('utf-8')
        except serial.SerialException as e:
            logging.error("Get data failed: %s", e)
            return None

    def _get_data_flow(self, flow_size=269):
        """
        从设备读取数据（设备主动推送）。
        返回值: 读取到的数据 (UTF-8 解码后的字符串)，或 None (未连接或出错时)。
        """
        if self.ser is None or not self.ser.is_open:
            logging.warning("Device not connected")
            return None
        try:
            data = self.ser.read(flow_size)
            return data.decode('utf-8')
        except serial.SerialException as e:
            logging.error("Get data failed: %s", e)
            return None

    def _disconnect(self):
        """关闭与设备的连接。"""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            logging.info("Connection closed")
            self.ser = None

    def get_bt_response(self, port_name, baudrate=115200, flow_size=269):
        # Initialize
        self.data.clear()

        # Connect
        self._connect(port_name, baudrate)
        data_str = self._get_data_flow(flow_size)

        # To array
        # data_array = np.array([
        #     [float(num) for num in line.split(',')]
        #     for line in data_str.strip().split('\n')
        # ])
        # print(data_array.mean(axis=0))

        # return data_array.mean(axis=0)
        return data_str


if __name__ == '__main__':
    handler = BluetoothHandler()
    # devices = handler.discover_devices()
    # print(devices)

    # 替换为您的串口名称，例如 'COM3'
    print(handler.get_bt_response('COM3', 115200, 269))
