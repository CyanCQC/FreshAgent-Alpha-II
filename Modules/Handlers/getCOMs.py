import serial.tools.list_ports

def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未找到任何可用串口。")
        return

    print("可用串口列表：")
    for port in ports:
        print(f"设备名称: {port.device}")
        print(f"描述信息: {port.description}")
        print(f"硬件ID : {port.hwid}")
        print(f"厂商ID : {port.vid}")
        print(f"产品ID : {port.pid}")
        print(f"序列号 : {port.serial_number}")
        print(f"位置    : {port.location}")
        print(f"制造商 : {port.manufacturer}")
        print(f"产品名 : {port.product}")
        print("-" * 40)

if __name__ == "__main__":
    list_serial_ports()
