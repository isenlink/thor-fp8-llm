#!/usr/bin/env python3
"""Thor 串口探测脚本：自动发现 ACM 设备 → 115200 8N1 → 断言 DTR/RTS → 发回车 → 监听 N 秒
用法: sudo python3 /tmp/thor_uart_probe.py [监听秒数, 默认15]
"""
import serial, time, sys, glob, re

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 15

# 发现设备
ports = sorted(glob.glob('/dev/ttyACM*'))
print(f'发现串口: {ports}')
if not ports:
    sys.exit('无 ACM 设备')

for port in ports:
    # 读序列号
    try:
        with open(f'/sys/class/tty/{port.split("/")[-1]}/device/../idVendor') as f:
            vid = f.read().strip()
    except Exception:
        vid = '?'
    print(f'\n===== {port} (vendor={vid}) =====')
    try:
        s = serial.Serial()
        s.port = port
        s.baudrate = 115200
        s.bytesize = serial.EIGHTBITS
        s.parity = serial.PARITY_NONE
        s.stopbits = serial.STOPBITS_ONE
        s.timeout = 0.5
        s.dtr = False; s.rts = False
        s.open()
        s.dtr = True; s.rts = True
        time.sleep(0.5)
        s.reset_input_buffer()
        s.write(b'\r\n')
        time.sleep(0.5)
        s.write(b'\r\n')
        buf = b''
        t0 = time.time()
        while time.time() - t0 < LISTEN:
            n = s.in_waiting
            if n:
                buf += s.read(n)
                if len(buf) > 8192:
                    break
            time.sleep(0.1)
        cd, dsr, ri, cts = s.cd, s.dsr, s.ri, s.cts
        s.close()
        print(f'收到 {len(buf)} bytes | CD={cd} DSR={dsr} RI={ri} CTS={cts}')
        if buf:
            print('--- 前 500 字节 (cat -v) ---')
            print(buf[:500].decode('latin-1'))
            print('--- hexdump 前 128 字节 ---')
            for i in range(0, min(len(buf), 128), 16):
                chunk = buf[i:i+16]
                hexs = ' '.join(f'{b:02x}' for b in chunk)
                asc = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                print(f'{i:04x}  {hexs:<48}  {asc}')
        else:
            print('(静默)')
    except Exception as e:
        print(f'ERROR: {e}')
