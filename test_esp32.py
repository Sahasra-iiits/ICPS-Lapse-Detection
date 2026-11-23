import serial
import time

PORT = "COM4"
BAUD = 115200

print("Connecting to ESP32...")

ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)

print("Connected to ESP32")
print("Waiting for MAX30100 data...")
print()

try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if line:
            print(line)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    ser.close()