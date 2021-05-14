# import serial
# import serial.tools.list_ports as sp
# import signal
# import threading
import struct

def signal_handler(sig, frame):
	exitThread = True

def makeCRC(crc, field):
	value_crc = crc ^ (field << 8)

	for i in range(8):
		if ((value_crc & 32768) == 32768):
			value_crc = (value_crc << 1) ^ 4129
		else:
			value_crc = value_crc << 1

	return value_crc & 0xffff

def serial_send_thread(serial_fd, pid, u):
	# removed {couple of SYNC (1byte)} and {CRC (2bytes)}
	header = [0x45, 0x00, 0x00, pid, 0x00, 0x00, 0x08, 0x22, 0x06]
	payload = list(bytearray(struct.pack('d', u)))
	temp_packet = header + payload
	
	packet = [0x7E]
	crc = 0
	for b in temp_packet:
		if b == 0x7E:
			packet.append(0x7D)
			packet.append(0x5E)
		elif b == 0x7D:
			packet.append(0x7D)
			packet.append(0x5D)
		else:
			packet.append(b)
		crc = makeCRC(crc, b)
	
	packet.append(crc & 255)
	packet.append(crc >> 8)
	packet.append(0x7E)
	serial_fd.write(bytearray(packet))

# if __name__ == "__main__":
#     # MAKE SIGNAL TO TERMINATE THREAD
#     signal.signal(signal.SIGINT, signal_handler)
    
#     # GET SERIAL PORT LIST
#     port_lists = sp.comports()
       
#     print("=== Serial Port List ===")
#     for i in range(len(port_lists)):
#         print("[" + str(i) + "] " + port_lists[i][0])
#     port_idx = int(input("Select a Port : "))
   
#     # GET SERIAL
#     serial_fd = serial.Serial(port_lists[port_idx][0],baudrate=115200,timeout=0)
#     serial_send_thread(serial_fd, 24, 1.23)
    
 	  # # MAKE THREAD
 	  # thread_1 = threading.Thread(target=serial_send_thread, args=(serial_fd,23,1.5246))

 	  # # START THREAD
 	  # thread_1.start()




