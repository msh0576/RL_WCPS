# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:03:59 2020

@author: Sihoon
"""

import socket, pickle
import numpy as np
import threading

server_addr = "192.168.0.4"
port = 65501
finish_message = np.array([100])


def UDP_receive(server_sock):
    while True:
        try:
            data, addr = server_sock.recvfrom(300)
            received_message = pickle.loads(data)
            print("Server, the received data:", received_message)
            if received_message.item(0) == finish_message.item(0):
                server_sock.close()
                break 
        except KeyboardInterrupt:
            print("socket close!")
            server_sock.close()
            break
            
            
if __name__ == "__main__":
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind((server_addr, port))
    print("server socket ready!")
    
    UDP_receiver = threading.Thread(target=UDP_receive, args=(server_sock,))
    UDP_receiver.start()
    
