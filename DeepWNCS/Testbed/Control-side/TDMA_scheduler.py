# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 08:15:06 2020

@author: Sihoon
"""
import time
import numpy as np
from A2C import A2C
import torch
import threading
from Util.utils import to_tensor
from controller_manager import controller_manager
from serial_send import serial_send_thread, signal_handler
####### serial ###########
import signal
import serial
import serial.tools.list_ports as sp
import signal
####### UDP #########
import socket, pickle
server_addr = "192.168.0.4"
port = 65501
finish_message = np.array([100])
###### Learning Model #####
path = 'C:/Users/Sihoon/OneDrive - dgist.ac.kr/문시훈_개인자료/Anaconda Code/Reinforcement Learning/DeepWNCS/Inverted_Pendulum_sihoon'
ACTOR_MODEL = path + '/LogData/model/actor.pickle'
ACTOR_MODEL_v2 = path + '/LogData/model/actor_cont.pickle'
ACTOR_MODEL_v3 = path + '/LogData/model/actor_dura_cont.pickle'
ACTOR_MODEL_v4 = path + '/LogData/model/actor_longdura_cont.pickle'

# global variable
testing_interval = 20
mprun_break = False
num_plant = 3
plant_state_dim = 4
plants_state = np.zeros((plant_state_dim * num_plant), dtype=np.float32)


class timer_thread(threading.Thread):
    def __init__(self, system_conf):
        global plants_state
        threading.Thread.__init__(self)
        self.conf = system_conf['conf']
        self.pend_conf = system_conf['pend_conf']
        self.agent = system_conf['agent'] # define an agent
        self.agent.actor_load_model(system_conf['model_path'])
        self.device = system_conf['device']
        
        # define a controller manager
        self.controller_mng = controller_manager(self.conf, self.pend_conf)
        
        # define each plant state variable
        self.plants_state = plants_state
        
        # serial forwarder
        self.serial_fd = serial_fd
        
        self.seq_cnt = 0
        
    def run(self):
        try:
            while not mprun_break:
                second = round(time.time(), 3)
                
                # update plants' state via UDP
                # obtain action from the learning model
                self.controller_mng.ref_update(second)
                state = self.controller_mng.get_state(plants_state) # errorVector
                print("state:", state)
                # action = self.sequence_scheduler()
                action = self.scheduler(state)
                
                commandVector = self.controller_mng.get_commands(plants_state)
                print("action:", action.item())
                # obtain output message
                if action.item() <= self.conf['num_plant']-1:
                    plant_id = action.item()
                    output = (plant_id, commandVector[plant_id])
                else:
                    output = (100, 0.)
                
                # send a control command via wireless network
                serial_send_thread(self.serial_fd, output[0], output[1]) # transmit the output to plant-side
                time.sleep(0.01)    # second unit
        finally:
            print('process finish')
    
    def scheduler(self, state):
        '''
        input: states for all plants
        output: current scheduled plant
        '''
        state_ts = to_tensor(state, is_cuda=False, device=self.device).reshape(-1)
        # print("state_ts:", state_ts)
        dist = self.agent.actor(state_ts)
        action = dist.sample()
        
        return action
    
    def sequence_scheduler(self):
        action = np.array((self.seq_cnt % self.conf['num_plant']))
        self.seq_cnt += 1
        return action
        

def UDP_receive(server_sock):
    global num_plant, plant_state_dim, plants_state
    while True:
        try:
            data, addr = server_sock.recvfrom(300)
            received_message = pickle.loads(data)
            # UDP close
            if received_message.item(0) == finish_message.item(0):
                server_sock.close()
                break 
            else: # update received plants' state values
                plant_id = int(received_message[0, 0])
                plant_state = received_message[1:, 0]   # shape (4,)
                plants_state[plant_state_dim * plant_id : (plant_state_dim * plant_id)+plant_state_dim] = plant_state
            
        except KeyboardInterrupt:
            print("socket close!")
            server_sock.close()
            break
        
if __name__ == '__main__':
    # =============================================================================
    # serial sender
    # =============================================================================
    signal.signal(signal.SIGINT, signal_handler)    # MAKE SIGNAL TO TERMINATE THREAD
    port_lists = sp.comports()  # GET SERIAL PORT LIST
       
    print("=== Serial Port List ===")
    for i in range(len(port_lists)):
        print("[" + str(i) + "] " + port_lists[i][0])
    port_idx = int(input("Select a Port : "))
   
    # GET SERIAL
    serial_fd = serial.Serial(port_lists[port_idx][0],baudrate=115200,timeout=0)
    
    # =============================================================================
    # UDP server (receiver)
    # =============================================================================
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind((server_addr, port))
    
    # =============================================================================
    # configuration
    # =============================================================================
    
    configuration = {
        'num_plant' : num_plant,
        'plant_state_dim' : plant_state_dim,
        'state_dim' : 2*num_plant,
        'action_dim': num_plant+1
        }
    # configuration['state_dim'] = configuration['plant_state_dim'] * num_plant + 1
    # configuration['action_dim'] = num_plant + 1
    
    pend_configuration = {}
    amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    # frequency_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    frequency_list = [0.1, 0.1, 0.1, 0.1, 0.1]
    trigger_list = [100, 200, 300, 300, 300]  # ms
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i,
                                             'amplitude': amplitude_list[i],
                                             'frequency': frequency_list[i],
                                             'trigger_time': trigger_list[i]}
        
    
    device = torch.device("cpu")
    print("device:", device)
    thread_break = False
    
    # A2C agent 생성
    agent = A2C(configuration, device)
    
    # make a total system configuration
    system_conf = {'conf': configuration,
                   'pend_conf': pend_configuration,
                   'agent': agent,
                   'device': device,
                   'model_path': ACTOR_MODEL_v2,
                   'serial': serial_fd
                   }    
    
    timer = timer_thread(system_conf)   # set timer thread
    UDP_receiver = threading.Thread(target=UDP_receive, args=(server_sock,))    # set UDP thread
    
    timer.start()
    UDP_receiver.start()
    
    time.sleep(testing_interval)
    mprun_break = True # close timer thread
    
