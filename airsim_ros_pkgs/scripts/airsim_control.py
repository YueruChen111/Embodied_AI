"""
The control patterns for Airsim drones, such as take off or reset
"""
import airsim
import time
import sys
import os
from datetime import datetime
import copy
import numpy as np


def confirm_and_initiate_drones(drone_list, height=-10):
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        for drone in drone_list:
            client.enableApiControl(True, drone)
    except:
        print("Airsim connection failed...")
        sys.exit(1)

    print("arming the drone...")
    for drone in drone_list:
        client.armDisarm(True, drone)

    print("taking off...")
    client_state = []
    for drone in drone_list:
        f = client.takeoffAsync(vehicle_name=drone)
        client_state.append(f)
    for ff in client_state:
        ff.join()

    time.sleep(1)
    for drone in drone_list:
        state = client.getMultirotorState(vehicle_name=drone)
        # print(state)
        if state.landed_state == airsim.LandedState.Landed:
            print("take off failed...")
            sys.exit(1)

    print("rising...")
    client_state = []
    for drone in drone_list:
        f1 = client.moveToZAsync(height, velocity=3, vehicle_name=drone)
        client_state.append(f1)
    for ff in client_state:
        ff.join()

    return client


def reset_drones(client, drone_list):
    time.sleep(1)
    print("Reset...")
    for drone in drone_list:
        client.armDisarm(False, vehicle_name=drone)
    client.reset()
    for drone in drone_list:
        client.enableApiControl(False, vehicle_name=drone)
    print("Done!\n")
