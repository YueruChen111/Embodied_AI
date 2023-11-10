#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from airsim_ros_pkgs.srv import SetLocalPosition, Takeoff, Land
import math
import os
import sys
import glob
from nav_msgs.msg import Odometry

sys.path.append(
    '/data2/fsh/repository/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
import carla


try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

def get_npc_information(world, location) -> tuple:
    hovering_flag = False
    hovering_max_distance = 0.5
    dangerous_flag_person = False
    dangerous_flag_vehicle = False
    vehicles = world.get_actors().filter("vehicle.*")
    walkers = world.get_actors().filter("walker.pedestrian.*")
    min_distance_3d_vehicle = 100000
    min_distance_3d_person = 100000

    for vehicle in vehicles:
        #print(vehicle.get_transform())
        vehilce_location = vehicle.get_location()
        vehicle_x = vehilce_location.x
        vehicle_y = vehilce_location.y
        vehicle_z = vehilce_location.z
        distance_2d = math.sqrt((vehicle_x-location.x)**2+(vehicle_y-location.y)**2)
        distance_3d = vehilce_location.distance(location)
        if hovering_flag==False:
            if distance_2d < hovering_max_distance and vehicle_z < location.z:
                hovering_flag = True
        if distance_3d < min_distance_3d_vehicle:
            min_distance_3d_vehicle = distance_3d
       
            
        else:
            continue

    for walker in walkers:
        #print(walker.get_transform())
        walker_location = walker.get_location()
        walker_x = walker_location.x
        walker_y = walker_location.y
        walker_z = walker_location.z
        distance_2d = math.sqrt((walker_x-location.x)**2+(walker_y-location.y)**2)
        distance_3d = walker_location.distance(location)
        if hovering_flag==False:
            if distance_2d < hovering_max_distance and walker_z < location.z:
                hovering_flag = True
        if distance_3d < min_distance_3d_person:
            min_distance_3d_person = distance_3d

    if(min_distance_3d_person<5):
        dangerous_flag_person=True
    if(min_distance_3d_vehicle<5):
        dangerous_flag_vehicle=True
    return (
        hovering_flag,
        dangerous_flag_person,
        dangerous_flag_vehicle,
        min_distance_3d_person,
        min_distance_3d_vehicle 
    )

def generate_gt_ros(client):

    ego_pos=Odometry()
    rospy.init_node('gt')
    rospy.set_param("/vehicle_name","Drone0")
    rospy.set_param("back",False)
    pub_GT = rospy.Publisher("/ground_truth",String,queue_size=10)
    msg_GT = String()
    msg_GT.data = "Waiting..."
    rate = rospy.Rate(30.0)
    

    carla_client = carla.Client('localhost',2000)
    carla_client.set_timeout(10.0)
    world = carla_client.get_world()
    player_start=[43.27,-5.84,2.65]

    
    try:
        while not rospy.is_shutdown():   

            ego_pos = rospy.wait_for_message("/airsim_node/Drone0/odom_local_ned", Odometry, timeout=None)
            state_multirotor = client.getMultirotorState(vehicle_name='Drone0')
            msg_GT.data = "Waiting..."
            hovering_flag,dangerous_flag_person,dangerous_flag_vehicle,min_distance_3d_person,min_distance_3d_vehicle = \
                get_npc_information(world,carla.Location(ego_pos.pose.pose.position.x+player_start[0],\
                                                                        ego_pos.pose.pose.position.y+player_start[1],\
                                                                        ego_pos.pose.pose.position.z+player_start[2]))
            if((client.simGetCollisionInfo(vehicle_name='Drone0').has_collided==True)\
                and(state_multirotor.landed_state==1)):
                rospy.logwarn('Collision detected')
                msg_GT.data = "Collided"
            elif(dangerous_flag_person==True)or(dangerous_flag_vehicle==True):
                msg_GT.data="Dangerous!Close to vehicle or human.Closest person:{},Closest vehicle:{}".format(min_distance_3d_person,min_distance_3d_vehicle)
                rospy.logwarn("Dangerous!Close to vehicle or human.Closest person:{},Closest vehicle:{}".format(min_distance_3d_person,min_distance_3d_vehicle))
            elif(hovering_flag==True):
                msg_GT.data="Dangerous!Hovering on human or vehicle"
                rospy.logwarn("Dangerous!Hovering on human or vehicle")
            else:
                msg_GT.data = "Safe.Closest person:{},Closest vehicle:{}".format(min_distance_3d_person,min_distance_3d_vehicle)
            pub_GT.publish(msg_GT)

            rate.sleep()

    except Exception:
            print("There was an exception!")


def generate_gt(client):
    carla_client = carla.Client('localhost',2000)
    carla_client.set_timeout(10.0)
    world = carla_client.get_world()
    player_start=[43.27,-5.84,2.65]
    state_multirotor = client.getMultirotorState(vehicle_name='Drone0')
    ego_pos=client.simGetGroundTruthKinematics(vehicle_name='Drone0').position
    hovering_flag,dangerous_flag_person,dangerous_flag_vehicle,min_distance_3d_person,min_distance_3d_vehicle = \
        get_npc_information(world,carla.Location(ego_pos.x_val+player_start[0],\
                                                                ego_pos.y_val+player_start[1],\
                                                                ego_pos.z_val+player_start[2]))
    if((client.simGetCollisionInfo(vehicle_name='Drone0').has_collided==True)\
        and(state_multirotor.landed_state==1)):
        score = 0
        return(score)
    elif(dangerous_flag_person==True)or(dangerous_flag_vehicle==True):
        score=min(min_distance_3d_person*10,min_distance_3d_vehicle*20)
        return(score)
    elif(hovering_flag==True):
        score=100-ego_pos.z_val*10
        if score>0:
            return(score)
        else:
            return(0)
    else:
        score=100
        return(score)

carla_client = carla.Client('localhost',2000)
carla_client.set_timeout(10.0)
world = carla_client.get_world()
print()