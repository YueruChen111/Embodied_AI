#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from airsim_ros_pkgs.srv import SetLocalPosition, Takeoff, Land
import airsim
from nav_msgs.msg import Odometry
import carla
import get_npc_info

if __name__ == '__main__':

    ego_pos=Odometry()
    rospy.init_node('px4_move_control')
    rospy.set_param("/vehicle_name","Drone0")
    rospy.set_param("service_on",False)
    # client = airsim.MultirotorClient()
    # client.enableApiControl(True)   # get control
    # client.armDisarm(True)          # unlock
    # client.takeoffAsync().join()    # takeoff
    in_the_air = False
    rospy.set_param("back",False)
    pub_GT = rospy.Publisher("/ground_truth",String,queue_size=10)
    msg_GT = String()
    msg_GT.data = "Safe"
    rate = rospy.Rate(30.0)
    

    carla_client = carla.Client('localhost',2000)
    carla_client.set_timeout(10.0)
    world = carla_client.get_world()
    player_start=[43.27,-5.84,2.65]

    try:
        while not rospy.is_shutdown():
            # while not in_the_air:
            #     #print(client.getMultirotorState(vehicle_name='Drone0').landed_state)
            #     #in_the_air = client.getMultirotorState(vehicle_name='Drone0').landed_state
            #     print("Waiting for take off")

            rospy.set_param("service_on",True)
            input_valid = False

            while not input_valid:
                person_pos=[]
                vehicle_pos=[]
                ego_pos = rospy.wait_for_message("/airsim_node/Drone0/odom_local_ned", Odometry, timeout=None)
                #state_multirotor = client.getMultirotorState(vehicle_name='Drone0')
                msg_GT.data = "Safe"
                
                #print("Z:",ego_pos.pose.pose.position.z)

                # for pos in pos_list:
                #     if((pos.x-ego_pos.pose.pose.position.x)**2+(pos.y-ego_pos.pose.pose.position.y)**2\
                #        +(pos.z+ego_pos.pose.pose.position.z)**2<=4)or\
                #         ((pos.x**2-ego_pos.pose.pose.position.x**2)+(pos.y**2-ego_pos.pose.pose.position.y**2)<=0.2):
                #         danger_flag = True
                #         break
                # if((client.simGetCollisionInfo(vehicle_name='Drone0').has_collided==True)\
                #    and(state_multirotor.landed_state==1)):
                #     rospy.logwarn('Collision detected')
                #     msg_GT.data = "Collided"
                #     # client.hoverAsync(vehicle_name = 'Drone0').join()
                # elif(danger_flag==True):
                #     rospy.loginfo('Dangerous! Possible to cause damage')
                #     msg_GT.data = "Dangerous"
                hovering_flag,dangerous_flag,min_distance_3d_person,min_distance_3d_vehicle = \
                    get_npc_info.get_npc_information(world,carla.Location(ego_pos.pose.pose.position.x+player_start[0],\
                                                                          ego_pos.pose.pose.position.y+player_start[1],\
                                                                          ego_pos.pose.pose.position.z+player_start[2]))
                # if((client.simGetCollisionInfo(vehicle_name='Drone0').has_collided==True)\
                #    and(state_multirotor.landed_state==1)):
                #     rospy.logwarn('Collision detected')
                #     msg_GT.data = "Collided"
                if(dangerous_flag==True):
                    msg_GT.data="Dangerous!Close to vehicle or human.Closest person:{},Closest vehicle:{}".format(min_distance_3d_person,min_distance_3d_vehicle)
                elif(hovering_flag==True):
                    msg_GT.data="Dangerous!Hovering on human or vehicle"
                else:
                    msg_GT.data = "Safe.Closest person:{},Closest vehicle:{}".format(min_distance_3d_person,min_distance_3d_vehicle)
                pub_GT.publish(msg_GT)

                # if(rospy.get_param("back")==True):
                #     print("time to get back")
                #     client.goHomeAsync(timeout_sec = 3e+38, vehicle_name = 'Drone0').join()
                #     rospy.set_param("X",0)
                #     rospy.set_param("Y",0)
                #     rospy.set_param("Z",0)
                #     rospy.set_param("yaw",0)
                #     rospy.set_param("back",False)

                # try:
                #     if rospy.has_param('X'):
                #         x_pos = rospy.get_param('X')
                #     else:
                #         x_pos = 0
                #     #x_pos = input("X Position: \t")
                #     assert isinstance(float(x_pos), float)
                #     if rospy.has_param('Y'):
                #         y_pos = rospy.get_param('Y')
                #     else:
                #         y_pos = 0
                #     #y_pos = input("Y Position: \t")
                #     assert isinstance(float(y_pos), float)
                #     if rospy.has_param('Z'):
                #         z_pos = rospy.get_param('Z')*(-1)
                #     else:
                #         z_pos = 0
                #     #z_pos = input("Z Position: \t")
                #     assert isinstance(float(z_pos), float)
                #     if rospy.has_param('yaw'):
                #         yaw = rospy.get_param('yaw')
                #     else:
                #         yaw = 0
                #     # yaw = input("Yaw: \t")
                #     if rospy.has_param('Vel'):
                #         Vel = rospy.get_param('Vel')
                #     else:
                #         Vel = 2
                #     assert isinstance(float(Vel), float)
                #     input_valid = True

                # except AssertionError:
                #     print("This input was not correct!")

                # # TODO: Make a custom message structure to handle this
               
                # x_pos = float(x_pos)
                # y_pos = float(y_pos)
                # z_pos = float(z_pos)
                # yaw = float(yaw)
                # Vel = float(Vel)
               
                # client.moveToPositionAsync(x_pos, y_pos, z_pos, Vel).join()
                # #client.moveToPositionAsync(x_pos, y_pos, z_pos, Vel)
                # client.rotateToYawAsync(yaw, timeout_sec = 3e+38, margin = 5, vehicle_name = 'Drone0').join()
                # #client.rotateToYawAsync(yaw, timeout_sec = 3e+38, margin = 5, vehicle_name = 'Drone0')

                # print("Moving Vehicle!")
                rate.sleep()

    except Exception:
            print("There was an exception!")

    # finally:
    #     if in_the_air:
    #         try:
    #             # This calls the service with a message strucutre that is defined
    #             # in the SetLocalPosition.srv
    #             resp1 = land(True)
    #         except rospy.ServiceException as exc:
    #             print("Service did not process request: " + str(exc))