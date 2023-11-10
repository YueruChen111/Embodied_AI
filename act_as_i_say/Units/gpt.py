#!/usr/bin/env python
# license removed for brevity
import rospy
import airsim
#import tf2_ros
import numpy as np
import math
import threading
from numpy.linalg import inv
from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from mavros_msgs.srv import SetMode
from pyquaternion import Quaternion


class ENV_API:

    def __init__(self):
        rospy.init_node('ENV_API', anonymous=True)

        print("take off!")

        self.name_space = rospy.get_namespace().strip('/')
        # self.object_name = []
        abs_pos=[45.2,-5.84,2.65]
        abs_red_car=[61.0,-6.8,2]
        abs_bicycle=[53.2,-1.2,0.5]
        abs_person1=[49.60,-0.8,1.0]
        abs_person2=[51.80,-10.1,1.0]
        abs_person3=[49.4,-8.9,0.8]
        red_car_pos=Odometry()
        red_car_pos.pose.pose.position.x=abs_red_car[0]-abs_pos[0]
        red_car_pos.pose.pose.position.y=abs_red_car[1]-abs_pos[1]
        red_car_pos.pose.pose.position.z=abs_red_car[2]-abs_pos[2]
        bicycle_pos=Odometry()
        bicycle_pos.pose.pose.position.x=abs_bicycle[0]-abs_pos[0]
        bicycle_pos.pose.pose.position.y=abs_bicycle[1]-abs_pos[1]
        bicycle_pos.pose.pose.position.z=abs_bicycle[2]-abs_pos[2]
        person1_pos=Odometry()
        person1_pos.pose.pose.position.x=abs_person1[0]-abs_pos[0]
        person1_pos.pose.pose.position.y=abs_person1[1]-abs_pos[1]
        person1_pos.pose.pose.position.z=abs_person1[2]-abs_pos[2]
        person2_pos=Odometry()
        person2_pos.pose.pose.position.x=abs_person2[0]-abs_pos[0]
        person2_pos.pose.pose.position.y=abs_person2[1]-abs_pos[1]
        person2_pos.pose.pose.position.z=abs_person2[2]-abs_pos[2]
        person3_pos=Odometry()
        person3_pos.pose.pose.position.x=abs_person3[0]-abs_pos[0]
        person3_pos.pose.pose.position.y=abs_person3[1]-abs_pos[1]
        person3_pos.pose.pose.position.z=abs_person3[2]-abs_pos[2]
        self.object_position ={"red car":red_car_pos, "bicycle":bicycle_pos, "person standing on the side of the road":person1_pos, "person on the road":person2_pos, "girl on the road":person3_pos}
        self.ego_position_sub = rospy.Subscriber("/airsim_node/drone_1/odom_local_ned", Odometry, self.save_ego_position)
        
        self.rate = rospy.Rate(10)
        
        print("Service has started")
        # if not self.set_offboard_mode():
        #     rospy.logerr("Failed to set OFFBOARD mode")
        #     return
        self.ego_position = Odometry()
        self.got_ego_pose = False
        # take off
        # rospy.set_param("X",0)
        # rospy.set_param("Y",0)
        # rospy.set_param("Z",0)
        # rospy.set_param("yaw",0)

    def run(self):
        rospy.spin()
    def save_ego_position(self, ego_position):
        self.ego_position = ego_position
        self.ego_position.pose.pose.position.z = -self.ego_position.pose.pose.position.z
        self.got_ego_pose = True
    
    def get_obj_pos(self, name):
        pos = self.object_position[name]
        print('global pose: ',pos)
        orientation = self.ego_position.pose.pose.orientation
        position = self.ego_position.pose.pose.position
        roll, pitch, yaw = self.quaternion_to_euler([orientation.x,orientation.y,orientation.z,orientation.w])
        # print('roll: ',roll,' pitch: ',pitch,'yaw: ',yaw)
        Quaternion_filter = self.euler_to_quaternion(0, 0, yaw)
        # print('after change: ',Quaternion_filter)
        local2world = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
        world2local_rotation = inv(local2world.rotation_matrix)
        world_to_local_trans_mat = self.get_trans_mat_world2local(position,world2local_rotation)
        global_pose = [pos.pose.pose.position.x, pos.pose.pose.position.y, pos.pose.pose.position.z, 1]
        local_pose = world_to_local_trans_mat @ global_pose
        return  [local_pose[0], local_pose[1], local_pose[2], 0, 0, 0]
    def get_ego_pos(self):
        return [0,0,0,0,0,0]

    def get_obj_names(self):
        return ['red car', 'bicycle', 'person standing on the side of the road', 'person on the road', 'girl on the road']

    def is_obj_visible(self,name):
        return True or False
    def move_to_pos(self,target_position):
        print('target pose: ',target_position)
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        orientation = self.ego_position.pose.pose.orientation
        position = self.ego_position.pose.pose.position
        roll, pitch, yaw = self.quaternion_to_euler([orientation.x,orientation.y,orientation.z,orientation.w])
        # print('roll: ',roll,' pitch: ',pitch,'yaw: ',yaw)
        #print('origin yaw: ',yaw)
        yaw += target_position[3]
        #print('changed yaw: ',yaw)
        Quaternion_filter = self.euler_to_quaternion(0, 0, yaw)
        # print('after change: ',Quaternion_filter)
        local2world = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
        world2local_rotation = inv(local2world.rotation_matrix)
        local_to_world_trans_mat = inv(self.get_trans_mat_world2local(position,world2local_rotation))
        target_point_local = np.append(target_position[:3], 1) 
        target_pose_world = local_to_world_trans_mat @ target_point_local
        target_pose.pose.position.x = float(target_pose_world[0])
        target_pose.pose.position.y = float(target_pose_world[1])
        target_pose.pose.position.z = float(target_pose_world[2])
        target_pose.pose.orientation.w = float(Quaternion_filter[0])
        target_pose.pose.orientation.x = float(Quaternion_filter[1])
        target_pose.pose.orientation.y = float(Quaternion_filter[2])
        target_pose.pose.orientation.z = float(Quaternion_filter[3])

        print('global target pose: ', target_pose.pose.position)
        _,_,target_yaw = self.quaternion_to_euler([orientation.x,orientation.y,orientation.z,orientation.w])
        target_yaw+=target_position[3]
        print('changed yaw: ',yaw)
        rospy.set_param("X",target_pose.pose.position.x)
        rospy.set_param("Y",target_pose.pose.position.y)
        rospy.set_param("Z",target_pose.pose.position.z)
        rospy.set_param("yaw",yaw)
        rospy.set_param("Vel",target_position[6])

    def get_trans_mat_world2local(self, drone_pose, rotation):
        im_position = [drone_pose.x,drone_pose.y, drone_pose.z]
        im_position = np.array(im_position).reshape((3, 1))
        extrinsic_mat = np.hstack((rotation, - rotation @ im_position))
        norm = np.array([0,0,0,1]).reshape((1,4))
        project_mat = np.vstack((extrinsic_mat, norm))
        return project_mat
    
    def go_back(self):
        rospy.set_param("back",True)

    def quaternion_to_euler(self, quaternion):
        roll = math.atan2(2 * (quaternion[3] * quaternion[0] + quaternion[1] * quaternion[2]), 1 - 2 * (quaternion[0]**2 + quaternion[1]**2))
        pitch = math.asin(2 * (quaternion[3] * quaternion[1] - quaternion[2] * quaternion[0]))
        yaw = math.atan2(2 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]), 1 - 2 * (quaternion[1]**2 + quaternion[2]**2))
        print("yaw:")
        print(yaw)
        return roll, pitch, yaw

    def euler_to_quaternion(self, roll, pitch, yaw):
        # 将角度转换为弧度
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        # 计算旋转矩阵的元素
        c1 = math.cos(roll/2)
        s1 = math.sin(roll/2)
        c2 = math.cos(pitch/2)
        s2 = math.sin(pitch/2)
        c3 = math.cos(yaw/2)
        s3 = math.sin(yaw/2)

        # 计算四元数的元素
        q0 = c1*c2*c3 + s1*s2*s3
        q1 = s1*c2*c3 - c1*s2*s3
        q2 = c1*s2*c3 + s1*c2*s3
        q3 = c1*c2*s3 - s1*s2*c3

        return [q0, q1, q2, q3]

    