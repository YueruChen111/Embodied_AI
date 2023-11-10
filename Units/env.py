#!/usr/bin/env python
# license removed for brevity
import sys
import os
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)
import rospy
import airsim
# import tf2_ros
import numpy as np
import math
import threading
from numpy.linalg import inv
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from mavros_msgs.srv import SetMode
from pyquaternion import Quaternion
import message_filters
from percetion_units import OWLVIT
from planning_units import path_planner
from tracking_units import predictor
# from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PIL_Image
import ros_numpy
import sensor_msgs
import torch
from functools import partial
import logging

FINISHED_TASK = True

class Object_Msg():
    def __init__(self):
        self.position = np.array([0,0,0,0,0,0])
        
    
class ENV_API:

    def __init__(self):
        rospy.init_node('ENV_API', anonymous=True)

        print("take off!")

        self.name_space = rospy.get_namespace().strip('/')
        # self.object_name = []
        self.image_dict = {}
        self.rate = rospy.Rate(10)
        self.lock = threading.Lock()
        self.VLM_detector =  OWLVIT()
        self.path_planner = path_planner()
        self.object_tracker = predictor()
        self.camera_intrinsic = np.array([[1332.8958740234375, 0.0, 320.0],
                                          [0.0, 1332.8958740234375, 240.0],
                                          [0.0, 0.0, 1.0]])
        self.Quaternion_camera= self.euler_to_quaternion(-90, 0, 0) # not sure the coordinate
        self.init_camera2local = np.array([[0,0,1],
                                             [1,0,0],
                                            [0,1,0]]) 
        self.camera_position = [0.50, 0, 0.10] # not sure the coordinate
        self.score_threshold = 0.1
        self.tracking_list = []
        self.task_finished = True
        print("Service has started")
        # if not self.set_offboard_mode():
        #     rospy.logerr("Failed to set OFFBOARD mode")
        #     return
        self.ego_position = Odometry()
        self.got_ego_pose = False
        logging.basicConfig(filename='/data2/fsh/repository/AirSim/ros/src/voxper/Units/prediction.log',level=logging.INFO)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print('ENV device: ', self.device)
        # take off
        # rospy.set_param("X",0)
        # rospy.set_param("Y",0)
        # rospy.set_param("Z",0)
        # rospy.set_param("yaw",0)
        self.ego_position_sub = rospy.Subscriber("/airsim_node/Drone0/odom_local_ned", Odometry, self.save_ego_position)
        self.image_sub = message_filters.Subscriber("/airsim_node/Drone0/front_center_custom/Scene", Image)
        self.ego_position_filter = message_filters.Subscriber("/airsim_node/Drone0/odom_local_ned", Odometry)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.ego_position_filter], 10, 0.1)
        self.ts.registerCallback(self.align_callback)
        
        self.prediction_filter = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.ego_position_filter], 10,0.1)
        self.prediction_filter.registerCallback(self.prediction_callback)
        self.acc_planning = False
        self.filter_dict , self.predict_dict = {}, {}
    def get_img_from_ros_image_msg(self, msg):
        """ Returns image as a numpy array. 
        Note: can be used with any topic of message type 'sensor_msgs/Image'
        """
        msg.__class__ = sensor_msgs.msg.Image
        return ros_numpy.numpify(msg)
    def run(self):
        # rospy.Timer(rospy.Duration(0.1), partial(self.prediction_callback, image_position_dict = self.image_dict, result_list = self.tracking_list))
        rospy.spin()
    def check_finish(self):
        global FINISHED_TASK
        state = self.path_planner.client.simGetGroundTruthKinematics(vehicle_name='Drone0')
        speed = math.sqrt(state.linear_velocity.x_val**2+state.linear_velocity.y_val**2+state.linear_velocity.z_val**2)
        x = state.position.x_val
        y = state.position.y_val
        z = state.position.z_val
        dis = math.sqrt((self.path_planner.dest_x-x)**2+(self.path_planner.dest_y-y)**2+(self.path_planner.dest_z-z)**2)
        if(speed<0.01)and(dis<0.5):
            self.task_finished = True
            FINISHED_TASK = True
    def save_ego_position(self, ego_position):
        self.ego_position = ego_position
        self.ego_position.pose.pose.position.z = - (self.ego_position.pose.pose.position.z)
    def align_callback(self, image, odometry):
        self.lock.acquire()
        self.image_dict.update({'image':image, 'ego_position':odometry})
        self.lock.release()
        self.got_ego_pose = True
    def prediction_callback(self, image_msg, position_msg):
        image = image_msg
        position = position_msg
        time_stamp = image.header.stamp.to_sec()
        numpy_image = self.get_img_from_ros_image_msg(image)
        xyxys, ids, confs, clss = self.object_tracker.tracking(numpy_image)
        print('ids: ',ids)
        if len(xyxys) == 0:
            return
        transformed_xyxys = self.object_tracker.transform_to_global(xyxys, position)
        self.lock.acquire()
        self.filter_dict , self.predict_dict = self.object_tracker.prediction(transformed_xyxys, ids, confs, clss, time_stamp)
        filter_dict = self.filter_dict
        predict_dict = self.predict_dict
        self.lock.release()
        print('filter_dict: ',filter_dict)
        print('predict_dict: ',predict_dict)
        filter_paths = []
        predict_paths = []
        for i in filter_dict.keys():
            path_points = []
            for xy in filter_dict[i]:
                point = airsim.Vector3r()
                point.x_val = xy[0]
                point.y_val = xy[1]
                point.z_val = 0.5
                path_points.append(point)
            filter_paths.append(path_points)
        for i in predict_dict.keys():
            path_points = []
            for xy in predict_dict[i]:
                point = airsim.Vector3r()
                point.x_val = xy[0]
                point.y_val = xy[1]
                point.z_val = 0.5
                path_points.append(point)
            predict_paths.append(path_points)
        for path in predict_paths:
            self.path_planner.plot_path(path,color = [0, 1, 0 ,1])
        for path in filter_paths:
            self.path_planner.plot_path(path)
    
    def detect(self,text_query):
        self.lock.acquire()
        image = self.image_dict['image']
        ego_position = self.image_dict['ego_position']
        self.lock.release()
        numpy_image = self.get_img_from_ros_image_msg(image)
        input_image = PIL_Image.fromarray(np.uint8(numpy_image)).convert("RGB")
        input_text_query = [[text_query]]
        outputs = self.VLM_detector.detection(input_image,input_text_query)
        target_sizes = torch.Tensor([img.size[::-1] for img in [input_image]]).to(self.device)
        results = self.VLM_detector.processor.post_process(outputs=outputs, target_sizes=target_sizes)
        object_list = []
        for i in range(len([input_image])):
            text = input_text_query[i] 
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= self.score_threshold:
                    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                    x0, y0, x1, y1 = box
                    cx = (x0 + x1)/2
                    cy = (y0 + y1)/2
                    local_position = self.pixel_to_local_position([cx,cy],ego_position)
                    object_msg = Object_Msg()
                    object_msg.position = local_position
                    object_list.append(object_msg)
        return object_list
    def pixel_to_local_position(self, pixel_position, ego_position):
        Quaternion_filter = self.Quaternion_camera
        camera2local = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
        print('camera2local: ', camera2local.rotation_matrix)
        camera2local = self.init_camera2local @ camera2local.rotation_matrix
        local2camera_rotation = inv(camera2local)
        print(print('camera2local 2: ', camera2local))
        print('local2camera_rotation: ',local2camera_rotation)
        camera_to_local_trans_mat = inv(self.get_trans_mat_local2camera(self.camera_position, local2camera_rotation))
        object_position_Camera = inv(self.camera_intrinsic) @ np.array([pixel_position[0],pixel_position[1],1]).reshape((3, 1)) * (ego_position.pose.pose.position.z + 2.0)
        print('origin z: ',ego_position.pose.pose.position.z)
        print('camera_to_local_trans_mat: ',camera_to_local_trans_mat)
        print('object_position_Camera: ',object_position_Camera)
        local_position =  camera_to_local_trans_mat @ np.append(object_position_Camera,1)
        local_position[2] = -local_position[2]
        local_position =  np.append(local_position[:3] ,np.array([0.0,0.0,0.0]))
        print('local_position: ',local_position)
        return local_position
    def get_ego_pos(self):
        return [0,0,0,0,0,0,0]

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
        local_to_world_trans_mat = inv(self.get_trans_mat_world2local(position, world2local_rotation))
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
        rospy.set_param("yaw",float(yaw))
        rospy.set_param("Vel",float(3))

    def execute(self, target_position, obstacles, velocity):
        global FINISHED_TASK
        print('target pose: ',target_position)
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        self.lock.acquire()
        orientation = self.ego_position.pose.pose.orientation
        planning_time_stamp = self.ego_position.header.stamp.to_sec()
        position = self.ego_position.pose.pose.position
        predict_dict = self.self.predict_dict
        self.lock.release()
        roll, pitch, yaw = self.quaternion_to_euler([orientation.x,orientation.y,orientation.z,orientation.w])
        # print('roll: ',roll,' pitch: ',pitch,'yaw: ',yaw)
        #print('origin yaw: ',yaw)
        yaw += target_position[3]
        #print('changed yaw: ',yaw)
        Quaternion_filter = self.euler_to_quaternion(0, 0, yaw)
        # print('after change: ',Quaternion_filter)
        local2world = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
        world2local_rotation = inv(local2world.rotation_matrix)
        local_to_world_trans_mat = inv(self.get_trans_mat_world2local(position, world2local_rotation))
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
        target_yaw+= target_position[3]
        start_point = [position.x, position.y, -position.z]
        target_point = [target_pose.pose.position.x,  target_pose.pose.position.y, - target_pose.pose.position.z]
        fly_velocity = velocity
        attention_obstacles = obstacles
        print('path start point(global): ',start_point)
        print('path target point(global): ',target_point)
        path_points, _, np_points = self.path_planner.generate_path(start_point ,target_point ,attention_obstacles, fly_velocity)
        print('path_points: ',path_points)
        self.value_function(np_points, predict_dict,self.object_tracker.predict_dt,velocity,planning_time_stamp)
        if self.acc_planning:
            self.path_planner.move_by_path_3d(path_points)
        else:
            self.path_planner.move_on_path(path_points, fly_velocity)
        self.check_finish = False
        FINISHED_TASK = False

    def get_trans_mat_world2local(self, drone_pose, rotation):
        im_position = [drone_pose.x,drone_pose.y, drone_pose.z]
        im_position = np.array(im_position).reshape((3, 1))
        extrinsic_mat = np.hstack((rotation, - rotation @ im_position))
        norm = np.array([0,0,0,1]).reshape((1,4))
        project_mat = np.vstack((extrinsic_mat, norm))
        return project_mat
    def get_trans_mat_local2camera(self, camera_pose, rotation):
        im_position = camera_pose
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

    def value_function (self, ego_path, predict_dict, dt, velocity,time_stamp):
        prediction_dict = predict_dict
        ego_trajectory = self.object_tracker.ego_trajectory_predictor(ego_path, dt, velocity, perfect_control = True)
        for id in prediction_dict.keys():
            object_unit = prediction_dict[id]
            begin_index = self.find_closest_value_index(object_unit['time_stamp'],time_stamp)
            print('id:', id ,' begin_index: ',begin_index)
            
            
    def find_closest_value_index(self, sorted_list, target_value):
        closest_index = 0
        min_diff = abs(target_value - sorted_list[closest_index])

        for i in range(1, len(sorted_list)):
            diff = abs(target_value - sorted_list[i])
            if diff < min_diff:
                closest_index = i
                min_diff = diff
            else:
                break

        return closest_index

            
        
        