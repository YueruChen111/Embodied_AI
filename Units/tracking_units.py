import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT, StrongSORT
from ultralytics import YOLO
from boxmot.utils import ROOT, WEIGHTS
from filterpy.kalman import KalmanFilter 
import math
from pyquaternion import Quaternion
from numpy.linalg import inv
import logging
    # Load a model
class predictor():
    def __init__(self):
        self.yolo = YOLO('/data2/fsh/repository/AirSim/ros/src/voxper/Units/weights/yolov8n.pt')
        self.tracker = StrongSORT(
            model_weights=Path('/data2/fsh/repository/AirSim/ros/src/voxper/Units/weights/osnet_x0_25_msmt17.pt'), # which ReID model to use
            device='cuda:0',
            fp16=True,
        )
        self.frame_number = 0
        self.last_time_stamp = None
        self.color = (0, 0, 255) 
        self.thickness = 2
        self.fontscale = 0.5
        self.result_dict = {}
        self.camera_intrinsic = np.array([[1332.8958740234375, 0.0, 320.0],
                                    [0.0, 1332.8958740234375, 240.0],
                                    [0.0, 0.0, 1.0]])
        self.Quaternion_camera= self.euler_to_quaternion(-90, 0, 0) # not sure the coordinate
        self.init_camera2local = np.array([[0,0,1],
                                             [1,0,0],
                                            [0,1,0]]) 
        self.camera_position = [0.50, 0, 0.10] # not sure the coordinate
        self.predict_dt = 0.2
        self.predict_time = 30
        logging.basicConfig(filename='/data2/fsh/repository/AirSim/ros/src/voxper/Units/prediction.log',level=logging.INFO)
    def tracking(self, image):
        img_w,img_h,_ = image.shape
        input_iamge = image
        dets = self.yolo.predict(source = input_iamge, save=False, imgsz=[640], classes = list(range(79)), conf = 0.5)
        for det in dets:
            boxes = det.boxes.xyxy
            confs = det.boxes.conf
            cls = det.boxes.cls
            if len(cls) !=0:
                print('cls: ',cls)
        # 将PyTorch张量转换为NumPy数组
            boxes_np = boxes.cpu().numpy()
            confs_np = confs.cpu().numpy()
            cls_np = cls.cpu().numpy()

        # 将boxes、confs和cls堆叠成一个数组
            detection_results = np.column_stack((boxes_np, confs_np, cls_np))
        
        self.frame_number = self.frame_number +1

        tracks = self.tracker.update(detection_results, input_iamge)
         # --> (x, y, x, y, id, conf, cls, ind)
        if tracks.shape[0] != 0:
            xyxys = tracks[:, 0:4].astype('int') # float64 to int
            ids = tracks[:, 4].astype('int') # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int') # float64 to int
        # print bboxes with their associated id, cls and conf
        else:
            return [],[],[],[]
        if tracks.shape[0] != 0:
            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                input_iamge = cv2.rectangle(
                    input_iamge,
                    (xyxy[0], xyxy[1]),
                    (xyxy[2], xyxy[3]),
                    self.color,
                    self.thickness
                )
                cv2.putText(
                    input_iamge,
                    f'{id} ',                #f'id: {id}, conf: {conf}, c: {cls}',
                    (xyxy[0], xyxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.fontscale,
                    self.color,
                    self.thickness
                )
        # show image with bboxes, ids, classes and confidences
        origin_size = (img_w, img_h)
        input_iamge  = cv2.resize(input_iamge, origin_size)
        # cv2.imshow('frame', input_iamge)
        # cv2.waitKey(3)
        return xyxys, ids, confs, clss
    
    def prediction(self, tranformed_list, ids, confs, clss, time_stamp):
        H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
        R = np.eye(2) * 1
        filter_dict = {}
        predict_dict = {}
        F_predict = np.array([[1, 0, self.predict_dt, 0],
                                [0, 1, 0, self.predict_dt],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        for i ,id in enumerate(ids):
            if id in self.result_dict.keys():
                self.result_dict[id]['position'].append(tranformed_list[i])
                self.result_dict[id]['time_stamp'].append(time_stamp)
            else:
                self.result_dict.update({id:{'position':[tranformed_list[i]], 'time_stamp':[time_stamp]}})
        for id in self.result_dict.copy().keys():
            if time_stamp - self.result_dict[id]['time_stamp'][-1] > 3.0:
                print('del the old id')
                del self.result_dict[id]
                continue
            if len(self.result_dict[id]['position']) < 4:
                continue
            
            print('result_dict[id]: ',id, 'contant: ',self.result_dict[id])
            for index, position in enumerate(self.result_dict[id]['position']):
                if index == 0:
                    init_x = position[0]
                    init_y = position[1]
                    init_time = self.result_dict[id]['time_stamp'][index]
                    continue
                elif index == 1:
                    delta_time = self.result_dict[id]['time_stamp'][index] - init_time
                    X =  np.array([position[0], position[1], (position[0] - init_x)/delta_time, (position[1] - init_y)/delta_time])
                    P = np.eye(len(X)) * 1
                    last_time_stamp = self.result_dict[id]['time_stamp'][index]
                    filter_dict.update({id:[X]})
                    continue
                dt = self.result_dict[id]['time_stamp'][index] - last_time_stamp
                print('id: ',id, 'dt: ',dt)
                F = np.array([[1, 0, dt, 0],
                                [0, 1, 0, dt],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
                X, P = self.extended_kalman_filter (X, P, F, H, R, np.array([position[0], position[1]]))
                filter_dict[id].append(X)
                last_time_stamp = self.result_dict[id]['time_stamp'][index]
                if index == 0:
                    predict_dict.update({id:{'position':[X],'time_stamp': [last_time_stamp]}})
                else: 
                    predict_dict[id]['position'].append(X)
                    predict_dict[id]['time_stamp'].append(last_time_stamp)

            for j in range(int(self.predict_time/self.predict_dt)):
                X, P = self.extended_kalman_filter (X, P, F_predict, H, R, np.array([0, 0]),only_predict = True)
                predict_dict[id]['position'].append(X)
                predict_dict[id]['time_stamp'].append((j + 1) * self.predict_dt + last_time_stamp)
        
        return filter_dict, predict_dict

    def transform_to_global (self, xyxys, ego_location):
        transformed_list = []
        for xyxy in xyxys:
            cx = (xyxy[0] + xyxy[2])/2
            cy = (xyxy[1] + xyxy[3])/2
            Quaternion_filter = self.Quaternion_camera
            camera2local = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
            camera2local = self.init_camera2local @ camera2local.rotation_matrix
            local2camera_rotation = inv(camera2local)
            camera_to_local_trans_mat = inv(self.get_trans_mat_local2camera(self.camera_position, local2camera_rotation))
            object_position_Camera = inv(self.camera_intrinsic) @ np.array([cx,cy,1]).reshape((3, 1)) * (ego_location.pose.pose.position.z + 2.0)
            local_position =  camera_to_local_trans_mat @ np.append(object_position_Camera,1)
            local_position[2] = - local_position[2]
        
            orientation = ego_location.pose.pose.orientation
            position = ego_location.pose.pose.position
            roll, pitch, yaw = self.quaternion_to_euler([orientation.x, orientation.y, orientation.z, orientation.w])
            Quaternion_filter = self.euler_to_quaternion(0, 0, yaw)
            local2world = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
            world2local_rotation = inv(local2world.rotation_matrix)
            local_to_world_trans_mat = inv(self.get_trans_mat_world2local(position, world2local_rotation))
            target_point_local = np.append(local_position[:3], 1) 
            target_pose_world = local_to_world_trans_mat @ target_point_local
            transformed_list.append(target_pose_world[:2])
        return transformed_list


    def quaternion_to_euler (self, quaternion):
        roll = math.atan2(2 * (quaternion[3] * quaternion[0] + quaternion[1] * quaternion[2]), 1 - 2 * (quaternion[0]**2 + quaternion[1]**2))
        pitch = math.asin(2 * (quaternion[3] * quaternion[1] - quaternion[2] * quaternion[0]))
        yaw = math.atan2(2 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]), 1 - 2 * (quaternion[1]**2 + quaternion[2]**2))
        print("yaw:")
        print(yaw)
        return roll, pitch, yaw

    def euler_to_quaternion (self, roll, pitch, yaw):
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
    
    def get_trans_mat_world2local (self, drone_pose, rotation):
        im_position = [drone_pose.x,drone_pose.y, drone_pose.z]
        im_position = np.array(im_position).reshape((3, 1))
        extrinsic_mat = np.hstack((rotation, - rotation @ im_position))
        norm = np.array([0,0,0,1]).reshape((1,4))
        project_mat = np.vstack((extrinsic_mat, norm))
        return project_mat
    def get_trans_mat_local2camera (self, camera_pose, rotation):
        im_position = camera_pose
        im_position = np.array(im_position).reshape((3, 1))
        extrinsic_mat = np.hstack((rotation, - rotation @ im_position))
        norm = np.array([0,0,0,1]).reshape((1,4))
        project_mat = np.vstack((extrinsic_mat, norm))
        return project_mat
    
    
    def extended_kalman_filter(self, x, P, F, H, R, y, only_predict = False):
        # 状态预测
        x_pred = F.dot(x)
        P_pred = F.dot(P).dot(F.T)
        if only_predict:
            return x_pred, P_pred
        # 观测预测
        y_pred = H.dot(x_pred)

        # 计算残差和协方差
        residual = y - y_pred
        S = H.dot(P_pred).dot(H.T) + R

        # 计算卡尔曼增益
        K = P_pred.dot(H.T).dot(np.linalg.inv(S))

        # 更新状态估计和协方差
        x = x_pred + K.dot(residual)
        P = (np.eye(len(x)) - K.dot(H)).dot(P_pred)

        return x, P
    
                # kf = KalmanFilter(dim_x = 4, dim_z = 2)
            # # 设置采样时间间隔，与前面 x_true=np.linspace 的采样间隔相等，即每隔0.1个单位采样一次
            # dt = 100 / size
            # # 设置初始状态为 x方向位置0，y方向位置0，x方向速度1，y方向速度1
            # kf.x = np.array([[0], [0], [1], [1]])
            # # 设置状态转移矩阵
            # kf.F = np.array([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]])
            # # 设置状态向量到测量值的转换矩阵
            # kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            # # 设置过程噪声的协方差矩阵 Q，该值参考了 《Bayesian Filtering and Smoothing》，Simo Sarkka著
            # # https://zhuanlan.zhihu.com/p/148046908
            # qc1, qc2 = 1, 1
            # kf.Q = np.array([[qc1 * t ** 3 / 3, 0, qc1 * t ** 2 / 2, 0],
            #                 [0, qc2 * t ** 3 / 3, 0, qc2 * t ** 2 / 2],
            #                 [qc1 * t ** 2 / 2, 0, qc1 * t, 0],
            #                 [0, qc2 * t ** 2 / 2, 0, qc2 * t]])
            # # 设置测量噪声的协方差矩阵 R，使用了前面人为制造噪声时的方差
            # kf.R = np.array([[sigma_x2, 0], [0, sigma_y2]])
            # # 设置先验误差的协方差矩阵 P，按默认的单位矩阵进行初始化
            # kf.P = np.eye(4)

            # # 记录估计过程
            # x_pred = []
            # y_pred = []
            # for i in range(size):
            #     kf.predict()
            #     kf.update(np.array([[x_mea[i]], [y_mea[i]]]))
            #     # 将预测结果保存
            #     x_pred.append(kf.x[0])
            #     y_pred.append(kf.x[1])

            # # 绘制结果
            # plt.plot(x_true, y_true, color='red', label="true value")
            # plt.plot(x_mea, y_mea, alpha=0.2, color='green', label='measurement value')
            # plt.plot(x_pred, y_pred, color='blue', label="estimate value")
            # plt.legend()
            # plt.show()
    def ego_trajectory_predictor(self,points,dt, velocity, perfect_control = True):
        if perfect_control:
            # 计算相邻点之间的距离
            fixed_distance = dt * velocity
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

            # 计算每个点之间的累积距离
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

            # 创建插值函数
            interpolator = interp1d(cumulative_distances, points, axis=0)

            # 计算重新采样后的点的累积距离
            resampled_distances = np.arange(0, cumulative_distances[-1], fixed_distance)

            # 使用插值函数进行插值
            resampled_points = interpolator(resampled_distances)

            return resampled_points