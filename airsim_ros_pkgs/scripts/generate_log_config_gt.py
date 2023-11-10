"""
Generate log and write data in NuScenes frame
"""
import airsim
import time
import sys
import os
import rospy
from datetime import datetime
from tqdm import tqdm
import copy
import cv2
import numpy as np
import glob
import uuid
import json
from generate_path import generate_single_drone_path, get_single_drone_path, get_group_drone_path, get_drone_formation
from generate_path import five_drone_paths_town5, five_drone_paths_town3, group_path_town6, group_path_town3, group_path_town4, group_path_town5
from airsim_control import confirm_and_initiate_drones, reset_drones

from data_utils import generate_unique_token, generate_log_json, add_map_json
from data_utils import create_scene_json, create_sample_json, create_calibrated_sensor_json_camera, create_sample_data_json_camera, create_ego_pose_json, create_instance_json, create_environment_json, create_sample_annotation_json, create_sample_data_json_lidar
from data_utils import write_list_into_json
rospy.init_node('generate_log_gt')
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

sensor_tokens = {"Drone0": "50nb0v30da0lj13fp936v861w5ykrva3",
                 "Drone1": "8950zzt5fq4gr3g2ip9w4s04s1kevv36",
                 "Drone2": "6oqlp8qf1070udfkr275i7q76g61xi1h",
                 "Drone3": "q0g4k034n3vsa7742pvole8c0by52n49",
                 "Drone4": "b5t20629x6v93e27c9d2ko8u7g55w724"}

camera_sensor_tokens = [['eaf85517abe34feb8eb7f389255b9fff', 'd727296245144251a8b2498aee137314',
                         '4ed7b5598fdd4841858946746aa6af62', 'b2bbf1764cc2477ca3da6a54ed5f62c5',
                         '32fc361666f448adb125b05e7190a935'],
                        ['412c85b64b6a4285905cdca7990ee3dc', '4fb60e0f923742e484024e41e12a4607',
                         '0530913bebb54c5bb0226eb3845febf0', '915e07cb0332412f9d5fff0ae6a238e5',
                         '7078232a4d3b4083bd062605346f9368'],
                        ['0e748f2525074c1b89f344f2859b33b7', '5c05b5f82a6b42498d6fea6c10f40ede',
                         '4c26e08daca34d7cb26c7caeeeb14635', 'd450659d132e46949cbca9e9d0df4e9e',
                         'b6536056305648349ec7dd402bcfdb6a'],
                        ['927cbaf0e8b247f0a6ccb3c7d80a2ed9', '76f9e919aa12474eb4039d8dcf9cddd2',
                         'd5964048d9c64d108430552cead69305', 'bf036760422b4f2dbaff2522043c1a6d',
                         '5d03bdbc4bc44caaa77ca4658953c5f6'],
                        ['210c0c26186046ff8de6c68b12ee04a4', 'a057f03bf8eb4c7aaec977901de08449',
                         '26bbc81eb7d742b8b4ad1ee6d6da764b', 'cc744bf04b094051b2e71c5d0cb2c9cd',
                         'a0e1a02d05a3429e9a03bb0d7cab1d07']]

camera_intrinsic = [[1332.8958740234375, 0.0, 320.0],
                    [0.0, 1332.8958740234375, 240.0],
                    [0.0, 0.0, 1.0]]


class Vehicle(object):
    def __init__(self, vehicle_):
        self.id = vehicle_.id
        self.type = vehicle_.type_id
        self.box = vehicle_.bounding_box
        self.transform = vehicle_.get_transform()
        self.attributes = vehicle_.attributes


class Walker(object):
    def __init__(self, walker_):
        self.id = walker_.id
        self.type = walker_.type_id
        self.box = walker_.bounding_box
        self.transform = walker_.get_transform()
        self.attributes = walker_.attributes

def generate_unique_token():
    """
    return a 32 bit unique token
    """
    return uuid.uuid4().hex


def confirm_carla_world():
    try:
        carla_client = carla.Client('127.0.0.1', 2000)
        carla_client.set_timeout(10.0)
    except:
        print("Carla connection failed...")
        sys.exit(1)

    carla_world = carla_client.get_world()
    vehicles = carla_world.get_actors().filter('vehicle.*')
    return len(vehicles)


def generate_category_json(data_root):
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_world = carla_client.get_world()
    vehicles = carla_world.get_actors().filter('vehicle.*')
    walkers = carla_world.get_actors().filter('walker.pedestrian.*')
    vehicles_types = [vehicle.type_id for vehicle in vehicles]
    walkers_types = [walker.type_id for walker in walkers]
    types = vehicles_types + walkers_types
    types = set(types)

    del carla_world
    del carla_client

    filename = os.path.join(data_root, "category.json")

    if os.path.exists(filename):
        with open(filename) as f_obj:
            category = json.load(f_obj)
    else:
        category = []

    current_types = set()
    for cc in category:
        current_types.add(cc["name"])

    for type_ in types:
        if type_ not in current_types:
            type_attr = dict()
            type_attr["token"] = generate_unique_token()
            type_attr["name"] = type_
            type_attr["description"] = type_
            category.append(type_attr)

    with open(filename, "w") as f_obj:
        json.dump(category, f_obj, indent=1)


def generate_log_json(data_root,
                      vehicle_num,
                      location="town3"):
    log = dict()
    log["token"] = generate_unique_token()
    log["logfile"] = ""
    log["vehicle"] = str(vehicle_num)
    log["date_captured"] = datetime.now().strftime("%Y-%m-%d")
    log["location"] = location

    filename = os.path.join(data_root, "log.json")
    if os.path.exists(filename):
        with open(filename) as f_obj:
            log_list = json.load(f_obj)
    else:
        log_list = []

    log_list.append(log)
    with open(filename, "w") as f_obj:
        json.dump(log_list, f_obj, indent=1)

    return log["token"]

def get_environment_data():
    try:
        carla_client = carla.Client('127.0.0.1', 2000)
        carla_client.set_timeout(10.0)
    except:
        print("Carla connection failed...")
        sys.exit(1)
    carla_world = carla_client.get_world()
    Buildings = carla_world.get_environment_objects(carla.CityObjectLabel.Buildings)
    Poles = carla_world.get_environment_objects(carla.CityObjectLabel.Poles)
    Sidewalks = carla_world.get_environment_objects(carla.CityObjectLabel.Sidewalks)

    pass

def add_map_json(data_root,
                 log_token="",
                 map_pic='Town05.jpg'):
    filename = os.path.join(data_root, "map.json")
    if os.path.exists(filename):
        with open(filename) as f_obj:
            map_list = json.load(f_obj)
            map_list[0]["log_tokens"].append(log_token)
    else:
        map_list = []
        map_attr = dict()
        map_attr["category"] = "semantic_prior"
        map_attr["token"] = generate_unique_token()
        map_attr["filename"] = "maps/{}".format(map_pic)
        map_attr["log_tokens"] = [log_token]
        map_list.append(map_attr)

    with open(filename, "w") as f_obj:
        json.dump(map_list, f_obj, indent=1)


def get_airsim_lidar_info(client,
                          drone_list,
                          data_root):
    lidar_pose = []
    lidar_filename = []
    time_ = None
    for drone in drone_list:
        lidarData = client.getLidarData(vehicle_name=drone)
        folder_dir = os.path.join(data_root, "../sweeps")
        folder_dir = os.path.join(folder_dir, "LIDAR_TOP_id_%s" % drone[-1])
        if not os.path.exists(folder_dir):
            os.mkdir(folder_dir)

        points = np.array(lidarData.point_cloud, dtype=np.dtype('float32'))
        if points.shape[0] < 3:
            print("Cannot get correct lidar points, skip sample")
            return "", "", ""
        else:
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            # for i in range(points.shape[0]):
            #     points[i][2] = - points[i][2]
            points = np.concatenate([points, np.zeros((points.shape[0], 2))], axis=1)
            points.astype(np.float32)
            filename = os.path.join(folder_dir, str(lidarData.time_stamp))
            filename = os.path.normpath(filename + '.bin')
            points.tofile(filename)

            lidar_file = os.path.join("sweeps/LIDAR_TOP_id_%s" % drone[-1], str(lidarData.time_stamp) + '.bin')
            lidar_pose.append(lidarData.pose)
            lidar_filename.append(lidar_file)
            time_ = lidarData.time_stamp

    # if time_:
    #     print(datetime.fromtimestamp(time_ / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f'))

    return lidar_pose, lidar_filename, time_


def get_airsim_camera_info(client,
                           drone_list,
                           data_root,
                           sample_dict):
    camera_list = ["front_center_custom",
                   "front_left_custom",
                   "front_right_custom"]
    camera_name = {"front_center_custom": "FRONT",
                   "front_left_custom": "LEFT",
                   "front_right_custom": "RIGHT"}
    camera_pose = []
    camera_file = []
    for drone in drone_list:
        drone_camera_pose = []
        drone_camera_file = []
        # depth
        # responses = client.simGetImages([airsim.ImageRequest(camera_list[0], airsim.ImageType.Scene, False, True),
        #                                 airsim.ImageRequest(camera_list[0], airsim.ImageType.DepthPerspective, True),
        #                                 airsim.ImageRequest(camera_list[1], airsim.ImageType.Scene, False, True),
        #                                 airsim.ImageRequest(camera_list[1], airsim.ImageType.DepthPerspective, True),
        #                                 airsim.ImageRequest(camera_list[2], airsim.ImageType.Scene, False, True),
        #                                 airsim.ImageRequest(camera_list[2], airsim.ImageType.DepthPerspective, True),
        #                                 airsim.ImageRequest(camera_list[3], airsim.ImageType.Scene, False, True),
        #                                 airsim.ImageRequest(camera_list[3], airsim.ImageType.DepthPerspective, True),
        #                                 airsim.ImageRequest(camera_list[4], airsim.ImageType.Scene, False, True),
        #                                 airsim.ImageRequest(camera_list[4], airsim.ImageType.DepthPerspective, True)],
        #                                 vehicle_name=drone)
        # if responses[1].height == 0 or responses[1].width == 0:
        #     return "", ""

        for cam_id, camera in enumerate(camera_list):
            # responses = client.simGetImages([airsim.ImageRequest(camera, airsim.ImageType.Scene, False, True),
            #                                  airsim.ImageRequest(camera, airsim.ImageType.DepthPerspective, True)],
            #                                 vehicle_name=drone)
            # print(drone, camera)
            # 5 drone, get segmentation
            responses = client.simGetImages([airsim.ImageRequest(camera, airsim.ImageType.Scene, False, True),
                                             airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, False)],
                                            vehicle_name=drone)
            
            if responses[1].height == 0 or responses[1].width == 0:
                return "", ""

            # depth
            # response = responses[cam_id * 2]

            folder_dir = os.path.join(data_root, "../sweeps")
            folder_dir = os.path.join(folder_dir, "CAMERA_%s_id_%s" % (camera_name[camera], drone[5:]))
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)

            img_file = os.path.join(folder_dir, str(responses[0].time_stamp))
            airsim.write_file(os.path.normpath(img_file + '.png'), responses[0].image_data_uint8)
            # sample_dict.update({os.path.normpath(img_file + '.png'): copy.deepcopy(response.image_data_uint8)})

            # segmentation
            img1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape((responses[1].height, responses[1].width, 3))
            cv2.imwrite(os.path.normpath(img_file + 'seg.png'), img_rgb)  # write to png
            # sample_dict.update({os.path.normpath(img_file + 'seg.png'): copy.deepcopy(img_rgb)})

            # depth
            # depth_response = responses[cam_id * 2 + 1]
            # airsim.write_pfm(os.path.normpath(img_file + '.pfm'), airsim.get_pfm_array(depth_response))
            # sample_dict.update({os.path.normpath(img_file + '.pfm'): copy.deepcopy(airsim.get_pfm_array(depth_response))})

            drone_camera_pose.append({"position": responses[0].camera_position,
                                      "orientation": responses[0].camera_orientation})
            drone_camera_file.append(os.path.join("sweeps/CAMERA_%s_id_%s" % (camera_name[camera], drone[5:]),
                                                  str(responses[0].time_stamp) + '.png'))
        camera_pose.append(drone_camera_pose)
        camera_file.append(drone_camera_file)

    # parallel the saving process

    return camera_pose, camera_file

def simulate_scene_config(airsim_client,
                   drone_list,
                   drone_paths,
                   data_root="/home/fsh/file/airsim_data/v1.0-mini",
                   velocity=5.0,
                   sample_num=40,
                   get_lidar=True,
                   get_camera=False):
    all_instance_data = []
    all_instance_id = []
    all_vehicles_data = []
    all_vehicles_id = []
    all_walkers_data = []
    all_walkers_id = []
    all_lidar_pose = []
    all_lidar_file = []
    all_timestamp = []
    all_camera_pose = []
    all_camera_file = []

    all_ego_pose = []

    # go to the start point
    print("Moving to start point...")
    motor_state = []
    for i, drone in enumerate(drone_list):
        pa = []
        pa.append(drone_paths[i][0])
        motor_f = airsim_client.moveOnPathAsync(pa,
                                                velocity=4,
                                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                yaw_mode=airsim.YawMode(False, 0),
                                                lookahead=-1,
                                                adaptive_lookahead=0,
                                                vehicle_name=drone)
        motor_state.append(motor_f)
    for ff in motor_state:
        ff.join()

    # turn forward
    print("Turning forward...")
    motor_state = []
    for i, drone in enumerate(drone_list):
        pa = []
        pa.append(drone_paths[i][1])
        motor_f = airsim_client.moveOnPathAsync(pa,
                                                velocity=2,
                                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                yaw_mode=airsim.YawMode(False, 0),
                                                lookahead=-1,
                                                adaptive_lookahead=0,
                                                vehicle_name=drone)
        motor_state.append(motor_f)
    for ff in motor_state:
        ff.join()

    # start the simulation
    print("Start the scene path...")
    motor_state = []
    for i, drone in enumerate(drone_list):
        pa = []
        pa.append(drone_paths[i][2])
        motor_f = airsim_client.moveOnPathAsync(pa,
                                                velocity=velocity,
                                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                yaw_mode=airsim.YawMode(False, 0),
                                                lookahead=-1,
                                                adaptive_lookahead=0,
                                                vehicle_name=drone)
        motor_state.append(motor_f)

    count = 0
    count_fail = 0
    prev_time = 0

    sample_raw_data = dict()

    while count < sample_num and count_fail < 50:
        time_delay = 0.3
        try:
            carla_client = carla.Client('127.0.0.1', 2000)
            carla_client.set_timeout(20.0)
            carla_world = carla_client.get_world()
        except:
            print("Connet carla server failed, continue...")
            count_fail += 1
            continue
        t1 = time.time()
        # max call times 128
        vehicles = carla_world.get_actors().filter('vehicle.*')
        walkers = carla_world.get_actors().filter('walker.pedestrian.*')
        ai_controller = carla_world.get_actors().filter('controller.ai.walker')
        # print(len(ai_controller))
        # print(ai_controller[0].attributes)
        # print(ai_controller[0].parent)
        # pause the environment
        airsim_client.simPause(True)
        print("pause==========")
        t2 = time.time()
        vehicles_ = []
        vehicles_id = []
        walkers_ = []
        walkers_id = []
        env_data = []
        env_id = []
        for vehicle in vehicles:
            vehicles_.append(Vehicle(vehicle))
            vehicles_id.append(vehicle.id)
        for walker in walkers:
            walkers_.append(Walker(walker))
            walkers_id.append(walker.id)
        # for ai in ai_controller:
        #     env_data.append(ai.attributes)
        #     env_id.append(ai.id)
        # print(vehicles[3].get_transform())

        # print(t1)
        # print(t2)
        print("carla time delay(ms): ", (t2 - t1) * 1000)
        print("%d vehicles detected." % len(vehicles))
        print("%d walkers detected." % len(walkers))


        # get lidar info
        if get_lidar:
            lidar_pose, lidar_filename, lidar_timestamp \
                = get_airsim_lidar_info(airsim_client, drone_list, data_root)
            if lidar_pose != "":
                all_lidar_pose.append(lidar_pose)
                all_lidar_file.append(lidar_filename)
                # all_timestamp.append(lidar_timestamp)
                print("Sample %d done, %d vehicles found" % (count, len(vehicles_)))
            else:
                airsim_client.simPause(False)
                print("Generate sample failed, continue...")
                del carla_world
                del carla_client
                count_fail += 1
                continue
        # get camera info
        if get_camera:
            camera_pose, camera_file \
                = get_airsim_camera_info(airsim_client, drone_list, data_root, sample_raw_data)
            if camera_pose != "":
                all_camera_pose.append(camera_pose)
                all_camera_file.append(camera_file)
            else:
                print("Generate sample failed, continue...")
                airsim_client.simPause(False)
                del carla_world
                del carla_client
                count_fail += 1
                continue
        # add vehicle info
        all_vehicles_data.append(vehicles_)
        all_vehicles_id.append(vehicles_id)
        all_walkers_data.append(walkers_)
        all_walkers_id.append(walkers_id)
        all_instance_data.append(vehicles_+walkers_)
        all_instance_id.append(vehicles_id+walkers_id)
        # get timestamp and ego pose
        state = airsim_client.getMultirotorState(vehicle_name="Drone0")
        all_timestamp.append(state.timestamp)
        print(datetime.fromtimestamp(state.timestamp / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f'))
        cur_time = state.timestamp
        ege_pose = []
        for drone in drone_list:
            state = airsim_client.getMultirotorState(vehicle_name=drone)
            ege_pose.append({"position": state.kinematics_estimated.position,
                             "orientation": state.kinematics_estimated.orientation})
        all_ego_pose.append(ege_pose)
        # time.sleep(0.1)
        count += 1

        airsim_client.simPause(False)
        del carla_world
        del carla_client
        if (cur_time - prev_time) / 1e9 < 1.5:
            time_delay = 2 - (cur_time - prev_time) / 1e9
        print(count)
        if prev_time:
            print('The interval is {}'.format((cur_time - prev_time) / 1e9))
        prev_time = cur_time
        # print("airsim time delay: {}".format(time_delay))
        # time.sleep(time_delay)
        time.sleep(0.5)
    
    # # Write scene data to disk
    # for file_name, raw_data in tqdm(sample_raw_data.items()):
    #     if file_name[-4:] == '.pfm':
    #         airsim.write_pfm(file_name, raw_data)
    #     elif file_name[-7:] == 'seg.png':
    #         cv2.imwrite(file_name, raw_data)
    #     else:
    #         airsim.write_file(file_name, raw_data)

    # # release scene data in RAM
    # sample_raw_data.clear()


    for ff in motor_state:
        ff.join()

    print("Whole path time is {}".format((all_timestamp[-1] - all_timestamp[0]) / 1e9))
    print("simulate data done.")
    # all_instance_data = all_vehicles_data + all_walkers_data
    # all_instance_id = all_vehicles_id + all_walkers_id
    print("len of all_vehicles_data",len(all_vehicles_data[0]))
    print("len of all_walkers_data",len(all_walkers_data[0]))
    print("len of all_instance_data",len(all_instance_data[0]))
    return all_instance_data, all_instance_id, all_lidar_file, all_lidar_pose, \
           all_timestamp, all_camera_pose, all_camera_file, all_ego_pose



def fly_to_starting_point_one_drone(airsim_client, scene_num, drone_name, height):
    scene_num = scene_num % 18
    for path_i in range(scene_num):
        drone_path = generate_single_drone_path(path_i, height)
        result = airsim_client.moveOnPathAsync(drone_path,
                                               velocity=10,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone_name)
        result.join()


def move_to_starting_point_group(airsim_client, drone_list, height, drone_relative_position, route):
    # client.moveToZAsync(height, velocity=3, vehicle_name=drone)
    new_height = -70
    print("raising")
    motor_state = []
    for i, drone in enumerate(drone_list):
        path = drone_relative_position[i]
        # drone_path = [airsim.Vector3r(float(path[0]), float(path[1]), float(height + 10 + i * 5))]
        result = airsim_client.moveToZAsync(new_height, velocity=3, vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()

    print("Move to static group position...")
    motor_state = []
    for i, drone in enumerate(drone_list):
        path = drone_relative_position[i]
        drone_path = [airsim.Vector3r(float(path[0]), float(path[1]), float(new_height + 10 + i * 5))]
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=10,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()

    print("Move to starting point...")
    start_point = route[0][0]
    motor_state = []
    for i, drone in enumerate(drone_list):
        start = (drone_relative_position[i][0] + start_point[0], drone_relative_position[i][1] + start_point[1])
        drone_path = [airsim.Vector3r(float(start[0]), float(start[1]), float(new_height))]
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=10,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()

    print("descending...")
    motor_state = []
    for i, drone in enumerate(drone_list):
        path = drone_relative_position[i]
        # drone_path = [airsim.Vector3r(float(path[0]), float(path[1]), float(height + 10 + i * 5))]
        result = airsim_client.moveToZAsync(height, velocity=3, vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()

    print("Adjusting...")
    motor_state = []
    for i, drone in enumerate(drone_list):
        start = (drone_relative_position[i][0] + start_point[0], drone_relative_position[i][1] + start_point[1])
        drone_path = [airsim.Vector3r(float(start[0]), float(start[1]), float(height))]
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=5,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()


def move_to_starting_point(airsim_client, scene_num, drone_list, height, map_='town5', route=five_drone_paths_town5):
    # path_num = {"Drone0": 10, "Drone1": 6, "Drone2": 8, "Drone3": 10, "Drone4": 8}
    path_num = {drone: len(route[drone]) for drone in route}

    print("Ascending")
    new_height = -90
    motor_state = []
    for i, drone in enumerate(drone_list):
        result = airsim_client.moveToZAsync(new_height, velocity=5, vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()

    print("Move to start point")
    motor_state = []
    for i, drone in enumerate(drone_list):
        scene_i = scene_num % path_num[drone]
        # result = None
        drone_path = get_single_drone_path(scene_i, drone, new_height + i*2, map_=map_)
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=10,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()

    print("Adjusting")
    motor_state = []
    for i, drone in enumerate(drone_list):
        scene_i = scene_num % path_num[drone]
        # result = None
        drone_path = get_single_drone_path(scene_i, drone, new_height + i*2, map_=map_)
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=3,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()

    print("Descending")
    motor_state = []
    for i, drone in enumerate(drone_list):
        result = airsim_client.moveToZAsync(height, velocity=3, vehicle_name=drone)
        motor_state.append(result)
    for ff in motor_state:
        ff.join()


    # print("Move to start point")
    # motor_state = []
    # for drone in drone_list:
    #     drone_path = get_single_drone_path(0, drone, height, map_=map_)
    #     result = airsim_client.moveOnPathAsync([drone_path[0]],
    #                                            velocity=10,
    #                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                                            yaw_mode=airsim.YawMode(False, 0),
    #                                            lookahead=-1,
    #                                            adaptive_lookahead=0,
    #                                            vehicle_name=drone)
    #     motor_state.append(result)
    # for ff in motor_state:
    #     ff.join()

    # print("Move to original point.")
    # motor_state = []
    # for drone in drone_list:
    #     scene_i = scene_num % path_num[drone]
    #     result = None
    #     all_path = []
    #     for path_i in range(scene_i):
    #         drone_path = get_single_drone_path(path_i, drone, height, map_=map_)
    #         all_path = all_path + drone_path
    #     result = airsim_client.moveOnPathAsync(all_path,
    #                                            velocity=10,
    #                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                                            yaw_mode=airsim.YawMode(False, 0),
    #                                            lookahead=-1,
    #                                            adaptive_lookahead=0,
    #                                            vehicle_name=drone)
    #     if result:
    #         motor_state.append(result)
    # for ff in motor_state:
    #     ff.join()


def generate_single_drone_log():
    # generate log for single drone lidar only dataset
    data_root = "/home/fsh/file/airsim_detection/v1.0-mini"
    airsim_drone = "Drone0"
    scene_num = 4
    # log_num is a hyper parameter, it should be set manually
    log_num = 8
    world_position = [43.27, -5.84, -2.65]
    drone_position = [0.0, 0.0, 0.0]

    # check airsim and carla connection, initiate airsim drones
    z = -12
    airsim_client = confirm_and_initiate_drones([airsim_drone], z)
    airsim.wait_key("Start carla simulator and then press any key to start the path...")

    nbr_vehicles = confirm_carla_world()
    print("%d vehicles detected in the environment." % nbr_vehicles)

    print("Flying to starting point...")
    fly_to_starting_point_one_drone(airsim_client, log_num * scene_num, airsim_drone, z)

    print("Writing log info into json...")
    # visibility, attribute, sensor, calibrated_sensor json copied directly
    # category json
    generate_category_json(data_root)
    # log json
    log_token = generate_log_json(data_root, nbr_vehicles)
    # map json
    add_map_json(data_root, log_token)

    for scene_i in range(scene_num):
        scene_no = scene_i + scene_num * log_num
        # get the scene path
        scene_path = generate_single_drone_path(scene_no, z)
        airsim.wait_key("Press any key to start simulating data for scene_%d" % scene_no)
        vehicle_data, vehicle_id, lidar_file, lidar_pose, timestamps, _, _, ego_pose = simulate_scene_config(airsim_client,
                                                                                                      [airsim_drone],
                                                                                                      [scene_path],
                                                                                                      data_root,
                                                                                                      sample_num=30)

        # Transform data
        print("Transforming nuscenes frame data...")
        nbr_drones = len(airsim_drone)
        nbr_samples = len(timestamps)
        nbr_vehicles = len(vehicle_data[0])

        sample_tokens = [generate_unique_token() for _ in range(nbr_samples)]
        instance_tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
        # sample_data_tokens = [generate_unique_token() for _ in range(nbr_samples * nbr_drones)]
        sample_data_tokens = []
        for _ in range(nbr_samples):
            tokens = [generate_unique_token() for _ in range(nbr_drones)]
            sample_data_tokens.append(tokens)

        # vehicle decrease during the simulation, correspond vehicle id to sample_annotation_token
        sample_annotation_tokens = []
        for _ in range(nbr_samples):
            tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
            sample_annotation_tokens.append(tokens)

        scene_dict = create_scene_json(scene_no, log_token, nbr_samples, sample_tokens[0], sample_tokens[-1])
        scene_token = scene_dict["token"]
        sample_json_list = create_sample_json(scene_token, nbr_samples, sample_tokens, timestamps)
        sample_data_json_list = create_sample_data_json_lidar(sample_tokens, sample_data_tokens,
                                                              [airsim_drone], lidar_file, timestamps)
        ego_pose_json_list = create_ego_pose_json(sample_data_tokens, lidar_pose, timestamps,
                                                  world_position, [drone_position])
        instance_json_list = create_instance_json(nbr_vehicles, instance_tokens, sample_annotation_tokens,
                                                  data_root, vehicle_data[0], vehicle_id)
        sample_annotation_json_list = create_sample_annotation_json(nbr_samples, nbr_vehicles, sample_tokens,
                                                                    instance_tokens,
                                                                    sample_annotation_tokens, vehicle_data, vehicle_id)

        # Write data into json file
        print("Writing scene data into json files...")
        write_list_into_json(data_root, "scene.json", [scene_dict])
        write_list_into_json(data_root, "sample.json", sample_json_list)
        write_list_into_json(data_root, "sample_data.json", sample_data_json_list)
        write_list_into_json(data_root, "ego_pose.json", ego_pose_json_list)
        write_list_into_json(data_root, "instance.json", instance_json_list)
        write_list_into_json(data_root, "sample_annotation.json", sample_annotation_json_list)

        print("Scene %d done===============================================" % scene_no)

    # reset
    airsim.wait_key("Press any key to reset...")
    reset_drones([airsim_drone])


def generate_log_with_camera():
    # generate log for multi drones lidar & camera dataset
    # data_root = "/data2/fsh/file/simulation_data/co-UAV_town3_v3/v1.0-mini"
    data_root = "/data2/fsh/file/simulation_data/co-UAV_town3_v3_part2/v1.0-mini_1108_config"
    # airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3", "Drone4"]
    # airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3"]
    airsim_drones = ["Drone0"]
    log_num = 20
    scene_num = 1
    z = - 10  # height
    map_ = 'town3'
    map_pic = 'Town03.jpg'
    route = five_drone_paths_town3
    velocity = 1.6
    # route = five_drone_paths_town5
    # UE4 global origin
    # town 3
    world_position = [43.27, -5.84, -2.65]  # [44.07, -5.84, -4.15]
    # town 5
    # world_position = [-2.6, -2, -3.82]
    # drone_position = [[0.0, 0.0, 0.0],
    #                   [2.0, 0.0, 0.0],
    #                   [4.0, 0.0, 0.0],
    #                   [6.0, 0.0, 0.0],
    #                   [8.0, 0.0, 0.0]]
    drone_position = [[0.0, 0.0, 0.0]]

    # check airsim and carla connection, initiate airsim drones
    airsim_client = confirm_and_initiate_drones(airsim_drones, z)
    # airsim.wait_key("Press any key to fly to the starting point...")
    print("Flying to starting point...")
    move_to_starting_point(airsim_client, log_num * scene_num, airsim_drones, z, map_=map_, route=route)

    airsim.wait_key("Start carla simulator and then press any key to start the path...")
    nbr_vehicles = confirm_carla_world()

    # save environment object info
    # environment_object_data = get_environment_data()
    # instance_tokens = [generate_unique_token() for _ in range(nbr_instances)]
    # environment_object_json_list = create_environment_json

    print("%d vehicles detected in the environment." % nbr_vehicles)

    print("Writing log info into json...")
    # visibility, attribute, sensor json copied directly
    # category json
    generate_category_json(data_root)
    # log json
    log_token = generate_log_json(data_root, nbr_vehicles, location=map_)
    # map json
    add_map_json(data_root, log_token, map_pic=map_pic)

    for scene_i in range(scene_num):
        scene_no = scene_i + scene_num * log_num
        scene_path = []
        for drone in airsim_drones:
            # scene_path.append(get_single_drone_path(scene_no, drone, z, map_=map_))
            # static path
            static_point = get_single_drone_path(scene_no, drone, z, map_=map_)[0]
            scene_path.append([static_point, static_point, static_point])
        # scene_path = [[paths_1[i], paths[i], paths[i]] for i in range(5)]
        print(scene_path)
        # break

        airsim.wait_key("Press any key to start simulating data for scene_%d" % scene_no)

        # TODO(GJH): add walker data
        instance_data, instance_id, _, _, timestamps, camera_pose, camera_file, ego_pose = simulate_scene_config(
            airsim_client,
            airsim_drones,
            scene_path,
            data_root,
            velocity=velocity,
            sample_num=40,
            get_lidar=False,
            get_camera=True)


        # Transform data ===================
        print("Transforming nuscenes frame data...")
        nbr_cameras = 5
        nbr_drones = len(airsim_drones)
        nbr_samples = len(timestamps)
        # nbr_vehicles = len(vehicle_data[0])
        nbr_instances = len(instance_data[0])

        sample_tokens = [generate_unique_token() for _ in range(nbr_samples)]
        # instance_tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
        instance_tokens = [generate_unique_token() for _ in range(nbr_instances)]
        # sample_data_tokens = [generate_unique_token() for _ in range(nbr_samples * nbr_drones * nbr_cameras)]
        sample_data_tokens = []
        for _ in range(nbr_samples):
            sample_data_drone_tokens = []
            for _ in range(nbr_drones):
                tokens = [generate_unique_token() for _ in range(nbr_cameras)]
                sample_data_drone_tokens.append(tokens)
            sample_data_tokens.append(sample_data_drone_tokens)

        # ego_pose_tokens = [generate_unique_token() for _ in range(nbr_samples * nbr_drones)]
        ego_pose_tokens = []
        for _ in range(nbr_samples):
            tokens = [generate_unique_token() for _ in range(nbr_drones)]
            ego_pose_tokens.append(tokens)

        # vehicle decrease during the simulation, correspond vehicle id to sample_annotation_token
        sample_annotation_tokens = []
        for _ in range(nbr_samples):
            # tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
            tokens = [generate_unique_token() for _ in range(nbr_instances)]
            sample_annotation_tokens.append(tokens)
        scene_dict = create_scene_json(scene_no, log_token, nbr_samples, sample_tokens[0], sample_tokens[-1])
        scene_token = scene_dict["token"]
        sample_json_list = create_sample_json(scene_token, nbr_samples, sample_tokens, timestamps)
        calibrated_sensor_json_list = create_calibrated_sensor_json_camera(sample_data_tokens, camera_pose,
                                                                           world_position, drone_position)
        sample_data_json_list = create_sample_data_json_camera(sample_tokens, sample_data_tokens, ego_pose_tokens,
                                                               camera_file, timestamps)
        ego_pose_json_list = create_ego_pose_json(ego_pose_tokens, ego_pose, timestamps,
                                                  world_position, drone_position)
        instance_json_list = create_instance_json(nbr_instances, instance_tokens, sample_annotation_tokens,
                                                  data_root, instance_data[0], instance_id)
        sample_annotation_json_list = create_sample_annotation_json(nbr_samples, nbr_instances, sample_tokens,
                                                                    instance_tokens,
                                                                    sample_annotation_tokens, instance_data, instance_id)

        # Write data into json file
        print("Writing data into json files...")
        write_list_into_json(data_root, "scene.json", [scene_dict])
        write_list_into_json(data_root, "sample.json", sample_json_list)
        write_list_into_json(data_root, "calibrated_sensor.json", calibrated_sensor_json_list)
        write_list_into_json(data_root, "sample_data.json", sample_data_json_list)
        write_list_into_json(data_root, "ego_pose.json", ego_pose_json_list)
        write_list_into_json(data_root, "instance.json", instance_json_list)
        write_list_into_json(data_root, "sample_annotation.json", sample_annotation_json_list)

        # scene_i done
        print("Scene %d done===============================================" % scene_no)

    # Done & reset
    airsim.wait_key("Press any key to reset...")
    reset_drones(airsim_client, airsim_drones)


def generate_log_with_camera_static_multi():
    # generate log for multi drones lidar & camera dataset
    data_root = "/data2/fsh/file/simulation_data/static_test/6_drone/v1.0-mini"
    airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3", "Drone4"]
    airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3", "Drone4", "Drone5", "Drone6", "Drone7", "Drone8", "Drone9"]
    airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3", "Drone4", "Drone5", "Drone6", "Drone7", "Drone8", "Drone9", "Drone10", "Drone11"]

    log_num = 0
    scene_num = 1
    z = - 30  # height
    map_ = 'town5'
    map_pic = 'Town05.jpg'
    route = five_drone_paths_town5
    # UE4 global origin
    # town 3
    world_position = [43.27, -5.84, -2.65]
    # town 5
    # world_position = [-2.6, -2, -3.82]
    drone_position = [[0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0],
                      [4.0, 0.0, 0.0],
                      [6.0, 0.0, 0.0],
                      [8.0, 0.0, 0.0],
                      [10.0, 0.0, 0.0],
                      [12.0, 0.0, 0.0],
                      [14.0, 0.0, 0.0],
                      [16.0, 0.0, 0.0],
                      [18.0, 0.0, 0.0],
                      [20.0, 0.0, 0.0],
                      [22.0, 0.0, 0.0],
                      [24.0, 0.0, 0.0],
                      [26.0, 0.0, 0.0],
                      [28.0, 0.0, 0.0]]

    # check airsim and carla connection, initiate airsim drones
    airsim_client = confirm_and_initiate_drones(airsim_drones, z)
    # airsim.wait_key("Press any key to fly to the starting point...")
    # print("Flying to starting point...")
    # # move_to_starting_point(airsim_client, log_num * scene_num, airsim_drones, z, map_=map_, route=route)

    # paths_1 = [airsim.Vector3r(-40.0, -70.0, float(z)),
    #            airsim.Vector3r(-110.0, 0.0, float(z)),
    #            airsim.Vector3r(-40.0, 70.0, float(z)),
    #            airsim.Vector3r(30.0, 0.0, float(z)),
    #            airsim.Vector3r(30.0, 30.0, float(z))]

    # motor_state = []
    # for i, drone in enumerate(airsim_drones):
    #     result = airsim_client.moveOnPathAsync([paths_1[i]],
    #                                            velocity=5,
    #                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                                            yaw_mode=airsim.YawMode(False, 0),
    #                                            lookahead=-1,
    #                                            adaptive_lookahead=0,
    #                                            vehicle_name=drone)
    #     motor_state.append(result)
    # for ff in motor_state:
    #     ff.join()

    # paths = [airsim.Vector3r(-40.0, -60.0, float(z)),
    #          airsim.Vector3r(-100.0, 0.0, float(z)),
    #          airsim.Vector3r(-40.0, 60.0, float(z)),
    #          airsim.Vector3r(20.0, 0.0, float(z)),
    #          airsim.Vector3r(20.0, 30.0, float(z))]

    # state = []
    # for i, drone in enumerate(airsim_drones):
    #     pa = paths[i]
    #     motor_f = airsim_client.moveOnPathAsync([pa],
    #                                             velocity=2,
    #                                             drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                                             yaw_mode=airsim.YawMode(False, 0),
    #                                             lookahead=-1,
    #                                             adaptive_lookahead=0,
    #                                             vehicle_name=drone)
    #     state.append(motor_f)
    #
    # for ff in state:
    #     ff.join()

    airsim.wait_key("Start carla simulator and then press any key to start the path...")
    nbr_vehicles = confirm_carla_world()
    print("%d vehicles detected in the environment." % nbr_vehicles)

    print("Writing log info into json...")
    # visibility, attribute, sensor json copied directly
    # category json
    generate_category_json(data_root)
    # log json
    log_token = generate_log_json(data_root, nbr_vehicles, location=map_)
    # map json
    add_map_json(data_root, log_token, map_pic=map_pic)

    for scene_i in range(scene_num):
        scene_no = scene_i + scene_num * log_num
        # scene_path = []
        # for drone in airsim_drones:
        #     scene_path.append(get_single_drone_path(scene_no, drone, z, map_=map_))
        # scene_path = [[paths_1[i], paths[i], paths[i]] for i in range(5)]

        # stay still
        scene_path = [[airsim.Vector3r(0.0, 0.0, float(z)), airsim.Vector3r(0.0, 0.0, float(z)), airsim.Vector3r(0.0, 0.0, float(z))] for _ in range(len(airsim_drones))]

        airsim.wait_key("Press any key to start simulating data for scene_%d" % scene_no)

        vehicle_data, vehicle_id, _, _, timestamps, camera_pose, camera_file, ego_pose = simulate_scene_config(
            airsim_client,
            airsim_drones,
            scene_path,
            data_root,
            velocity=2,
            sample_num=10,
            get_lidar=False,
            get_camera=True)

        # Transform data ===================
        print("Transforming nuscenes frame data...")
        nbr_cameras = 5
        nbr_drones = len(airsim_drones)
        nbr_samples = len(timestamps)
        nbr_vehicles = len(vehicle_data[0])

        sample_tokens = [generate_unique_token() for _ in range(nbr_samples)]
        instance_tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
        # sample_data_tokens = [generate_unique_token() for _ in range(nbr_samples * nbr_drones * nbr_cameras)]
        sample_data_tokens = []
        for _ in range(nbr_samples):
            sample_data_drone_tokens = []
            for _ in range(nbr_drones):
                tokens = [generate_unique_token() for _ in range(nbr_cameras)]
                sample_data_drone_tokens.append(tokens)
            sample_data_tokens.append(sample_data_drone_tokens)

        # ego_pose_tokens = [generate_unique_token() for _ in range(nbr_samples * nbr_drones)]
        ego_pose_tokens = []
        for _ in range(nbr_samples):
            tokens = [generate_unique_token() for _ in range(nbr_drones)]
            ego_pose_tokens.append(tokens)

        # vehicle decrease during the simulation, correspond vehicle id to sample_annotation_token
        sample_annotation_tokens = []
        for _ in range(nbr_samples):
            tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
            sample_annotation_tokens.append(tokens)

        scene_dict = create_scene_json(scene_no, log_token, nbr_samples, sample_tokens[0], sample_tokens[-1])
        scene_token = scene_dict["token"]
        sample_json_list = create_sample_json(scene_token, nbr_samples, sample_tokens, timestamps)
        calibrated_sensor_json_list = create_calibrated_sensor_json_camera(sample_data_tokens, camera_pose,
                                                                           world_position, drone_position)
        sample_data_json_list = create_sample_data_json_camera(sample_tokens, sample_data_tokens, ego_pose_tokens,
                                                               camera_file, timestamps)
        ego_pose_json_list = create_ego_pose_json(ego_pose_tokens, ego_pose, timestamps,
                                                  world_position, drone_position)
        instance_json_list = create_instance_json(nbr_vehicles, instance_tokens, sample_annotation_tokens,
                                                  data_root, vehicle_data[0], vehicle_id)
        sample_annotation_json_list = create_sample_annotation_json(nbr_samples, nbr_vehicles, sample_tokens,
                                                                    instance_tokens,
                                                                    sample_annotation_tokens, vehicle_data, vehicle_id)

        # Write data into json file
        print("Writing data into json files...")
        write_list_into_json(data_root, "scene.json", [scene_dict])
        write_list_into_json(data_root, "sample.json", sample_json_list)
        write_list_into_json(data_root, "calibrated_sensor.json", calibrated_sensor_json_list)
        write_list_into_json(data_root, "sample_data.json", sample_data_json_list)
        write_list_into_json(data_root, "ego_pose.json", ego_pose_json_list)
        write_list_into_json(data_root, "instance.json", instance_json_list)
        write_list_into_json(data_root, "sample_annotation.json", sample_annotation_json_list)

        # scene_i done
        print("Scene %d done===============================================" % scene_no)

    # Done & reset
    airsim.wait_key("Press any key to reset...")
    reset_drones(airsim_client, airsim_drones)


def generate_log_with_camera_group():
    # generate log for multi drones lidar & camera dataset
    dataset_root = "/data2/fsh/file/simulation_data/airsim_camera_seg_town4_v2_40m"
    version = "v1.0-40m-group"  # "v1.0-height-formation"
    data_root = os.path.join(dataset_root, version)
    paths = group_path_town4
    path_num = 29
    k = 0
    total_scene = 31
    z = -43  # height
    map_ = 'town4'
    map_pic = 'Town04.jpg'

    world_position = [138.85, -171.74, -2.9]

    route = paths[path_num][k]
    route = [[(route[0][0] - world_position[0], route[0][1] - world_position[1]),
              (route[1][0] - world_position[0], route[1][1] - world_position[1])]]

    # route = [[(0, 60), (350, 60)]]  # test route
    scene_num = len(route)
    # UE4 global origin
    # town 3
    # world_position = [43.27, -5.84, -2.65]
    # town 4

    # town 5
    # world_position = [-2.6, -2, -3.82]
    # town 6
    # world_position = [0, 0, -5.64]
    airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3", "Drone4"]
    drone_position = [[0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0],
                      [4.0, 0.0, 0.0],
                      [6.0, 0.0, 0.0],
                      [8.0, 0.0, 0.0]]


    # generate random relative position
    # r = random.uniform(130, 140)
    # alpha = random.uniform(0.0, 72.0)
    # print(r)
    # print(alpha)
    # drone_relative = [[-np.cos((alpha + 72 * 0) * 180 / np.pi) * r, -np.sin((alpha + 72 * 0) * 180 / np.pi) * r],
    #                   [-np.cos((alpha + 72 * 4) * 180 / np.pi) * r, -np.sin((alpha + 72 * 4) * 180 / np.pi) * r],
    #                   [-np.cos((alpha + 72 * 1) * 180 / np.pi) * r, -np.sin((alpha + 72 * 1) * 180 / np.pi) * r],
    #                   [-np.cos((alpha + 72 * 3) * 180 / np.pi) * r, -np.sin((alpha + 72 * 3) * 180 / np.pi) * r],
    #                   [-np.cos((alpha + 72 * 2) * 180 / np.pi) * r, -np.sin((alpha + 72 * 2) * 180 / np.pi) * r]]
    # drone_relative = [[-150, 8],
    #                   [-39, 145],
    #                   [-54, -140],
    #                   [116, -94],
    #                   [125, 81]]

    drone_relative = get_drone_formation(route)

    # check airsim and carla connection, initiate airsim drones
    airsim_client = confirm_and_initiate_drones(airsim_drones, z)
    # airsim.wait_key("Press any key to fly to the starting point...")
    print("Flying to starting point...")
    move_to_starting_point_group(airsim_client, airsim_drones, z, drone_relative, route)

    # # test the path
    # airsim.wait_key("Press any key to start the path...")
    # scene_path = get_group_drone_path(0, airsim_drones, z, drone_relative, route)
    # motor_state = []
    # for i, drone in enumerate(airsim_drones):
    #     pa = []
    #     pa.append(scene_path[i][2])
    #     motor_f = airsim_client.moveOnPathAsync(pa,
    #                                             velocity=15,
    #                                             drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                                             yaw_mode=airsim.YawMode(False, 0),
    #                                             lookahead=-1,
    #                                             adaptive_lookahead=0,
    #                                             vehicle_name=drone)
    #     motor_state.append(motor_f)
    # for ff in motor_state:
    #     ff.join()
    # print('path {}: {} done'.format(path_num, k))
    # airsim.wait_key("Press any key to reset...")
    # reset_drones(airsim_client, airsim_drones)
    # return

    airsim.wait_key("Start carla simulator and then press any key to start the path...")
    nbr_vehicles = confirm_carla_world()
    print("%d vehicles detected in the environment." % nbr_vehicles)

    print("Writing category info into json...")
    # visibility, attribute, sensor json copied directly
    # category json
    generate_category_json(data_root)
    # # log json
    # log_token = generate_log_json(data_root, nbr_vehicles, location=map_)
    # # map json
    # add_map_json(data_root, log_token, map_pic=map_pic)

    for scene_i in range(scene_num):
        scene_no = scene_i + total_scene
        scene_path = get_group_drone_path(scene_i, airsim_drones, z, drone_relative, route)
        # scene_path = []
        # for drone in airsim_drones:
        #     scene_path.append(get_single_drone_path(scene_no, drone, z, map_=map_))

        airsim.wait_key("Press any key to start simulating data for scene_%d" % scene_no)

        vehicle_data, vehicle_id, _, _, timestamps, camera_pose, camera_file, ego_pose = simulate_scene(
            airsim_client,
            airsim_drones,
            scene_path,
            data_root,
            velocity=2.8,
            sample_num=25,
            get_lidar=False,
            get_camera=True)

        # Transform data ===================
        print("Transforming nuscenes frame data...")
        nbr_cameras = 5
        nbr_drones = len(airsim_drones)
        nbr_samples = len(timestamps)
        nbr_vehicles = len(vehicle_data[0])

        sample_tokens = [generate_unique_token() for _ in range(nbr_samples)]
        instance_tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
        # sample_data_tokens = [generate_unique_token() for _ in range(nbr_samples * nbr_drones * nbr_cameras)]
        sample_data_tokens = []
        for _ in range(nbr_samples):
            sample_data_drone_tokens = []
            for _ in range(nbr_drones):
                tokens = [generate_unique_token() for _ in range(nbr_cameras)]
                sample_data_drone_tokens.append(tokens)
            sample_data_tokens.append(sample_data_drone_tokens)

        # ego_pose_tokens = [generate_unique_token() for _ in range(nbr_samples * nbr_drones)]
        ego_pose_tokens = []
        for _ in range(nbr_samples):
            tokens = [generate_unique_token() for _ in range(nbr_drones)]
            ego_pose_tokens.append(tokens)

        # vehicle decrease during the simulation, correspond vehicle id to sample_annotation_token
        sample_annotation_tokens = []
        for _ in range(nbr_samples):
            tokens = [generate_unique_token() for _ in range(nbr_vehicles)]
            sample_annotation_tokens.append(tokens)

        print("Writing log info into json...")
        # log json
        log_token = generate_log_json(data_root, nbr_vehicles, location=map_)
        # map json
        add_map_json(data_root, log_token, map_pic=map_pic)

        scene_dict = create_scene_json(scene_no, log_token, nbr_samples, sample_tokens[0], sample_tokens[-1])
        scene_token = scene_dict["token"]
        sample_json_list = create_sample_json(scene_token, nbr_samples, sample_tokens, timestamps)
        calibrated_sensor_json_list = create_calibrated_sensor_json_camera(sample_data_tokens, camera_pose,
                                                                           world_position, drone_position)
        sample_data_json_list = create_sample_data_json_camera(sample_tokens, sample_data_tokens, ego_pose_tokens,
                                                               camera_file, timestamps)
        ego_pose_json_list = create_ego_pose_json(ego_pose_tokens, ego_pose, timestamps,
                                                  world_position, drone_position)
        instance_json_list = create_instance_json(nbr_vehicles, instance_tokens, sample_annotation_tokens,
                                                  data_root, vehicle_data[0], vehicle_id)
        sample_annotation_json_list = create_sample_annotation_json(nbr_samples, nbr_vehicles, sample_tokens,
                                                                    instance_tokens,
                                                                    sample_annotation_tokens, vehicle_data, vehicle_id)

        # Write data into json file
        print("Writing data into json files...")
        write_list_into_json(data_root, "scene.json", [scene_dict])
        write_list_into_json(data_root, "sample.json", sample_json_list)
        write_list_into_json(data_root, "calibrated_sensor.json", calibrated_sensor_json_list)
        write_list_into_json(data_root, "sample_data.json", sample_data_json_list)
        write_list_into_json(data_root, "ego_pose.json", ego_pose_json_list)
        write_list_into_json(data_root, "instance.json", instance_json_list)
        write_list_into_json(data_root, "sample_annotation.json", sample_annotation_json_list)

        # scene_i done
        print("Scene %d done===============================================" % scene_no)

    # Done & reset
    print("path done", path_num)
    airsim.wait_key("Press any key to reset...")
    reset_drones(airsim_client, airsim_drones)


def test_path():
    airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3", "Drone4"]
    world_position = [138.85, -171.74, -2.9]
    drone_position = [[0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0],
                      [4.0, 0.0, 0.0],
                      [6.0, 0.0, 0.0],
                      [8.0, 0.0, 0.0]]

    z = -40
    airsim_client = confirm_and_initiate_drones(["Drone0"], z)

    i = 1
    route = group_path_town3[i]
    drone_relative = get_drone_formation(route)

    for i in range(24, len(group_path_town4), 1):
        route = group_path_town4[i]
        relative_position = [0, 0]
        start = route[0][0]
        end = route[-1][1]
        drone_path = [airsim.Vector3r(float(start[0] + relative_position[0] - world_position[0]),
                                      float(start[1] + relative_position[1] - world_position[1]), float(z)),
                      airsim.Vector3r(float(end[0] + relative_position[0] - world_position[0]),
                                      float(end[1] + relative_position[1] - world_position[1]), float(z))]
        print(drone_path)
        print("move to start point")
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=20,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name="Drone0").join()
        print("adjusting")
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=5,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name="Drone0").join()
        print("start path", i)
        result = airsim_client.moveOnPathAsync(drone_path,
                                               velocity=20,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name="Drone0").join()

    # print("Flying to starting point...")
    # move_to_starting_point_group(airsim_client, airsim_drones, z, drone_relative, route)
    #
    # start = route[0][0]
    # end = route[-1][1]
    #
    # state = []
    # for i, drone in enumerate(airsim_drones):
    #     position = drone_relative[i]
    #     drone_path = [airsim.Vector3r(float(start[0] + position[0]), float(start[1] + position[1]), float(z)),
    #                   airsim.Vector3r(float(end[0] + position[0]), float(end[1] + position[1]), float(z))]
    #     result = airsim_client.moveOnPathAsync(drone_path,
    #                                            velocity=20,
    #                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                                            yaw_mode=airsim.YawMode(False, 0),
    #                                            lookahead=-1,
    #                                            adaptive_lookahead=0,
    #                                            vehicle_name=drone).join()
    #     print('drone {} done'.format(drone))


# for i in range(len(group_path_town3)):
#     print(i)
#     route = group_path_town6[i]
#     drone_relative = group_path_town3(route)
#     position = drone_relative[2]
#     start = route[0][0]
#     end = route[-1][1]
#     drone_path = [airsim.Vector3r(float(start[0] + position[0]), float(start[1] + position[1]), float(z)),
#                   airsim.Vector3r(float(end[0] + position[0]), float(end[1] + position[1]), float(z))]
#     result = airsim_client.moveOnPathAsync(drone_path,
#                                            velocity=20,
#                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
#                                            yaw_mode=airsim.YawMode(False, 0),
#                                            lookahead=-1,
#                                            adaptive_lookahead=0,
#                                            vehicle_name="Drone0").join()
#     print("{} route done".format(i))


if __name__ == '__main__':
    try:
        generate_log_with_camera()
    except KeyboardInterrupt:
        print('keyboard interrupt.')
    finally:
        print('\ndone.')
