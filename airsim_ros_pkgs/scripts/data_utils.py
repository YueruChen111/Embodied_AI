import os
from datetime import datetime
import uuid
import json
import rospy
from utils import quaternionr2quaternion, vector3r2numpy, euler2quaternion


sensor_tokens = {"Drone0": "50nb0v30da0lj13fp936v861w5ykrva3",
                 "Drone1": "8950zzt5fq4gr3g2ip9w4s04s1kevv36",
                 "Drone2": "6oqlp8qf1070udfkr275i7q76g61xi1h",
                 "Drone3": "q0g4k034n3vsa7742pvole8c0by52n49",
                 "Drone4": "b5t20629x6v93e27c9d2ko8u7g55w724",
                 "Drone5": "4ac5b0c687584401811947e0141cb22d",
                 "Drone6": "35f3300114c94e4099711cb25c4d2b5b",
                 "Drone7": "dc099dbadb024d758607b105bd75bc89",
                 "Drone8": "c3f0ba72fdb749889e3cd6533cf5bbf9",
                 "Drone9": "1957fef54de445aa9b66380b79dca885",
                 "Drone10": "e67274fc983b4590b937a48474c8584c",
                 "Drone11": "581fbc05a26248138e1f1c9726e13860",
                 "Drone12": "6aa352b3bea94ce68d09ace16071a43f",
                 "Drone13": "dae4203090ce42359ed60c2f763f0d32",
                 "Drone14": "62fae3d5fd3041dba5bf84b34b51b39d"}



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
                         'a0e1a02d05a3429e9a03bb0d7cab1d07'],
                        ['6b7e04b70db248d89455262d51116da5', 'c9fe43fd0eaa419c8c686203497ae2d6',
                         'a8fb92aad8f34e09bcc8aa5a13c0ff4d', 'f2f557f5e95f4d95955e65985f915174',
                         '721e793b70074a6183a0a26cf8a777de'],
                        ['f8061eb13cb640c4abbeb2707dee391d', '54c3c91fc3a14555ad2a8b3d20c64133',
                         '4591d488acdb4849ba23da3ca08a2c5b', '12794c15d36544e9862f7f5616f87886',
                         'e48e5e6bee9a4245a6f91612a22fff8a'],
                        ['10dd4349b01044b48f0c195171cb6e33', '1fd0ca60fc874901ab94ed42b1531d04',
                         'c0db70fd950b45119f6a158bd8c4ec20', 'ff986a981efb4d1d981ce13d4cb91624',
                         'c0e1f83cba7949f8a07771359f6a9043'],
                        ['268a0021a51a453db467834170ec1bf2', '16555a39712f4e35bccbea6b7b50c5ee',
                         'ba52c4183642466ea641b4051cd7153b', '26c142d14c5c48e38ddf19b6c8898b9c',
                         '8117381e36474bcead6fafed6ac836de'],
                        ['4365b9bd4434408396bb0e690e6c67c7', 'ad1e28b6670d4455be2f2854e21c08bd',
                         '84c819e8d989468fa8135d6d86837278', 'e6046fa109dd444e964aad500dc9af2b',
                         'b2765a9823214e46adc16a57497b61a4'],
                        ['952f3278cf624dcc8a2cbed922d812f0', '745aa453f6834a26b24d1b47c7088f11',
                         'fff912c2c068460aa5d36716ca061a7c', '764d7322edf24d1ca511f95b83d05e62',
                         '93064ba278c2457da7a0726cae55ef8a'],
                        ['c111a82d1a904b00b118f62ca84f41b6', '95244435a2cb4842b81043321454cfb3',
                         '07ecc77cd5ae4ed580686ad8afa8cf1c', '440073f36a854440b054a62ac511a6f4',
                         '8f1f358940e94d2689e95605144dee85']]  # 12 drone


camera_intrinsic = [[400.0, 0.0, 400.0],
                    [0.0, 400.0, 225.0],
                    [0.0, 0.0, 1.0]]


def generate_unique_token():
    """
    return a 32 bit unique token
    """
    return uuid.uuid4().hex


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


def create_scene_json(scene_i, log_token, nbr_samples, first_sample_token, last_sample_token, map_='town5'):
    scene = dict()
    scene["token"] = generate_unique_token()
    scene["log_token"] = log_token
    scene["nbr_samples"] = nbr_samples
    scene["first_sample_token"] = first_sample_token
    scene["last_sample_token"] = last_sample_token
    scene["name"] = "scene_%d" % scene_i
    scene["description"] = "drone images of map {}".format(map_)
    return scene


def create_sample_json(scene_token, sample_num, sample_tokens, timestamps):
    sample_json_list = []
    for i in range(sample_num):
        sample = dict()
        sample["token"] = sample_tokens[i]
        sample["timestamp"] = timestamps[i]
        sample["prev"] = sample_tokens[i - 1] if i > 0 else ""
        sample["next"] = sample_tokens[i + 1] if i < sample_num - 1 else ""
        sample["scene_token"] = scene_token
        sample_json_list.append(sample)
    return sample_json_list


def create_sample_data_json_lidar(sample_tokens,
                                  sample_data_tokens,
                                  drone_list,
                                  lidar_file_list,
                                  timestamps):
    sample_num = len(sample_tokens)
    sample_data_list = []
    for i in range(sample_num):
        for j, drone in enumerate(drone_list):
            sample_data = dict()
            sample_data["token"] = sample_data_tokens[i][j]
            sample_data["sample_token"] = sample_tokens[i]
            sample_data["ego_pose_token"] = sample_data_tokens[i][j]
            sample_data["calibrated_sensor_token"] = sensor_tokens[drone]
            sample_data["timestamp"] = timestamps[i]
            sample_data["fileformat"] = ".bin"
            sample_data["is_key_frame"] = True
            sample_data["height"] = 0
            sample_data["width"] = 0
            sample_data["filename"] = lidar_file_list[i][j]
            sample_data["prev"] = sample_data_tokens[i - 1][j] if i > 0 else ""
            sample_data["next"] = sample_data_tokens[i + 1][j] if i < sample_num - 1 else ""
            sample_data_list.append(sample_data)

    return sample_data_list


def create_calibrated_sensor_json_camera(camera_calibrated_tokens,
                                         camera_pose,
                                         world_position,
                                         drone_position):
    sample_num = len(camera_calibrated_tokens)
    drone_num = len(camera_calibrated_tokens[0])
    camera_num = 5
    calibrated_sensor_list = []
    for i in range(sample_num):
        for j in range(drone_num):
            for k in range(camera_num):
                pose = camera_pose[i][j][k]
                translation = vector3r2numpy(pose["position"])
                translation = translation + world_position + drone_position[j]
                quaternionr = pose["orientation"]
                rotation = [quaternionr.w_val, quaternionr.x_val, quaternionr.y_val, quaternionr.z_val]
                calibrated_sensor = dict()
                calibrated_sensor["token"] = camera_calibrated_tokens[i][j][k]
                calibrated_sensor["sensor_token"] = camera_sensor_tokens[j][k]
                calibrated_sensor["translation"] = list(translation)
                calibrated_sensor["rotation"] = list(rotation)
                calibrated_sensor["camera_intrinsic"] = list(camera_intrinsic)
                calibrated_sensor_list.append(calibrated_sensor)
    return calibrated_sensor_list


def create_sample_data_json_camera(sample_tokens,
                                   sample_data_tokens,
                                   ego_pose_tokens,
                                   camera_file_list,
                                   timestamps):
    sample_num = len(sample_tokens)
    drone_num = len(ego_pose_tokens[0])
    camera_num = 5
    sample_data_list = []
    for i in range(sample_num):
        for j in range(drone_num):
            for k in range(camera_num):
                sample_data = dict()
                sample_data["token"] = sample_data_tokens[i][j][k]
                sample_data["sample_token"] = sample_tokens[i]
                sample_data["ego_pose_token"] = ego_pose_tokens[i][j]
                sample_data["calibrated_sensor_token"] = sample_data_tokens[i][j][k]
                sample_data["timestamp"] = timestamps[i]
                sample_data["fileformat"] = ".png"
                sample_data["is_key_frame"] = True
                sample_data["height"] = 900
                sample_data["width"] = 1600
                sample_data["filename"] = camera_file_list[i][j][k]
                sample_data["prev"] = sample_data_tokens[i - 1][j][k] if i > 0 else ""
                sample_data["next"] = sample_data_tokens[i + 1][j][k] if i < sample_num - 1 else ""
                sample_data_list.append(sample_data)
    return sample_data_list


def create_ego_pose_json(ego_pose_tokens, ego_poses, timestamps, world_position, drones_position):
    sample_num = len(ego_pose_tokens)
    drone_num = len(drones_position)
    ego_pose_list = []
    for i in range(sample_num):
        for j in range(drone_num):
            pose = ego_poses[i][j]
            translation = vector3r2numpy(pose["position"])
            translation = translation + world_position + drones_position[j]
            rotation = quaternionr2quaternion(pose["orientation"])
            ego_pose = dict()
            ego_pose["token"] = ego_pose_tokens[i][j]
            ego_pose["timestamp"] = timestamps[i]
            ego_pose["rotation"] = list(rotation)
            ego_pose["translation"] = list(translation)
            ego_pose_list.append(ego_pose)
    return ego_pose_list


def get_category(data_dir):
    with open(os.path.join(data_dir, "category.json")) as f_obj:
        category = json.load(f_obj)
    category_dict = dict()
    for cc in category:
        category_dict[cc["name"]] = cc["token"]
    return category_dict


def create_instance_json(vehicle_num,
                         instance_tokens,
                         sample_annotation_tokens,
                         data_dir,
                         vehicles_data,
                         vehicles_id):
    instance_list = []
    category = get_category(data_dir)
    for i in range(vehicle_num):
        # get nbr_annotation
        nbr_annotations = 0
        id_ = vehicles_id[0][i]
        for ids in vehicles_id:
            if id_ in ids:
                nbr_annotations += 1
            else:
                break

        instance = dict()
        instance["token"] = instance_tokens[i]
        instance["category_token"] = category[vehicles_data[i].type]
        instance["attributes"] = vehicles_data[i].attributes
        instance["nbr_annotations"] = nbr_annotations
        instance["first_annotation_token"] = sample_annotation_tokens[0][i]
        instance["last_annotation_token"] = sample_annotation_tokens[nbr_annotations - 1][i]
        instance_list.append(instance)
    return instance_list


def create_environment_json(vehicle_num,
                         instance_tokens,
                         sample_annotation_tokens,
                         data_dir,
                         vehicles_data,
                         vehicles_id):
    instance_list = []
    category = get_category(data_dir)
    for i in range(vehicle_num):
        # get nbr_annotation
        nbr_annotations = 0
        id_ = vehicles_id[0][i]
        for ids in vehicles_id:
            if id_ in ids:
                nbr_annotations += 1
            else:
                break

        instance = dict()
        instance["token"] = instance_tokens[i]
        instance["category_token"] = category[vehicles_data[i].type]
        instance["attributes"] = vehicles_data[i].attributes
        instance["nbr_annotations"] = nbr_annotations
        instance["first_annotation_token"] = sample_annotation_tokens[0][i]
        instance["last_annotation_token"] = sample_annotation_tokens[nbr_annotations - 1][i]
        instance_list.append(instance)
    return instance_list

def create_sample_annotation_json(sample_num,
                                  vehicle_num,
                                  sample_tokens,
                                  instance_tokens,
                                  sample_annotation_tokens,
                                  vehicles_data,
                                  vehicles_id):
    sample_annotation_list = []
    for j in range(vehicle_num):
        vehicle_id = vehicles_id[0][j]
        # get nbr_annotation
        nbr_annotations = 0
        for ids in vehicles_id:
            if vehicle_id in ids:
                nbr_annotations += 1
            else:
                break
        for i in range(sample_num):
            if vehicle_id in vehicles_id[i]:
                vehicle_idx = vehicles_id[i].index(vehicle_id)
            else:
                break
            vehicle = vehicles_data[i][vehicle_idx]
            yaw, pitch, roll = vehicle.transform.rotation.yaw, \
                               vehicle.transform.rotation.pitch, \
                               vehicle.transform.rotation.roll
            w, x, y, z = euler2quaternion(yaw, pitch, roll)
            sample_annotation = dict()
            sample_annotation["token"] = sample_annotation_tokens[i][j]
            sample_annotation["sample_token"] = sample_tokens[i]
            sample_annotation["instance_token"] = instance_tokens[j]
            sample_annotation["visibility_token"] = ""
            sample_annotation["attribute_tokens"] = ["cb5118da1ab342aa947717dc53544259"]
            sample_annotation["attributes"] = vehicle.attributes
            sample_annotation["translation"] = [vehicle.transform.location.x + vehicle.box.location.x,
                                                vehicle.transform.location.y + vehicle.box.location.y,
                                                vehicle.transform.location.z + vehicle.box.location.z]
            # not sure
            sample_annotation["size"] = [2 * vehicle.box.extent.y, 2 * vehicle.box.extent.x, 2 * vehicle.box.extent.z]
            sample_annotation["rotation"] = [w, x, y, z]
            sample_annotation["prev"] = sample_annotation_tokens[i - 1][j] if i > 0 else ""
            sample_annotation["next"] = sample_annotation_tokens[i + 1][j] if i < nbr_annotations - 1 else ""
            sample_annotation["num_lidar_pts"] = 0
            sample_annotation["num_radar_pts"] = 0
            sample_annotation_list.append(sample_annotation)
    return sample_annotation_list


def write_list_into_json(data_dir, filename, data_list):
    file_dir = os.path.join(data_dir, filename)
    if os.path.exists(file_dir):
        with open(file_dir) as f_obj:
            content = json.load(f_obj)
    else:
        content = []
    content = content + data_list
    with open(file_dir, "w") as f_obj:
        json.dump(content, f_obj, indent=1)
    print("Success to write into %s" % file_dir)
    return
