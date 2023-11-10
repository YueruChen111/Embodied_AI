import glob
import os
import sys
import math
#from nav_msgs.msg import Odometry
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

class Instance(object):
    def __init__(self, instance, instance_type):
        self.id = instance.id
        self.type_id = instance.type_id
        self.bounding_box = instance.bounding_box
        self.transform = instance.get_transform()
        self.attributes = instance.attributes
        self.instance_type = instance_type


def get_npc_information(
    world, location
) -> tuple:
    """
    Get all the NPCs information near certain location.

    Args:
        world(carla.World object): carla world. Get by world = client.get_world()
        location(carla.Location object):the center location of the searching area. Get by location = carla.Location(x, y, z) or Transform.location
        radius(float): the radius of the searching area, default is 0

    Returns:
        all_instance_data(list): all the NPCs information near certain location
        all_instance_id(list): all the NPCs id near certain location
        all_vehicles_data(list): all the vehicles information near certain location
        all_vehicles_id(list): all the vehicles id near certain location
        all_walkers_data(list): all the walkers information near certain location
        all_walkers_id(list): all the walkers id near certain location
    """
    
    hovering_flag = False
    hovering_max_distance = 0.5
    dangerous_flag = False
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
            if distance_2d < hovering_max_distance and vehicle_z > location.z:
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
            if distance_2d < hovering_max_distance and walker_z > location.z:
                hovering_flag = True
        if distance_3d < min_distance_3d_person:
            min_distance_3d_person = distance_3d

    if(min_distance_3d_person<5)or(min_distance_3d_vehicle<5):
        dangerous_flag=True
    return (
        hovering_flag,
        dangerous_flag,
        min_distance_3d_person,
        min_distance_3d_vehicle 
    )

# if __name__ == '__main__':
#     carla_client = carla.Client('localhost',2000)
#     carla_client.set_timeout(10.0)
#     world = carla_client.get_world()
#     #print(carla_client.get_available_maps())
#     get_npc_information(location=carla.Location(x=0,y=0,z=0), radius = 0)