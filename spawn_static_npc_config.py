#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time
import yaml 
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls
import random
random.seed(303)
import argparse
import logging
from numpy import random

"""
command line:
python3 spawn_npc_config.py --safe --autopilot False --actor-config /data2/fsh/repository/carla/PythonAPI/actor_config/static_actor_config.yaml --center-point "0,0,0" --max-radius 100
"""

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=7000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    argparser.add_argument(
        '--actor-config',
        metavar='PATH',
        default=None,
        help='Path to actor configuration file')
    argparser.add_argument(
        '--autopilot',
        metavar='A',
        default=True,
        help='whether to set autopilot for vehicles (default: True) and pedestrians (default: True)')
    argparser.add_argument(
        '--center-point',
        default='0,0,0',
        help='Spawn vehicles and walkers around a center point')
    argparser.add_argument(
        '--max-radius',
        default=50000,
        help='Max radius to spawn vehicles and walkers around a center point')
    
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    # read config file
    with open(args.actor_config) as f:
        actor_config = yaml.load(f, Loader=yaml.FullLoader)
        wheels_flag = actor_config["global_settings"]["vehicle_four_wheels"]
        vehicle_config = actor_config["actor"]["vehicle"]
        walker_config = actor_config["actor"]["walker"]

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.global_percentage_speed_difference(-50.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)


        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        # blueprints = world.get_blueprint_library().filter(args.filterv)
        # blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        # if args.safe:
        #     blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        #     blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        #     blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        #     blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        #     blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        # blueprints = sorted(blueprints, key=lambda bp: bp.id)
        if args.center_point is not None and args.max_radius is not None:
            spawn_center_point = carla.Location()
            spawn_center_point.x = float(args.center_point.split(',')[0])
            spawn_center_point.y = float(args.center_point.split(',')[1])
            spawn_center_point.z = float(args.center_point.split(',')[2])
        else:
            spawn_center_point = None
        spawn_points = world.get_map().get_spawn_points()

        # Spawn vehicles around a center point
        if spawn_center_point is not None:
            for spawn_point in spawn_points:
                if spawn_point.location.distance(spawn_center_point) > args.max_radius:
                    spawn_points.remove(spawn_point)
        number_of_spawn_points = len(spawn_points)
        spawn_points = random.shuffle(spawn_points)
        # if args.number_of_vehicles < number_of_spawn_points:
        #     random.shuffle(spawn_points)
        # elif args.number_of_vehicles > number_of_spawn_points:
        #     msg = 'requested %d vehicles, but could only find %d spawn points'
        #     logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
        #     args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        spawned_vehicle_num = 0
        for vehicle_type in vehicle_config:
            blueprint = world.get_blueprint_library().filter(vehicle_type)[0]
            # print(blueprint.get_attribute('number_of_wheels'))
            if wheels_flag:
                if not(int(blueprint.get_attribute('number_of_wheels')) == 4):
                    continue
            if spawned_vehicle_num >= number_of_spawn_points:
                    break
            for i in range(vehicle_config[vehicle_type]["actor_num"]):
                if spawned_vehicle_num >= number_of_spawn_points:
                    break
                if blueprint.has_attribute('color'):
                    if vehicle_config[vehicle_type]["if_recommended_color"]:
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                    else:
                        color = vehicle_config[vehicle_type]["color"]
                        blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                if args.autopilot:
                    blueprint.set_attribute('role_name', 'autopilot')

                # prepare the light state of the cars to spawn
                light_state = vls.NONE
                if args.car_lights_on:
                    light_state = vls.Position | vls.LowBeam | vls.LowBeam

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, spawn_points[spawned_vehicle_num])
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                    .then(SetVehicleLightState(FutureActor, light_state)))
                spawned_vehicle_num += 1

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        max_number_of_walkers = 1000
        spawn_points = []
        spawn_points_x_int = []
        for i in range(max_number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                if spawn_center_point is not None:
                    if spawn_point.location.distance(spawn_center_point) > args.max_radius:
                        continue
                if int(loc.x) in spawn_points_x_int:
                    continue
                # print(spawn_point)
                spawn_points.append(spawn_point)
                spawn_points_x_int.append(int(loc.x))
        spawn_points = random.shuffle(spawn_points)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        walker_spawn_num = 0
        print("len of walker spawn point: ", len(spawn_points))
        # max_attempt = 10
        for walker_type in walker_config:
            walker_bp = world.get_blueprint_library().filter(walker_type)[0]
            for i in range(walker_config[walker_type]["actor_num"]): 
                if walker_spawn_num >= len(spawn_points):
                    break
                # spawn_point = spawn_points[len(spawn_points) - 1 - walker_spawn_num]
                spawn_point = spawn_points[walker_spawn_num]
                # spawn_point = None
                # attempt = 0
                # while spawn_point is None:
                #     spawn_point = carla.Transform()
                #     loc = world.get_random_location_from_navigation()
                #     if (loc != None):
                #         spawn_point.location = loc
                #         if spawn_point in spawn_points:
                #             if attempt < max_attempt:
                #                 attempt += 1
                #                 print("spawn_point already in spawn_points")
                #                 spawn_point = None
                #                 continue
                #             else:
                #                 print("max attempt reached")
                #                 break
                #         spawn_points.append(spawn_point)
                #     else: 
                #         spawn_point = None
                        
                # originally set as not invincible, but set by user in config yaml
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', str(walker_config[walker_type]["is_invincible"]))
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if walker_config[walker_type]["if_recommended_value"] == 'walk':
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    elif walker_config[walker_type]["if_recommended_value"] == 'run':
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                    else:
                        walker_speed.append(walker_config[walker_type]["speed"])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
                walker_spawn_num += 1
        results = client.apply_batch_sync(batch, True)

        if args.autopilot:
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            print("walker_speed: ", walker_speed)
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = world.get_actors(all_id)

            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            if not args.sync or not synchronous_master:
                world.wait_for_tick()
            else:
                world.tick()

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            # set how many pedestrians can cross the road
            world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(world.get_random_location_from_navigation())
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))


        # set physics of actors
        vehicles = world.get_actors().filter('vehicle.*')
        walkers = world.get_actors().filter('walker.pedestrian.*')
        for vehicle in vehicles:
            vehicle.set_simulate_physics(True)
        for walker in walkers:
            walker.set_simulate_physics(True)


        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        while True:
            if args.sync and synchronous_master:
                # print("0000")
                world.tick()
            else:
                # print("1")
                world.wait_for_tick()

    finally:

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
