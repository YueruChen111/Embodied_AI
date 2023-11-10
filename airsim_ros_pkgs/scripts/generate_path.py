"""
Generate path for the drones
"""
import airsim
import time
import sys
import os
from datetime import datetime
import copy
import numpy as np
from airsim_control import confirm_and_initiate_drones, reset_drones
from utils import quaternionr2quaternion, vector3r2numpy, euler2quaternion
import glob
import json

town_paths = {'0-8': [(0, 0, -10), (190, 0, -10)],
              '8-6': [(190, 0, -10), (190, 127, -10)],
              '5-4': [(175, 200, -10), (-54, 200, -10)],
              '2-1': [(-125, 134, -10), (-125, 0, -10)],
              '1-12': [(-125, 0, -10), (-125, -195, -10)],
              '12-10': [(-125, -195, -10), (85, -195, -10)],
              '9-11': [(85, -195, -10), (-43, -195, -10)],
              '11-16': [(-43, -195, -10), (-43, -30, -10)],
              '17-4': [(-43, 27, -10), (-43, 200, -10)],
              '4-17': [(-43, 200, -10), (-43, 27, -10)],
              '16-11': [(-43, -30, -10), (-43, -195, -10)],
              '11-9': [(-43, -195, -10), (85, -195, -10)],
              '10-12': [(85, -195, -10), (-140, -195, -10)],
              '13-1': [(-125, -136, -10), (-125, 0, -10)],
              '1-2': [(-125, 0, -10), (-125, 134, -10)],
              '4-5': [(-54, 200, -10), (175, 200, -10)],
              '6-8': [(190, 127, -10), (190, 15, -10)],
              '8-0': [(190, 15, -10), (0, 15, -10)]}

drone_paths = ['0-8', '8-6', '5-4', '2-1', '1-12', '12-10', '9-11', '11-16', '17-4',
               '4-17', '16-11', '11-9', '10-12', '13-1', '1-2', '4-5', '6-8', '8-0']

five_drone_paths_town3 = {"Drone1": [[(-43, 0), (-43, 100)],
                                     [(-43, 100), (-43, 200)],
                                     [(-43, 200), (-125, 200)],
                                     [(-125, 200), (-125, 100)],
                                     [(-125, 100), (-125, 0)],
                                     [(-125, 0), (-125, -100)],
                                     [(-125, -100), (-125, 195)],
                                     [(-125, -195), (-43, -195)],
                                     [(-43, -195), (-43, -100)],
                                     [(-43, -100), (-43, 0)]],
                          "Drone4": [[(85, 0), (85, -136)],
                                     [(85, -136), (-15, -136)],
                                     [(-15, -136), (-120, -136)],
                                     [(-120, -136), (-120, 0)],
                                     [(-120, 0), (-15, 0)],
                                     [(-15, 0), (85, 0)]],
                          "Drone0": [[(-120, 15), (-15, 15)],
                                     [(-15, 15), (85, 15)],
                                     [(85, 15), (85, 190)],
                                     [(85, 190), (-15, 190)],
                                     [(-15, 190), (-120, 190)],
                                     [(-120, 190), (-120, 15)]],
                          "Drone2": [[(-30, 100), (-30, 10)],
                                     [(-30, 10), (80, 10)],
                                     [(80, 10), (190, 10)],
                                     [(190, 10), (190, 100)],
                                     [(190, 100), (190, 200)],
                                     [(190, 200), (80, 200)],
                                     [(80, 200), (-30, 200)],
                                     [(-30, 200), (-30, 100)]],
                          "Drone3": [[(-30, 5), (80, 5)],
                                     [(80, 5), (190, 5)],
                                     [(190, 5), (190, -95)],
                                     [(190, -95), (190, -195)],
                                     [(190, -195), (80, -195)],
                                     [(80, -195), (-30, -195)],
                                     [(-30, -195), (-30, -95)],
                                     [(-30, -95), (-30, 5)]]}

swarm_paths_town3 = {"Drone0": [[(-43, 200), (-125, 200)],
                                [(-125, 200), (-125, 100)],
                                [(-125, 100), (-125, 0)],
                                [(-125, 0), (-125, -100)],
                                [(-125, -100), (-125, 195)],
                                [(-125, -195), (-43, -195)],
                                [(-43, -195), (-43, -100)],
                                [(-43, -100), (-43, 0)],
                                [(-43, 0), (-43, 100)],
                                [(-43, 100), (-43, 200)]],
                    "Drone9": [[(-125, 0), (-125, -100)],
                                [(-125, -100), (-125, 195)],
                                [(-125, -195), (-43, -195)],
                                [(-43, -195), (-43, -100)],
                                [(-43, -100), (-43, 0)],
                                [(-43, 0), (-43, 100)],
                                [(-43, 100), (-43, 200)],
                                [(-43, 200), (-125, 200)],
                                [(-125, 200), (-125, 100)],
                                [(-125, 100), (-125, 0)]],
                    "Drone2": [[(-43, -195), (-43, -100)],
                                [(-43, -100), (-43, 0)],
                                [(-43, 0), (-43, 100)],
                                [(-43, 100), (-43, 200)],
                                [(-43, 200), (-125, 200)],
                                [(-125, 200), (-125, 100)],
                                [(-125, 100), (-125, 0)],
                                [(-125, 0), (-125, -100)],
                                [(-125, -100), (-125, 195)],
                                [(-125, -195), (-43, -195)]],
                    "Drone3": [[(-15, 0), (85, 0)],
                                [(85, 0), (85, -136)],
                                [(85, -136), (-15, -136)],
                                [(-15, -136), (-120, -136)],
                                [(-120, -136), (-120, 0)],
                                [(-120, 0), (-15, 0)]],
                    "Drone4": [[(-15, -136), (-120, -136)],
                                [(-120, -136), (-120, 0)],
                                [(-120, 0), (-15, 0)],
                                [(-15, 0), (85, 0)],
                                [(85, 0), (85, -136)],
                                [(85, -136), (-15, -136)]],
                    "Drone5": [[(-120, 15), (-15, 15)],
                                [(-15, 15), (85, 15)],
                                [(85, 15), (85, 190)],
                                [(85, 190), (-15, 190)],
                                [(-15, 190), (-120, 190)],
                                [(-120, 190), (-120, 15)]],
                    "Drone10": [[(85, 190), (-15, 190)],
                                [(-15, 190), (-120, 190)],
                                [(-120, 190), (-120, 15)],
                                [(-120, 15), (-15, 15)],
                                [(-15, 15), (85, 15)],
                                [(85, 15), (85, 190)]],
                    "Drone7": [[(-30, -195), (-30, -95)],
                                [(-30, -95), (-30, 5)],
                                [(-30, 5), (80, 5)],
                                [(80, 5), (190, 5)],
                                [(190, 5), (190, -95)],
                                [(190, -95), (190, -195)],
                                [(190, -195), (80, -195)],
                                [(80, -195), (-30, -195)]],
                    "Drone8": [[(190, 5), (190, -95)],
                                [(190, -95), (190, -195)],
                                [(190, -195), (80, -195)],
                                [(80, -195), (-30, -195)],
                                [(-30, -195), (-30, -95)],
                                [(-30, -95), (-30, 5)],
                                [(-30, 5), (80, 5)],
                                [(80, 5), (190, 5)]],
                    "Drone1": [[(-30, 100), (-30, 10)],
                                [(-30, 10), (80, 10)],
                                [(80, 10), (190, 10)],
                                [(190, 10), (190, 100)],
                                [(190, 100), (190, 200)],
                                [(190, 200), (80, 200)],
                                [(80, 200), (-30, 200)],
                                [(-30, 200), (-30, 100)]],
                    "Drone6": [[(190, 10), (190, 100)],
                                [(190, 100), (190, 200)],
                                [(190, 200), (80, 200)],
                                [(80, 200), (-30, 200)],
                                [(-30, 200), (-30, 100)],
                                [(-30, 100), (-30, 10)],
                                [(-30, 10), (80, 10)],
                                [(80, 10), (190, 10)]],
                    "Drone11": [[(190, 200), (80, 200)],
                                [(80, 200), (-30, 200)],
                                [(-30, 200), (-30, 100)],
                                [(-30, 100), (-30, 10)],
                                [(-30, 10), (80, 10)],
                                [(80, 10), (190, 10)],
                                [(190, 10), (190, 100)],
                                [(190, 100), (190, 200)]]}

five_drone_paths_town4 = {"Drone0": [[(-280, 0), (-280, -110)],
                                     [(-280, -110), (-280, -220)],
                                     [(-280, -220), (-176, -220)],
                                     [(-176, -220), (-176, -220)],
                                     [(-72, -220), (33, -220)],
                                     [(33, -220), (33, -110)],
                                     [(33, -110), (33, 0)],
                                     [(33, 0), (-72, 0)],
                                     [(-72, 0), (-176, 0)],
                                     [(-176, 0), (-280, 0)]],
                          "Drone3": [[(33, 220), (-72, 220)],
                                     [(-72, 220), (-176, 220)],
                                     [(-176, 220), (-280, 220)],
                                     [(-280, 220), (-280, 110)],
                                     [(-280, 110), (-280, 4)],
                                     [(-280, 4), (-176, 4)],
                                     [(-176, 4), (-72, 4)],
                                     [(-72, 4), (33, 4)],
                                     [(33, 4), (33, 110)],
                                     [(33, 110), (33, 220)]],
                          "Drone1": [[(-48, 2), (-48, -190)],
                                     [(-48, -190), (-236, -190)],
                                     [(-236, -190), (-236, 2)],
                                     [(-236, 2), (-236, 190)],
                                     [(-236, 190), (-48, 190)],
                                     [(-48, 190), (-48, 2)]],
                          "Drone2": [[(0, -240), (120, -240)],
                                     [(120, -240), (240, -240)],
                                     [(240, -240), (240, -90)],
                                     [(240, -90), (120, -90)],
                                     [(120, -90), (0, -90)],
                                     [(0, -90), (-126, -90)],
                                     [(-126, -90), (-126, -240)],
                                     [(-126, -240), (0, -240)]],
                          "Drone4": [[(240, 89), (240, 240)],
                                     [(240, 240), (120, 240)],
                                     [(120, 240), (0, 240)],
                                     [(0, 240), (-126, 240)],
                                     [(-126, 240), (-126, 89)],
                                     [(-126, 89), (0, 89)],
                                     [(0, 89), (120, 89)],
                                     [(120, 89), (240, 89)]]}

five_drone_paths_town5 = {"Drone0": [[(-280, 0), (-280, -110)],
                                     [(-280, -110), (-280, -220)],
                                     [(-280, -220), (-176, -220)],
                                     [(-176, -220), (-176, -220)],
                                     [(-72, -220), (33, -220)],
                                     [(33, -220), (33, -110)],
                                     [(33, -110), (33, 0)],
                                     [(33, 0), (-72, 0)],
                                     [(-72, 0), (-176, 0)],
                                     [(-176, 0), (-280, 0)]],
                          "Drone3": [[(33, 220), (-72, 220)],
                                     [(-72, 220), (-176, 220)],
                                     [(-176, 220), (-280, 220)],
                                     [(-280, 220), (-280, 110)],
                                     [(-280, 110), (-280, 4)],
                                     [(-280, 4), (-176, 4)],
                                     [(-176, 4), (-72, 4)],
                                     [(-72, 4), (33, 4)],
                                     [(33, 4), (33, 110)],
                                     [(33, 110), (33, 220)]],
                          "Drone1": [[(-48, 2), (-48, -190)],
                                     [(-48, -190), (-236, -190)],
                                     [(-236, -190), (-236, 2)],
                                     [(-236, 2), (-236, 190)],
                                     [(-236, 190), (-48, 190)],
                                     [(-48, 190), (-48, 2)]],
                          "Drone2": [[(0, -240), (120, -240)],
                                     [(120, -240), (240, -240)],
                                     [(240, -240), (240, -90)],
                                     [(240, -90), (120, -90)],
                                     [(120, -90), (0, -90)],
                                     [(0, -90), (-126, -90)],
                                     [(-126, -90), (-126, -240)],
                                     [(-126, -240), (0, -240)]],
                          "Drone4": [[(240, 89), (240, 240)],
                                     [(240, 240), (120, 240)],
                                     [(120, 240), (0, 240)],
                                     [(0, 240), (-126, 240)],
                                     [(-126, 240), (-126, 89)],
                                     [(-126, 89), (0, 89)],
                                     [(0, 89), (120, 89)],
                                     [(120, 89), (240, 89)]]}

five_drone_paths_town6 = {"Drone0": [[(-280, 0), (-280, -110)],
                                     [(-280, -110), (-280, -220)],
                                     [(-280, -220), (-176, -220)],
                                     [(-176, -220), (-176, -220)],
                                     [(-72, -220), (33, -220)],
                                     [(33, -220), (33, -110)],
                                     [(33, -110), (33, 0)],
                                     [(33, 0), (-72, 0)],
                                     [(-72, 0), (-176, 0)],
                                     [(-176, 0), (-280, 0)]],
                          "Drone3": [[(33, 220), (-72, 220)],
                                     [(-72, 220), (-176, 220)],
                                     [(-176, 220), (-280, 220)],
                                     [(-280, 220), (-280, 110)],
                                     [(-280, 110), (-280, 4)],
                                     [(-280, 4), (-176, 4)],
                                     [(-176, 4), (-72, 4)],
                                     [(-72, 4), (33, 4)],
                                     [(33, 4), (33, 110)],
                                     [(33, 110), (33, 220)]],
                          "Drone1": [[(-48, 2), (-48, -190)],
                                     [(-48, -190), (-236, -190)],
                                     [(-236, -190), (-236, 2)],
                                     [(-236, 2), (-236, 190)],
                                     [(-236, 190), (-48, 190)],
                                     [(-48, 190), (-48, 2)]],
                          "Drone2": [[(0, -240), (120, -240)],
                                     [(120, -240), (240, -240)],
                                     [(240, -240), (240, -90)],
                                     [(240, -90), (120, -90)],
                                     [(120, -90), (0, -90)],
                                     [(0, -90), (-126, -90)],
                                     [(-126, -90), (-126, -240)],
                                     [(-126, -240), (0, -240)]],
                          "Drone4": [[(240, 89), (240, 240)],
                                     [(240, 240), (120, 240)],
                                     [(120, 240), (0, 240)],
                                     [(0, 240), (-126, 240)],
                                     [(-126, 240), (-126, 89)],
                                     [(-126, 89), (0, 89)],
                                     [(0, 89), (120, 89)],
                                     [(120, 89), (240, 89)]]}


# path for arrow array
group_path_town3 = [[[(100, -250), (100, 110)]],
                    [[(-100, -110), (-100, 250)]]]

group_path_town4 = [[[(300, -260), (300, 90)]],
                    [[(250, 90), (250, -260)]],  # h=-54
                    [[(200, -260), (200, 90)]],
                    [[(150, 90), (150, -260)]],
                    [[(100, -260), (100, 90)]],  # 4
                    [[(50, 90), (50, -260)]],
                    [[(0, -320), (0, 30)], [(0, 30), (0, 380)]],
                    [[(300, -300), (-50, -300)]],  # 7
                    [[(0, -250), (350, -250)]],
                    [[(300, -200), (-50, -200)]],
                    [[(0, -150), (350, -150)]],
                    [[(300, -100), (-50, -100)], [(-50, -100), (-400, -100)]],  # 11
                    [[(0, -50), (350, -50)]],
                    [[(300, 0), (-50, 0)]],
                    [[(60, -185), (410, -385)]],  # 14, 30 degree
                    [[(350, -230), (0, -30)]],
                    [[(60, 105), (410, -95)]],
                    [[(350, 110), (0, 310)]],
                    [[(0, -230), (-350, -30)]],
                    [[(200, 0), (400, -350)]],  # 19, 60 degree
                    [[(230, -350), (30, 0)]],
                    [[(-105, -60), (95, -410)]],
                    [[(-110, -350), (-310, 0)]],
                    [[(230, 0), (30, 350)]],
                    [[(300, 0), (100, -350)]],  # 24, 120 degree
                    [[(70, -230), (270, 120)]],
                    [[(100, 0), (-100, -350)]],
                    [[(350, -100), (0, -300)]],  # 27, 150 degree
                    [[(0, -200), (350, 0)]],
                    [[(350, 100), (0, -100)]]]


group_path_town5 = []

group_path_town6 = [[[(-305, -35), (45, -35)], [(45, -35), (395, -35)], [(395, -35), (745, -35)]],
                    [[(605, 65), (255, 65)], [(255, 65), (-95, 65)], [(-95, 65), (-445, 65)]],
                    [[(-305, 165), (45, 165)], [(45, 165), (395, 165)], [(395, 165), (745, 165)]],
                    [[(605, 265), (255, 265)], [(255, 265), (-95, 265)], [(-95, 265), (-445, 265)]],
                    [[(-325, 220), (-325, -130)]],  # 4
                    [[(-215, 10), (-215, 360)]],
                    [[(-45, 220), (-45, -130)]],
                    [[(65, 10), (65, 360)]],
                    [[(235, 220), (235, -130)]],  # 8
                    [[(345, 10), (345, 360)]],
                    [[(515, 220), (515, -130)]],
                    [[(625, 10), (625, 360)]],
                    [[(-315, 80), (-15, -95)]],  # 12, 30 degree
                    [[(390, -25), (90, 150)], [(90, 150), (-210, 325)]],
                    [[(435, 255), (735, 80)]],
                    [[(-135, 255), (-435, 80)]],  # 15, 150 degree
                    [[(-90, -25), (210, 150)], [(210, 150), (510, 325)]],
                    [[(615, 80), (315, -95)]],
                    [[(-290, 230), (-90, -120)]],  # 18, 60 degree
                    [[(240, 0), (40, 350)]],
                    [[(510, 230), (710, -120)]],
                    [[(-340, 0), (-140, 350)]],  # 21, 120 degree
                    [[(190, 230), (-10, -120)]],
                    [[(460, 0), (660, 350)]]]


def generate_single_drone_path(scene_i, z):
    scene_i = scene_i % 18
    scene_path = drone_paths[scene_i]
    drone_path = town_paths[scene_path]
    if drone_path[0][0] - drone_path[1][0] == 0:
        path_direction = 'y'
    else:
        path_direction = 'x'

    if path_direction == 'x':
        path = []
        for i, loc in enumerate(drone_path):
            loc_ = list(loc)
            loc_[2] = z
            path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float((loc_[2]))))
            if i == 0:
                loc_[0] += 10 * float(np.sign(drone_path[1][0] - drone_path[0][0]))
                path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float(loc_[2])))
        return path
    else:
        path = []
        for i, loc in enumerate(drone_path):
            loc_ = list(loc)
            loc_[2] = z
            path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float((loc_[2]))))
            if i == 0:
                loc_[1] += 10 * float(np.sign(drone_path[1][1] - drone_path[0][1]))
                path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float(loc_[2])))
        return path


def get_group_drone_path(scene_i, airsim_drones, height, relative_position, route):
    paths = []
    for i, drone in enumerate(airsim_drones):
        drone_path = route[scene_i]
        p = relative_position[i]
        path = []
        for j, loc in enumerate(drone_path):
            path.append(airsim.Vector3r(float(loc[0] + p[0]), float(loc[1] + p[1]), float(height)))
            if j == 0:
                loc_ = list(loc)
                loc_[0] += 0.04 * float(drone_path[1][0] - drone_path[0][0])
                loc_[1] += 0.04 * float(drone_path[1][1] - drone_path[0][1])
                path.append(airsim.Vector3r(float(loc_[0] + p[0]), float(loc_[1] + p[1]), float(height)))
        paths.append(path)
    return paths


def get_single_drone_path(scene_i, drone, height, map_='town3'):
    if map_ == 'town3':
        five_drone_paths = five_drone_paths_town3
    elif map_ == 'town4':
        five_drone_paths = five_drone_paths_town4
    elif map_ == 'town5':
        five_drone_paths = five_drone_paths_town5
    elif map_ == 'town6':
        five_drone_paths = five_drone_paths_town6
    else:
        five_drone_paths = five_drone_paths_town5
    drone_path_all = five_drone_paths[drone]
    scene_i = scene_i % len(drone_path_all)
    drone_path = drone_path_all[scene_i]
    if drone_path[0][0] - drone_path[1][0] == 0:
        path_direction = 'y'
    else:
        path_direction = 'x'

    if path_direction == 'x':
        path = []
        for i, loc in enumerate(drone_path):
            path.append(airsim.Vector3r(float(loc[0]), float(loc[1]), float(height)))
            if i == 0:
                loc_ = list(loc)
                loc_[0] += 10 * float(np.sign(drone_path[1][0] - drone_path[0][0]))
                path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float(height)))
        return path
    else:
        path = []
        for i, loc in enumerate(drone_path):
            path.append(airsim.Vector3r(float(loc[0]), float(loc[1]), float(height)))
            if i == 0:
                loc_ = list(loc)
                loc_[1] += 10 * float(np.sign(drone_path[1][1] - drone_path[0][1]))
                path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float(height)))
        return path


def generate_drones_path(scene_i, z):
    scene_path = drone_paths[scene_i]
    drone_path = town_paths[scene_path]
    if drone_path[0][0] - drone_path[1][0] == 0:
        path_direction = 'y'
    else:
        path_direction = 'x'

    if path_direction == 'x':
        path = []
        for i, loc in enumerate(drone_path):
            loc_ = list(loc)
            loc_[2] = z
            path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float((loc_[2]))))
            if i == 0:
                loc_[0] += 5 * float(np.sign(drone_path[1][0] - drone_path[0][0]))
                path.append(airsim.Vector3r(float(loc_[0]), float(loc_[1]), float(loc_[2])))
        return [copy.deepcopy(path), copy.deepcopy(path), copy.deepcopy(path)]
    else:
        path_0 = []
        path_1 = []
        path_2 = []
        for i, loc in enumerate(drone_path):
            loc_0 = list(loc)
            loc_0[2] = z
            path_0.append(airsim.Vector3r(loc_0[0], loc_0[1], loc_0[2]))
            loc_1 = list(loc)
            loc_1[2] = z
            loc_1[0] -= 4
            loc_1[1] += 4
            path_1.append(airsim.Vector3r(loc_1[0], loc_1[1], loc_1[2]))
            loc_2 = list(loc)
            loc_2[2] = z
            loc_2[0] += 4
            loc_2[1] -= 4
            path_2.append(airsim.Vector3r(loc_2[0], loc_2[1], loc_2[2]))
            if i == 0:
                loc_0[1] += 5 * float(np.sign(drone_path[1][1] - drone_path[0][1]))
                loc_1[1] += 5 * float(np.sign(drone_path[1][1] - drone_path[0][1]))
                loc_2[1] += 5 * float(np.sign(drone_path[1][1] - drone_path[0][1]))
                path_0.append(airsim.Vector3r(loc_0[0], loc_0[1], loc_0[2]))
                path_1.append(airsim.Vector3r(loc_1[0], loc_1[1], loc_1[2]))
                path_2.append(airsim.Vector3r(loc_2[0], loc_2[1], loc_2[2]))
        return [path_0, path_1, path_2]


def generate_five_drones_path(scene_i, drones, height):
    paths = dict()
    for drone in drones:
        circuit_path = five_drone_paths_town3[drone]
        drone_path = circuit_path[scene_i % len(circuit_path)]
    return paths


def get_2d_rotation_matrix(x):
    return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])


def get_drone_formation(route, angle=np.pi / 2, distance=100):
    direction = np.array([[route[0][1][0] - route[0][0][0], route[0][1][1] - route[0][0][1]]])
    direction = direction / np.linalg.norm(direction)
    d1 = direction @ get_2d_rotation_matrix(np.pi - angle / 2)
    d2 = direction @ get_2d_rotation_matrix(np.pi + angle / 2)
    return [list((d1 * distance * 2)[0]),
            list((d1 * distance)[0]),
            [0.0, 0.0],
            list((d2 * distance)[0]),
            list((d2 * distance * 2)[0])]


def test():
    for route in group_path_town6:
        relative_position = get_drone_formation(route)
        print(route[0])
        print(relative_position)


def test_path():
    airsim_drones = ["Drone0", "Drone1", "Drone2", "Drone3", "Drone4"]
    z = - 40
    world_position = [-2.6, -2, -3.82]
    drone_position = [[0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0],
                      [4.0, 0.0, 0.0],
                      [6.0, 0.0, 0.0],
                      [8.0, 0.0, 0.0]]
    airsim_client = confirm_and_initiate_drones(airsim_drones, z)
    airsim.wait_key("Press any key to fly the whole path")

    motor_state = []
    for drone in airsim_drones:
        drone_path = get_single_drone_path(0, drone, z, 'town5')
        result = airsim_client.moveOnPathAsync([drone_path[0]],
                                               velocity=10,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone)
        motor_state.append(result)
    print("Fly to original point.")
    for ff in motor_state:
        ff.join()
    airsim.wait_key("Press any key to continue...")
    for drone in airsim_drones:
        result = None
        path_len = len(five_drone_paths_town5[drone])
        all_path = []
        for path_i in range(path_len):
            drone_path = get_single_drone_path(path_i, drone, z, 'town5')
            all_path = all_path + drone_path
        result = airsim_client.moveOnPathAsync(all_path,
                                               velocity=10,
                                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                                               yaw_mode=airsim.YawMode(False, 0),
                                               lookahead=-1,
                                               adaptive_lookahead=0,
                                               vehicle_name=drone)
        if result:
            motor_state.append(result)
        print("{}: flying the whole path.".format(drone))
    for ff in motor_state:
        ff.join()
    print("done")
    airsim.wait_key("Press any key to reset...")
    reset_drones(airsim_client, airsim_drones)


if __name__ == '__main__':
    try:
        test()
    except KeyboardInterrupt:
        print("keyboard interrupt")
        pass
    finally:
        print('\ndone.')
