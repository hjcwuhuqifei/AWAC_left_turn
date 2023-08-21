"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pyglet
from math import sqrt
import pickle
from pathlib import Path
import random
from collision import *
import pygame as pg
import sys

v = Vector
# clock = pg.time.Clock()
# SCREENSIZE = (500,500)
# screen = pg.display.set_mode(SCREENSIZE, pg.DOUBLEBUF|pg.HWACCEL)


def get_reward(observation, terminal, collision, action, reach):
    if collision:
        reward = 0.5 * action/20  - collision * 100
    else:
         reward = 0.5 * action/20 +  terminal * 10 * reach

    return reward


def normalize_angle(angle_rad):
    # to normalize an angle to [-pi, pi]
    a = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a = a + 2.0 * math.pi
    return a - math.pi


def linear_interpolate(path_point_0, path_point_1, rs_inter):
    ''' path point interpolated linearly according to rs value
    path_point_0 should be prior to path_point_1'''

    def lerp(x0, x1, w):
        return x0 + w * (x1 - x0)

    def slerp(a0, a1, w):
        # angular, for theta
        a0_n = normalize_angle(a0)
        a1_n = normalize_angle(a1)
        d = a1_n - a0_n
        if d > math.pi:
            d = d - 2 * math.pi
        elif d < -math.pi:
            d = d + 2 * math.pi
        a = a0_n + w * d
        return normalize_angle(a)

    rs_0 = path_point_0[2]
    rs_1 = path_point_1[2]
    weight = (rs_inter - rs_0) / (rs_1 - rs_0)
    if weight < 0 or weight > 1:
        print("weight error, not in [0, 1]")

    rx_inter = lerp(path_point_0[0], path_point_1[0], weight)
    ry_inter = lerp(path_point_0[1], path_point_1[1], weight)
    rtheta_inter = slerp(path_point_0[3], path_point_1[3], weight)
    return rx_inter, ry_inter, rtheta_inter


def object_to_ego(x, y, yaw):
    res_x = math.cos(yaw) * x - math.sin(yaw) * y
    res_y = math.sin(yaw) * x + math.cos(yaw) * y
    return res_x, res_y


class OfflineRL(gym.Env):
    def __init__(self):
        self.scenarios = pickle.load(open( '/home/haojiachen/桌面/AWAC_for_biye/AWAC/left_turn_data_and_scen/scenarios', 'rb'))

        self.scenario = random.choice(self.scenarios)

        self.ego_track = self.scenario['ego_car']
        self.collision_point = self.scenario['collision_point']

        self.start_time = self.ego_track.time_stamp_ms_first
        self.end_time = self.ego_track.time_stamp_ms_last

        self.ego_x = self.ego_track.motion_states[0].x 
        self.ego_y = self.ego_track.motion_states[0].y
        self.ego_yaw = self.ego_track.motion_states[0].psi_rad
        self.ego_v = sqrt(self.ego_track.motion_states[0].vx  ** 2 +
                          self.ego_track.motion_states[0].vy  ** 2)
        self.trajectory = []
        s = 0
        for i in range(self.start_time, self.end_time, 100):
            if i == 0:
                self.trajectory.append(
                    [self.ego_track.motion_states[0].x , self.ego_track.motion_states[0].y ,
                     s, self.ego_track.motion_states[0].psi_rad ])
            else:
                delta_s = sqrt((self.ego_track.motion_states[i].x  -
                                self.ego_track.motion_states[i - 100].x ) ** 2 +
                               (self.ego_track.motion_states[i].y  -
                                self.ego_track.motion_states[i - 100].y ) ** 2)
                s += delta_s
                self.trajectory.append(
                    [self.ego_track.motion_states[i].x, self.ego_track.motion_states[i].y,
                     s, self.ego_track.motion_states[i].psi_rad])

        # s += 10
        # self.trajectory.append([self.trajectory[-1][0] + 10 * math.cos(self.trajectory[-1][3]), self.trajectory[-1][1] + 10 * math.sin(self.trajectory[-1][3]),
        #              s, self.trajectory[-1][3]])

        self.surround1_poly = None
        self.surround2_poly = None
        self.surround1_track = None
        self.surround2_track = None

        self.surround1_poly = Concave_Poly(v(0,0), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])
        self.surround2_poly = Concave_Poly(v(0,0), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])
        self.ego_poly = Concave_Poly(v(0,0), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])

        self.s = 0
        self.time = 0
        self.dt = 100 #ms
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self.x_threshold = 100000  # 小车x方向最大运动范围
        self.v_threshold = 100000
        self.max_a = 1
        high = np.array([self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold
                         ],
                        dtype=np.float32)

        self.action_space = spaces.Box(
            low=-self.max_a,
            high=self.max_a, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        done = 0
        collision = 0
        reach = 0

        self.s += (action[0] * 4+ 4) * 0.1
        self.ego_v = action[0] * 4 + 4
        # 插值得到现在的x, y, yaw
        if self.s > self.trajectory[-1][2]:
            self.ego_x = self.trajectory[-1][0]
            self.ego_y = self.trajectory[-1][1]
            self.ego_yaw = self.trajectory[-1][3]
            reach = 1
            done = 1
        else:
            for i in range(1, len(self.trajectory)):
                if self.trajectory[i - 1][2] < self.s < self.trajectory[i][2]:
                    self.ego_x, self.ego_y, self.ego_yaw = linear_interpolate(self.trajectory[i - 1],
                                                                              self.trajectory[i],
                                                                              self.s)
                    break

        # 通过环境车辆获取当前观测值
        state = [0] * 17
        state[0] = self.ego_x 
        state[1] = self.ego_y 
        state[2] = self.ego_v * math.cos(self.ego_yaw)
        state[3] = self.ego_v * math.sin(self.ego_yaw)
        state[4] = self.ego_yaw 

        state[5] = self.collision_point[0]
        state[6] = self.collision_point[1]

        

        if 'surround1_car' in self.scenario:
            surround1_track = self.scenario['surround1_car']
            if surround1_track.time_stamp_ms_first <= self.time <= surround1_track.time_stamp_ms_last:
                state[7] = surround1_x = surround1_track.motion_states[self.time].x 
                state[8] = surround1_y = surround1_track.motion_states[self.time].y 
                state[9] = surround1_vx = surround1_track.motion_states[self.time].vx 
                state[10] = surround1_vy = surround1_track.motion_states[self.time].vy 
                state[11] = surround1_yaw = surround1_track.motion_states[self.time].psi_rad 
                self.surround1_poly.pos.x = surround1_x
                self.surround1_poly.pos.y = surround1_y
                self.surround1_poly.angle = surround1_yaw + math.pi / 2

        if 'surround2_car' in self.scenario:
            surround2_track = self.scenario['surround2_car']
            if surround2_track.time_stamp_ms_first <= self.time <= surround2_track.time_stamp_ms_last:
                state[12] = surround2_x = surround2_track.motion_states[self.time].x 
                state[13] = surround2_y = surround2_track.motion_states[self.time].y 
                state[14] = surround2_vx = surround2_track.motion_states[self.time].vx 
                state[15] = surround2_vy = surround2_track.motion_states[self.time].vy 
                state[16] = surround2_yaw = surround2_track.motion_states[self.time].psi_rad 
                self.surround2_poly.pos.x = surround2_x
                self.surround2_poly.pos.y = surround2_y
                self.surround2_poly.angle = surround2_yaw + math.pi / 2

        observation = np.array(state)

        # 碰撞检测  
        self.ego_poly.pos.x = self.ego_x
        self.ego_poly.pos.y = self.ego_y
        self.ego_poly.angle = self.ego_yaw + math.pi / 2

        if self.surround1_poly != None:
            if collide(self.ego_poly,self.surround1_poly):   collision = 1
        if self.surround2_poly != None:
            if collide(self.ego_poly,self.surround2_poly):   collision = 1

        self.time += 100

        if collision == 1:
            print('collision')
            done = 1
        if self.time > self.end_time:
            done = 1

        return observation, np.array(get_reward(observation, done, collision, action[0] * 10 + 10, reach)), np.array(done), collision

    def reset(self):
        self.scenarios = pickle.load(open( '/home/haojiachen/桌面/AWAC_for_biye/AWAC/left_turn_data_and_scen/scenarios', 'rb'))

        self.scenario = random.choice(self.scenarios)
        self.scenario = self.scenarios[8]

        self.ego_track = self.scenario['ego_car']
        self.collision_point = self.scenario['collision_point']

        self.start_time = self.ego_track.time_stamp_ms_first
        self.end_time = self.ego_track.time_stamp_ms_last

        self.ego_x = self.ego_track.motion_states[0].x 
        self.ego_y = self.ego_track.motion_states[0].y
        self.ego_yaw = self.ego_track.motion_states[0].psi_rad
        self.ego_v = sqrt(self.ego_track.motion_states[0].vx  ** 2 +
                          self.ego_track.motion_states[0].vy  ** 2)
        self.trajectory = []
        s = 0
        for i in range(self.start_time, self.end_time, 100):
            if i == 0:
                self.trajectory.append(
                    [self.ego_track.motion_states[0].x , self.ego_track.motion_states[0].y ,
                     s, self.ego_track.motion_states[0].psi_rad ])
            else:
                delta_s = sqrt((self.ego_track.motion_states[i].x  -
                                self.ego_track.motion_states[i - 100].x ) ** 2 +
                               (self.ego_track.motion_states[i].y  -
                                self.ego_track.motion_states[i - 100].y ) ** 2)
                s += delta_s
                self.trajectory.append(
                    [self.ego_track.motion_states[i].x, self.ego_track.motion_states[i].y,
                     s, self.ego_track.motion_states[i].psi_rad])
        
        self.s = 0
        self.time = 0

        self.ego_poly = Concave_Poly(v(self.ego_x,self.ego_y), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])
        self.ego_poly.angle = self.ego_yaw
        self.surround1_poly = None
        self.surround2_poly = None
        self.surround1_track = None
        self.surround2_track = None

        self.surround1_poly = Concave_Poly(v(0,0), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])
        self.surround2_poly = Concave_Poly(v(0,0), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])
        self.ego_poly = Concave_Poly(v(0,0), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])

        state = [0] * 17
        state[0] = self.ego_x 
        state[1] = self.ego_y 
        state[2] = self.ego_v * math.cos(self.ego_yaw)
        state[3] = self.ego_v * math.sin(self.ego_yaw)
        state[4] = self.ego_yaw 

        state[5] = self.collision_point[0]
        state[6] = self.collision_point[1]

        if 'surround1_car' in self.scenario:
            self.surround1_track = self.scenario['surround1_car']
            if self.surround1_track.time_stamp_ms_first <= self.time <= self.surround1_track.time_stamp_ms_last:
                state[7] = surround1_x = self.surround1_track.motion_states[self.time].x 
                state[8] = surround1_y = self.surround1_track.motion_states[self.time].y 
                state[9] = surround1_vx = self.surround1_track.motion_states[self.time].vx 
                state[10] = surround1_vy = self.surround1_track.motion_states[self.time].vy 
                state[11] = surround1_yaw = self.surround1_track.motion_states[self.time].psi_rad 
                self.surround1_poly = Concave_Poly(v(surround1_x,surround1_y), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])
                self.surround1_poly.angle = surround1_yaw

        if 'surround2_car' in self.scenario:
            surround2_track = self.scenario['surround2_car']
            if surround2_track.time_stamp_ms_first <= self.time <= surround2_track.time_stamp_ms_last:
                state[12] = surround2_x = surround2_track.motion_states[self.time].x 
                state[13] = surround2_y = surround2_track.motion_states[self.time].y 
                state[14] = surround2_vx = surround2_track.motion_states[self.time].vx 
                state[15] = surround2_vy = surround2_track.motion_states[self.time].vy 
                state[16] = surround2_yaw = surround2_track.motion_states[self.time].psi_rad 
                self.surround2_poly = Concave_Poly(v(surround2_x,surround2_y), [v(-0.9,2.3),  v(0.9,2.3), v(0.9,-2.3), v(-0.9,-2.3)])
                self.surround2_poly.angle = surround2_yaw
        
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(3,))
        observation = np.array(state)
        self.viewer = None
        self.steps_beyond_done = None
        return observation

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        ego_x = 300
        ego_y = 200
        car_width = 18
        car_length = 46

        # number_of_car = len(self.object_position_for_view)
        from gym.envs.classic_control import rendering
        l, r, t, b = -car_width / 2, car_width / 2, car_length / 2, -car_length / 2
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            ego_car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            ego_car.add_attr(self.carttrans)
            ego_car.set_color(1, 0, 0)
            self.viewer.add_geom(ego_car)

            if 'surround1_car' in self.scenario:
                surround1_car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.surround1_trans = rendering.Transform()
                surround1_car.add_attr(self.surround1_trans)
                self.viewer.add_geom(surround1_car)

            if 'surround2_car' in self.scenario:
                surround2_car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.surround2_trans = rendering.Transform()
                surround2_car.add_attr(self.surround2_trans)
                self.viewer.add_geom(surround2_car)

        # Edit the pole polygon vertex

        self.carttrans.set_translation(self.ego_x * 10 + ego_x, self.ego_y  * 10 + ego_y)
        self.carttrans.set_rotation(self.ego_yaw - math.pi / 2)

        if 'surround1_car' in self.scenario:
            self.surround1_track = self.scenario['surround1_car']
            if self.surround1_track.time_stamp_ms_first <= self.time <= self.surround1_track.time_stamp_ms_last:
                surround1_x = self.surround1_track.motion_states[self.time].x 
                surround1_y = self.surround1_track.motion_states[self.time].y
                surround1_yaw = self.surround1_track.motion_states[self.time].psi_rad 
                self.surround1_trans.set_translation(surround1_x * 10 + ego_x, surround1_y * 10 + ego_y)
                self.surround1_trans.set_rotation(surround1_yaw - math.pi / 2)
            else:
                self.surround1_trans.set_translation(10000, 10000)

        if 'surround2_car' in self.scenario:
            self.surround2_track = self.scenario['surround2_car']
            if self.surround2_track.time_stamp_ms_first <= self.time <= self.surround2_track.time_stamp_ms_last:
                surround2_x = self.surround2_track.motion_states[self.time].x 
                surround2_y = self.surround2_track.motion_states[self.time].y
                surround2_yaw = self.surround2_track.motion_states[self.time].psi_rad 
                self.surround2_trans.set_translation(surround2_x * 10 + ego_x, surround2_y * 10 + ego_y)
                self.surround2_trans.set_rotation(surround2_yaw - math.pi/2)
            else:
                self.surround2_trans.set_translation(10000, 10000)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # def plot_collision_for_test(self):
        
    #     for event in pg.event.get():
    #         if event.type == pg.QUIT:
    #             sys.exit()

    #     screen.fill((0,0,0))

        
    #     p0c, p1c, p2c = (0,255,255),(0,255,255),(0,255,255)
    #     self.ego_poly.pos.x += 250
    #     self.ego_poly.pos.y += 250
    #     pg.draw.polygon(screen, p0c, self.ego_poly.points, 3)
    #     if 'surround1_car' in self.scenario:
    #         self.surround1_poly.pos.x += 250
    #         self.surround1_poly.pos.y += 250
    #         pg.draw.polygon(screen, p0c, self.surround1_poly.points, 3)
    #     if 'surround2_car' in self.scenario:
    #         self.surround2_poly.pos.x += 250
    #         self.surround2_poly.pos.y += 250
    #         pg.draw.polygon(screen, p0c, self.surround2_poly.points, 3)
    #     pg.display.flip()
    #     clock.tick(100)
        

