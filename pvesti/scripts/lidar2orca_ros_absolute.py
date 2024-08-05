#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import rospy
from pvesti.msg import UAVPose, UAVPosVel
from geometry_msgs.msg import Pose, Point, Twist, Vector3



class UAVInfo:
    def __init__(self) :
        self.timestamp = None
        self.num = 0
        self.positions = []
        
UAVInfo_pre = UAVInfo()
UAVInfo_cur = UAVInfo()


class PositionMatcher:
    def __init__(self):
        self.previous_positions = []

    def match_positions(self, current_positions):
        if len(self.previous_positions) == 0:
            self.previous_positions = current_positions
            return list(range(len(current_positions))), list(range(len(current_positions)))

        # Calculate cost matrix
        cost_matrix = np.zeros((len(self.previous_positions), len(current_positions)))
        for i in range(len(self.previous_positions)):
            for j in range(len(current_positions)):
                cost_matrix[i, j] = np.linalg.norm(np.array(self.previous_positions[i]) - np.array(current_positions[j]))

        # Use Hungarian algorithm to find the minimum cost matching
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matched_indices = col_idx

        self.previous_positions = current_positions

        #return the current index to previous index
        return row_idx.tolist(), matched_indices.tolist()

class KalmanFilterPredictor:
    def __init__(self, dt, process_noise=0.01, measurement_noise=5):
        # Initial state (position and velocity)
        self.x = None

        # State transition matrix
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]], dtype=float)

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]], dtype=float)

        # Initial state covariance
        self.P = np.eye(6) * 1000

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise

        # Process noise covariance
        self.Q = np.eye(6) * process_noise

        # Store the filtered positions and velocities
        self.filtered_positions = []
        self.filtered_velocities = []

    def initialize_state(self, initial_position):
        self.x = np.array([initial_position[0], initial_position[1], initial_position[2], 0, 0, 0], dtype=float)

    def predict(self, z, dt):
        # update delta time, because it is changed
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Prediction step
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # Measurement update step
        y = np.array(z) - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(6) - np.dot(K, self.H)), self.P)

        # Store the results
        self.filtered_positions.append((self.x[0], self.x[1], self.x[2]))
        self.filtered_velocities.append((self.x[3], self.x[4], self.x[5]))



def UAV_pose_callback(data):
    global UAVInfo_pre, UAVInfo_cur
    
    #UAVInfo_pre = UAVInfo_cur 这样有问题，因为UAVInfo_pre是类的实力，直接复制其实就是引用值，会一直与UAVInfo_cur指向同一个变量
    
    UAVInfo_pre = UAVInfo()
    UAVInfo_pre.timestamp = UAVInfo_cur.timestamp
    UAVInfo_pre.num = UAVInfo_cur.num
    UAVInfo_pre.positions = UAVInfo_cur.positions
    
    UAVInfo_cur.timestamp = data.header.stamp
    UAVInfo_cur.num = data.uav_count
    UAVInfo_cur.positions = [(pose.position.x, pose.position.y, pose.position.z) for pose in data.uav_positions]



def main():
    matcher = PositionMatcher()

    dt = 1.0
    kalman_predictors = {}
    kalman_index_map = {} # Map frrom current index to initial index

    # m = 10  # number of time steps
    # n = 5   # number of objects
    # positions = [[(random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)] for _ in range(m)]
    
    m = 5
    n = 3
    positions = [[(1,1,2), (3,3,2), (6,1,2)],
                            [(3.2, 3.2, 2), (1.7, 1.8, 2), (6.1, 1.8, 2)],
                            [(6.4, 1.7, 2), (1.2, 1.2, 2), (3.9, 3.9, 2)],
                            [(1.5, 1.5, 2), (3.3, 3.4, 2), (6.7, 1.2, 2)],
                            [(1.3, 1.7, 2), (6.8, 1.6, 2), (3.2, 3.9, 2)]]

    initial_indices = list(range(n))
    for i in range(m):
        positions_cur = positions[i]

        # Match positions
        row_index, matched_index = matcher.match_positions(positions_cur)

        
        if i == 0:
            for idx in initial_indices:
                kalman_predictors[idx] = KalmanFilterPredictor(dt)
                kalman_predictors[idx].initialize_state(positions_cur[idx])
                kalman_index_map[idx] = idx
        else:
            #！！关键转换！！
            kalman_index_map_new = {}
            for j, idx in zip(row_index, matched_index):
                #当前匹配中的last index对应的初始坐标
                initial_index = kalman_index_map[j]
                kalman_index_map_new[idx] = initial_index
            kalman_index_map = kalman_index_map_new
                
        for value, key in  kalman_index_map.items():
            kalman_predictors[key].predict(positions_cur[value], dt)

    for i in kalman_predictors:
        filtered_positions = [(float(x), float(y), float(z)) for (x, y, z) in kalman_predictors[i].filtered_positions]
        filtered_velocities = [(float(vx), float(vy), float(vz)) for (vx, vy, vz) in kalman_predictors[i].filtered_velocities]
        print(f"Object {i} filtered positions: {filtered_positions}\n")
        print(f"Object {i} filtered velocities: {filtered_velocities}\n")



def main2():
    rospy.init_node('pos_vel_transfer')
    rospy.Subscriber('/uav_positions', UAVPose, UAV_pose_callback,queue_size=1)
    
    pub = rospy.Publisher('/uav_pos_vel', UAVPosVel, queue_size=10)
    
    matcher = PositionMatcher()
    kalman_predictors = {}
    kalman_index_map = {} 
    
    global UAVInfo_pre, UAVInfo_cur
    
    first_run = True
    rate = rospy.Rate(1)
    
    #增加鲁棒性，判断无人机个数
    UAV_NUM = 3
    
    while not rospy.is_shutdown():
        if UAVInfo_cur.timestamp is not None and UAVInfo_pre.timestamp is not None:
            dt = UAVInfo_cur.timestamp.to_sec() - UAVInfo_pre.timestamp.to_sec()
            positions_cur = UAVInfo_cur.positions
            
             # Match positions
            row_index, matched_index = matcher.match_positions(positions_cur)
            
            if first_run:
                # initialize kalman filter
                for idx in range(UAV_NUM):
                    kalman_predictors[idx] = KalmanFilterPredictor(dt)
                    kalman_predictors[idx].initialize_state(positions_cur[idx])
                    kalman_index_map[idx] = idx
                first_run = False
            else:
                # match points
                kalman_index_map_new = {}
                for j, idx in zip(row_index, matched_index):
                    initial_index = kalman_index_map[j]
                    kalman_index_map_new[idx] = initial_index
                kalman_index_map = kalman_index_map_new
            
            for value, key in  kalman_index_map.items():
                kalman_predictors[key].predict(positions_cur[value], dt)
                
            # create publish topic
            msg = UAVPosVel()
            msg.header.frame_id = "world"
            msg.header.stamp = UAVInfo_cur.timestamp
            
            msg.uav_count = UAVInfo_cur.num
            msg.uav_positions = []
            msg.uav_velocities = []
            
            #kalman_predictors会不断变大，每次取最后一个为最新的值
            for i in kalman_predictors:
                # filtered_positions = [(float(x), float(y), float(z)) for (x, y, z) in kalman_predictors[i].filtered_positions]
                # filtered_velocities = [(float(vx), float(vy), float(vz)) for (vx, vy, vz) in kalman_predictors[i].filtered_velocities]
                
                filtered_position = kalman_predictors[i].filtered_positions[-1]
                filtered_velocity = kalman_predictors[i].filtered_velocities[-1] 
                
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = filtered_position
                msg.uav_positions.append(pose)
            
                twist = Twist()
                twist.linear.x, twist.linear.y, twist.linear.z = filtered_velocity
                msg.uav_velocities.append(twist)
                
        try:
            pub.publish(msg)
            rate.sleep()
        except:
            continue
            
            
        
    
    
    
    
if __name__ == "__main__":
    # test with fixed position
    #main()

    # run with ros topic
    main2()
