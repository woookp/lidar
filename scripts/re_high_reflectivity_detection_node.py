#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

def point_cloud_callback(msg):
    # 从消息中读取点云数据
    cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
    points = np.array(cloud_points)
    intensities = points[:, 3]

    # 设定反射率阈值
    reflectivity_threshold = 150
    high_reflectivity_points = points[intensities > reflectivity_threshold]

    if high_reflectivity_points.size > 0:
        # 按反射率对点云排序
        sorted_indices = np.argsort(high_reflectivity_points[:, 3])[::-1]
        sorted_high_reflectivity_points = high_reflectivity_points[sorted_indices]

        # 转换为Open3D点云
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(sorted_high_reflectivity_points[:, :3])

        # 体素下采样
        voxel_size = 0.05
        downsampled_pcd = o3d_pcd.voxel_down_sample(voxel_size)

        # 统计滤波去除离散点与噪声
        nb_neighbors = 20
        std_ratio = 2.0
        cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        cleaned_pcd = downsampled_pcd.select_by_index(ind)
        cleaned_points = np.asarray(cleaned_pcd.points)

        # 检查cleaned_points数组是否为空或点数不足2
        if cleaned_points.shape[0] < 2:
            rospy.loginfo("Cleaned points are insufficient for clustering")
            return

        # 使用DBSCAN聚类算法
        clustering = DBSCAN(eps=0.2, min_samples=10)
        labels = clustering.fit_predict(cleaned_points)
        max_label = labels.max()
        rospy.loginfo(f"Point cloud has {max_label + 1} clusters")

        t_shape_found = False

        # 找到第一个T型簇
        for label in range(max_label + 1):
            cluster_points = cleaned_points[labels == label, :3]
            if len(cluster_points) >= 10 and is_t_shape(cluster_points):
                t_shape_found = True
                rospy.loginfo(f"T形状点云位置: {cluster_points.mean(axis=0)}")
                
                # 计算簇的中心点坐标
                cluster_center = cluster_points.mean(axis=0)
                rospy.loginfo(f"Cluster center: {cluster_center}")
                
                # 计算中心点到原点的距离
                distance_to_origin = np.linalg.norm(cluster_center)
                
                # 获取簇内点的数量
                num_points_in_cluster = cluster_points.shape[0]
                
                # 打印结果
                rospy.loginfo(f"Cluster center: {cluster_center}")
                rospy.loginfo(f"Distance to origin: {distance_to_origin}")
                rospy.loginfo(f"Number of points in cluster: {num_points_in_cluster}")
                
                # 可视化
                colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                colors[labels < 0] = 0
                colors[labels == label, :3] = [1, 0, 0]

                o3d_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                o3d.visualization.draw_geometries([o3d_pcd])
                break

        if not t_shape_found:
            rospy.loginfo("未检测到T形状点云")

    else:
        rospy.loginfo("No high reflectivity points found")

def is_t_shape(points):
    """
    判断给定点云是否为T形状
    """
    # 使用主成分分析 (PCA) 获取主方向向量
    mean_centered = points - np.mean(points, axis=0)
    cov_matrix = np.cov(mean_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 投影点到主方向向量上
    projections = np.dot(points, principal_vector)
    min_proj, max_proj = np.min(projections), np.max(projections)
    threshold = (max_proj - min_proj) / 3
    
    # 分成两个部分进行分析
    vertical_part = points[(projections > -threshold) & (projections < threshold)]
    horizontal_part = points[(projections <= -threshold) | (projections >= threshold)]

    # 检查垂直部分和水平部分是否满足T形状的条件
    if vertical_part.shape[0] < 3 or horizontal_part.shape[0] < 3:
        return False

    # 使用凸包（Convex Hull）计算点云的凸包面积
    try:
        vertical_hull = ConvexHull(vertical_part[:, :2])
        horizontal_hull = ConvexHull(horizontal_part[:, :2])
    except QhullError:
        return False
    
    vertical_area = vertical_hull.volume
    horizontal_area = horizontal_hull.volume

    # 假设T形状的垂直部分和水平部分的面积有显著差异，并且水平部分点数多于垂直部分
    return vertical_area > horizontal_area and len(horizontal_part) > len(vertical_part)

def main():
    rospy.init_node('high_reflectivity_detection_node', anonymous=True)
    rospy.Subscriber("/livox/lidar", PointCloud2, point_cloud_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
