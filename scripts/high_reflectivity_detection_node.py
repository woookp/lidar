#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

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

        # 聚类分析
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2)
        labels = clustering.fit_predict(cleaned_points)
        max_label = labels.max()
        rospy.loginfo(f"Point cloud has {max_label + 1} clusters")

        min_variance = float('inf')
        best_cluster_points = None
        best_cluster_label = None

        # 选择方差最小的簇且簇内点数大于10
        for label in range(max_label + 1):
            cluster_points = cleaned_points[labels == label, :3]
            if len(cluster_points) >= 10:
                variance = np.var(cluster_points, axis=0).sum()
                if variance < min_variance:
                    min_variance = variance
                    best_cluster_points = cluster_points
                    best_cluster_label = label

        if best_cluster_points is None:
            rospy.loginfo("未找到长度大于10的子簇点")
            return

        # 计算簇的中心点坐标
        cluster_center = best_cluster_points.mean(axis=0)

        # 计算中心点到原点的距离
        distance_to_origin = np.linalg.norm(cluster_center)

        # 获取簇内点的数量
        num_points_in_cluster = best_cluster_points.shape[0]

        # 打印结果
        rospy.loginfo(f"Cluster center: {cluster_center}")
        rospy.loginfo(f"Distance to origin: {distance_to_origin}")
        rospy.loginfo(f"Number of points in cluster: {num_points_in_cluster}")

        # Visualize the point cloud with clusters
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0

        # 将最佳子簇点标记为红色
        colors[labels == best_cluster_label, :3] = [1, 0, 0]

        o3d_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([o3d_pcd])

    else:
        rospy.loginfo("No high reflectivity points found")

def main():
    rospy.init_node('high_reflectivity_detection_node', anonymous=True)
    rospy.Subscriber("/livox/lidar", PointCloud2, point_cloud_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
