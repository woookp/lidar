#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from high_reflectivity_detection.msg import DetectionResult

# Publisher
detection_pub = rospy.Publisher('/detection_result', DetectionResult, queue_size=10)

def point_cloud_callback(msg):
    # Read point cloud data from message
    cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
    points = np.array(cloud_points)
    intensities = points[:, 3]

    # Set reflectivity threshold
    reflectivity_threshold = 150
    high_reflectivity_points = points[intensities > reflectivity_threshold]

    if high_reflectivity_points.size > 0:
        # Sort point cloud by reflectivity
        sorted_indices = np.argsort(high_reflectivity_points[:, 3])[::-1]
        sorted_high_reflectivity_points = high_reflectivity_points[sorted_indices]

        # Convert to Open3D point cloud
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(sorted_high_reflectivity_points[:, :3])

        # Voxel downsampling
        voxel_size = 0.05
        downsampled_pcd = o3d_pcd.voxel_down_sample(voxel_size)

        # Statistical outlier removal
        nb_neighbors = 20
        std_ratio = 2.0
        cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        cleaned_pcd = downsampled_pcd.select_by_index(ind)
        cleaned_points = np.asarray(cleaned_pcd.points)

        # Check if cleaned_points array is empty or has fewer than 2 points
        if cleaned_points.shape[0] < 2:
            rospy.loginfo("Cleaned points are insufficient for clustering")
            return

        # Custom clustering algorithm
        clusters = custom_clustering(cleaned_points, eps=0.1)

        variances_and_clusters = []

        # Calculate the variance of each cluster and select the clusters with the smallest variance
        for cluster in clusters:
            if len(cluster) >= 10:
                variance = np.var(cluster, axis=0).sum()
                variances_and_clusters.append((variance, cluster))

        if len(variances_and_clusters) == 0:
            rospy.loginfo("No subclusters with more than 10 points found")
            return

        # Sort by variance and select the smallest 2
        variances_and_clusters.sort(key=lambda x: x[0])
        best_clusters = variances_and_clusters[:2]

        # Log the results
        for i, (variance, best_cluster_points) in enumerate(best_clusters):
            cluster_center = best_cluster_points.mean(axis=0)
            distance_to_origin = np.linalg.norm(cluster_center)
            num_points_in_cluster = best_cluster_points.shape[0]
            rospy.loginfo("Cluster {} center: {}".format(i+1, cluster_center))
            rospy.loginfo("Distance to origin for cluster {}: {}".format(i+1, distance_to_origin))
            rospy.loginfo("Number of points in cluster {}: {}".format(i+1, num_points_in_cluster))

        # Publish the detection result
        publish_detection_result([(best_cluster_points.mean(axis=0), best_cluster_points) for variance, best_cluster_points in best_clusters])

        # Visualize clustering results
        # visualize_clusters(cleaned_points, clusters)

    else:
        rospy.loginfo("No high reflectivity points found")

def custom_clustering(points, eps):
    """
    Custom clustering algorithm based on connectivity.
    """
    visited = np.zeros(points.shape[0], dtype=bool)
    clusters = []

    def bfs(start_idx):
        queue = [start_idx]
        cluster = []
        visited[start_idx] = True
        while queue:
            idx = queue.pop(0)
            cluster.append(points[idx])
            neighbors = np.where(np.linalg.norm(points - points[idx], axis=1) < eps)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return np.array(cluster)

    for i in range(points.shape[0]):
        if not visited[i]:
            cluster = bfs(i)
            if len(cluster) > 0:
                clusters.append(cluster)

    return clusters

def visualize_clusters(points, clusters):
    """
    Visualize clustering results
    """
    labels = np.zeros(points.shape[0])
    label_counter = 1
    for cluster in clusters:
        for idx, point in enumerate(points):
            if any(np.all(point == cluster_point) for cluster_point in cluster):
                labels[idx] = label_counter
        label_counter += 1

    colors = plt.get_cmap("tab20")(labels / (label_counter if label_counter > 0 else 1))
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([o3d_pcd])

def publish_detection_result(clusters):
    """
    Publish detection results
    """
    detection_msg = DetectionResult()
    detection_msg.header = Header()
    detection_msg.header.stamp = rospy.Time.now()
    detection_msg.uav_count = len(clusters)

    for cluster_center, cluster_points in clusters:
        pose = Pose()
        pose.position.x = cluster_center[0]
        pose.position.y = cluster_center[1]
        pose.position.z = cluster_center[2]
        detection_msg.uav_positions.append(pose)

    detection_pub.publish(detection_msg)

def main():
    rospy.init_node('high_reflectivity_detection_node', anonymous=True)
    rospy.Subscriber("/livox/lidar", PointCloud2, point_cloud_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
