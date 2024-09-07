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
from pvesti.msg import UAVPose

# Publisher
detection_pub = rospy.Publisher('/uav_positions', UAVPose, queue_size=1)

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
        voxel_size = 0.02  # Finer downsampling
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

        # Visualize clustering results
        # visualize_clusters(cleaned_points, clusters)

        t_shape_clusters = []

        # Find all T-shaped clusters
        for cluster_points in clusters:
            cluster_center = cluster.mean(axis = 0)
            distance_to_origin = np.linalg.norm(cluster_center)
            if distance_to_origin <= 3.5 and len(cluster_points) >= 10 and is_t_shape(cluster_points):
                cluster_center = cluster_points.mean(axis=0)
                t_shape_clusters.append((cluster_center, cluster_points))
                rospy.loginfo(f"T-shaped point cloud position: {cluster_center}")
            if distance_to_origin > 3.5 and len(cluster_points) >= 5 and is_t_shape(cluster_points):
                cluster_center = cluster_points.mean(axis=0)
                t_shape_clusters.append((cluster_center, cluster_points))
                rospy.loginfo(f"T-shaped point cloud position: {cluster_center}")


        if t_shape_clusters:
            rospy.loginfo(f"Detected {len(t_shape_clusters)} T-shaped point clouds")
            publish_detection_result(t_shape_clusters)
        else:
            rospy.loginfo("No T-shaped point clouds detected")
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

def is_t_shape(points):
    """
    Determine if a given point cloud is T-shaped
    """
    if len(points) < 3:
        return False

    # Use Convex Hull to compute the convex hull of the point cloud
    try:
        hull = ConvexHull(points[:, :2])
    except Exception:
        return False

    # Get vertices of the convex hull
    hull_points = points[hull.vertices]

    # Find the highest and lowest points of the convex hull
    min_y = np.min(hull_points[:, 1])
    max_y = np.max(hull_points[:, 1])
    min_x = np.min(hull_points[:, 0])
    max_x = np.max(hull_points[:, 0])

    # Compute height and width
    height = max_y - min_y
    width = max_x - min_x

    # Determine conditions for T-shaped structure
    vertical_threshold = height / 4
    horizontal_threshold = width / 4

    # Find points in the vertical and horizontal parts
    vertical_part = hull_points[(hull_points[:, 1] > (min_y + vertical_threshold)) & (hull_points[:, 1] < (max_y - vertical_threshold))]
    horizontal_part = hull_points[(hull_points[:, 0] > (min_x + horizontal_threshold)) & (hull_points[:, 0] < (max_x - horizontal_threshold))]

    # Check if the vertical and horizontal parts meet the conditions for a T-shape
    if len(vertical_part) > 0 and len(horizontal_part) > 0:
        return True

    return False

def publish_detection_result(t_shape_clusters):
    """
    Publish detection results
    """
    detection_msg = UAVPose()
    detection_msg.header = Header()
    detection_msg.header.stamp = rospy.Time.now()
    detection_msg.uav_count = len(t_shape_clusters)

    for cluster_center, _ in t_shape_clusters:
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
