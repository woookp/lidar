import rospy
from pvesti.msg import UAVPose
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

# 此py文件模拟无人机产生的vins数据，以及激光雷达获得的其他无人机的相对位置数据

def create_uav_pose_message(timestamp, positions):
    """
    Create a UAVPose message with the given timestamp and positions.
    """
    msg = UAVPose()
    msg.header = Header()
    msg.header.stamp = timestamp
    msg.uav_count = len(positions)
    msg.uav_positions = []
    for pos in positions:
        pose = Pose()
        pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        msg.uav_positions.append(pose)
    return msg

def create_odometry_message():
    """
    Create a fixed Odometry message.
    """
    odom_msg = Odometry()
    odom_msg.header = Header()
    odom_msg.header.stamp = rospy.Time.now()
    odom_msg.header.frame_id = "odom"
    odom_msg.child_frame_id = "base_link"
    
    # Setting a fixed position and orientation
    odom_msg.pose.pose.position = Point(1.0, 2.0, 2.0)
    odom_msg.pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    
    # Setting a fixed velocity
    odom_msg.twist.twist.linear.x = 0.0
    odom_msg.twist.twist.linear.y = 0.0
    odom_msg.twist.twist.linear.z = 0.0
    odom_msg.twist.twist.angular.x = 0.0
    odom_msg.twist.twist.angular.y = 0.0
    odom_msg.twist.twist.angular.z = 0.0
    
    return odom_msg

def main():
    rospy.init_node('uav_pose_publisher')
    pub_pose = rospy.Publisher('/uav_positions', UAVPose, queue_size=10)
    pub_odom = rospy.Publisher('/vins_estimator/odometry', Odometry, queue_size=10)

    rate = rospy.Rate(1)  # 1 Hz
    
    # Define UAV positions for testing
    positions_list = [
        [(1,1,2), (3,3,2), (6,1,2)],
        [(3.2, 3.2, 2), (1.7, 1.8, 2), (6.1, 1.8, 2)],
        [(6.4, 1.7, 2), (1.2, 1.2, 2), (3.9, 3.9, 2)],
        [(1.5, 1.5, 2), (3.3, 3.4, 2), (6.7, 1.2, 2)],
        [(1.3, 1.7, 2), (6.8, 1.6, 2), (3.2, 3.9, 2)]
    ]
    
    # Publish messages
    while not rospy.is_shutdown():
        for positions in positions_list:
            timestamp = rospy.Time.now()
            uav_pose_msg = create_uav_pose_message(timestamp, positions)
            pub_pose.publish(uav_pose_msg)
            
            odom_msg = create_odometry_message()
            pub_odom.publish(odom_msg)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

