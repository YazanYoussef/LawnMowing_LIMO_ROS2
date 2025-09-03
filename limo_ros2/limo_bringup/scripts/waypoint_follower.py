import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Quaternion
from action_msgs.msg import GoalStatus
import math
import time
from Pattern_Waypoint_Generation import GazeboVisualizer, parse_square_data ,generate_pattern

def euler_to_quaternion(roll, pitch, yaw):
    """Converts Euler angles (in radians) to Quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

class WaypointFollowerNode(Node):

    def __init__(self):
        super().__init__('waypoint_follower_node')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.get_logger().info("WaypointFollowerNode initialized.")
        self.get_logger().info("Waiting for '/navigate_to_pose' action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Action server '/navigate_to_pose' is available.")

    def send_goal(self, x, y, yaw_degrees):
        goal_msg = NavigateToPose.Goal()

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'  # IMPORTANT: Or your global frame (e.g., 'odom')

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0  # Assuming 2D navigation

        # Convert yaw (degrees) to radians for quaternion calculation
        yaw_radians = math.radians(yaw_degrees)
        # For 2D navigation, roll and pitch are 0
        q = euler_to_quaternion(0.0, 0.0, yaw_radians)
        pose.pose.orientation = q

        goal_msg.pose = pose

        self.get_logger().info(f"Sending goal: x={x}, y={y}, yaw={yaw_degrees}°")
        
        # Send the goal
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        # Wait for the server to accept the goal
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by server')
            return False

        self.get_logger().info('Goal accepted by server.')

        # Wait for the result
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        
        result = get_result_future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal reached successfully!')
            return True
        else:
            self.get_logger().error(f'Goal failed with status code: {status}')
            # You can map status codes to human-readable messages if needed
            # e.g., GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED
            return False

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # self.get_logger().info(f"Received feedback: Distance remaining = {feedback.distance_remaining:.2f} m")
        # You can log other feedback fields like navigation_time, estimated_time_remaining
        pass # Keep it less verbose for now


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollowerNode()

    # Define your list of waypoints: (x, y, yaw_in_degrees)
    # Load area data
    file_path = '/home/navinst/Desktop/LIMO_Experiment/square_coordinates.txt'
    areas = parse_square_data(file_path)

    # Select area and points (can be modified to accept user input)
    area_num = str(3)  # Example: using area 3
    area_data = areas[area_num]
    
    # Define start and stop points (using corners by default)
    start = area_data['corners'][0]  # Bottom-left corner
    stop = area_data['corners'][3]   # Top-right corner

    visualizer = GazeboVisualizer()
    visualizer.get_logger().info("Starting visualization tasks...")
    visualizer.get_logger().info("Clearing existing markers...")
    visualizer.get_logger().info("Visualizing the corners...")
    visualizer.visualize_square_in_gazebo(area_data['corners'])
    visualizer.get_logger().info("Visualizing start and end points...")
    visualizer.visualize_start_stop_points(start,stop)

    waypoints = generate_pattern(start, stop)

    try:
        for i, (x, y, yaw) in enumerate(waypoints):
            node.get_logger().info(f"--- Navigating to Waypoint {i+1}/{len(waypoints)} ---")
            success = node.send_goal(x, y, yaw)
            if not success:
                node.get_logger().error(f"Failed to reach waypoint {i+1}. Aborting sequence.")
                break
            node.get_logger().info(f"--- Waypoint {i+1} reached (or navigation attempt finished) ---")
            # Optional: Add a small delay between waypoints if needed
            # time.sleep(1.0) 
        
        node.get_logger().info("All waypoints processed or sequence aborted.")

    except KeyboardInterrupt:
        node.get_logger().info("Navigation interrupted by user.")
    except Exception as e:
        node.get_logger().error(f"An error occurred: {e}")
    finally:
        # It's good practice to destroy the node explicitly.
        # Though PyROS2 will do it automatically on shutdown.
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()