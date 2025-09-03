import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point # Twist removed as it's not used in this snippet
from gazebo_msgs.msg import ModelStates
# visualization_msgs.msg.Marker and std_msgs.msg.ColorRGBA removed as they are not used by SDF spawning
from gazebo_msgs.srv import SpawnModel, DeleteModel

#------------------------ Function for reading the areas----------------------------------------#
# This function is non-ROS and remains largely unchanged.
def parse_square_data(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        
        if not lines:
            # print(f"Warning: File {file_path} is empty or contains only whitespace.")
            return {} # Return empty dict if file is empty

        split_index = -1
        try:
            split_index = lines.index("Centers")
        except ValueError:
            # print(f"Error: 'Centers' keyword not found in {file_path}.")
            return {} # Or raise an error

        corners_lines = lines[1:split_index]
        centers_lines = lines[split_index + 1:]
        
        squares = {}
        if len(corners_lines) != len(centers_lines):
            # print(f"Warning: Mismatch between number of corner lines and center lines in {file_path}.")
            # Process only the minimum number of pairs
            pass

        for i, (corner_line, center_line) in enumerate(zip(corners_lines, centers_lines), 1):
            try:
                nums = list(map(float, corner_line.split()))
                if len(nums) != 8:
                    # print(f"Warning: Incorrect number of corner coordinates for square {i} in {file_path}. Expected 8, got {len(nums)}. Skipping.")
                    continue
                
                corners = [
                    (nums[0], nums[1]),  # bottom_left
                    (nums[2], nums[3]),  # bottom_right
                    (nums[4], nums[5]),  # top_left
                    (nums[6], nums[7])   # top_right
                ]
                
                center_coords = list(map(float, center_line.split()))
                if len(center_coords) != 2:
                    # print(f"Warning: Incorrect number of center coordinates for square {i} in {file_path}. Expected 2, got {len(center_coords)}. Skipping.")
                    continue
                center = tuple(center_coords)
                
                squares[str(i)] = {
                    'corners': corners,
                    'center': center
                }
            except ValueError:
                # print(f"Warning: Could not parse numbers for square {i} in {file_path}. Line content: corner='{corner_line}', center='{center_line}'. Skipping.")
                continue
        
        return squares
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while parsing {file_path}: {e}")
        return {}

#################################################################################################

# Global constants for colors
start_marker_color = (0.0, 1.0, 0.0, 1.0)  # Green (ensure float values)
stop_marker_color = (1.0, 0.0, 0.0, 1.0)   # Red (ensure float values)

# Helper function for Euler to Quaternion conversion
# def euler_to_quaternion(roll, pitch, yaw):
#     """Convert Euler angles to quaternion."""
#     cy = math.cos(yaw * 0.5)
#     sy = math.sin(yaw * 0.5)
#     cp = math.cos(pitch * 0.5)
#     sp = math.sin(pitch * 0.5)
#     cr = math.cos(roll * 0.5)
#     sr = math.sin(roll * 0.5)

#     qw = cr * cp * cy + sr * sp * sy
#     qx = sr * cp * cy - cr * sp * sy
#     qy = cr * sp * cy + sr * cp * sy
#     qz = cr * cp * sy - sr * sp * cy
#     return qx, qy, qz, qw

class GazeboVisualizer(Node):
    def __init__(self):
        super().__init__('gazebo_visualizer_node')
        
        # Service clients
        self.spawn_model_client = self.create_client(SpawnModel, '/gazebo/spawn_sdf_model')
        self.delete_model_client = self.create_client(DeleteModel, '/gazebo/delete_model')

        self.get_logger().info("Waiting for Gazebo services...")
        while not self.spawn_model_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('/gazebo/spawn_sdf_model service not available, waiting again...')
        while not self.delete_model_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('/gazebo/delete_model service not available, waiting again...')
        
        # Subscription for model states (used by clear_existing_markers)
        self.model_states_msg = None
        self.model_states_subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10) # QoS profile depth 10 is common

        self.get_logger().info("GazeboVisualizer node initialized. Services and subscription are ready.")

    def model_states_callback(self, msg):
        self.model_states_msg = msg

    def wait_for_model_states(self, timeout_sec=5.0):
        """Waits for a ModelStates message to be received via the callback."""
        self.model_states_msg = None # Reset to ensure a fresh message is processed if needed
        start_time = self.get_clock().now()
        while self.model_states_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1) # Process callbacks
            if (self.get_clock().now() - start_time).nanoseconds / 1e9 > timeout_sec:
                self.get_logger().warn("Timeout waiting for /gazebo/model_states message.")
                return None
        # self.get_logger().info("Received /gazebo/model_states message.")
        return self.model_states_msg

    def spawn_gazebo_marker(self, name, pose, size=0.1, color=(1.0,0.0,0.0,1.0)):
        """Spawn a colored cube marker in Gazebo."""
        r, g, b, a = [float(c) for c in color] # Ensure color components are float

        marker_xml = f"""
        <sdf version="1.6">
        <model name="{name}">
            <static>true</static>
            <link name="link">
            <visual name="visual">
                <geometry>
                <box>
                    <size>{float(size)} {float(size)} 0.01</size>
                </box>
                </geometry>
                <material>
                <ambient>{r} {g} {b} {a}</ambient>
                <diffuse>{r} {g} {b} {a}</diffuse>
                </material>
            </visual>
            </link>
        </model>
        </sdf>
        """
        
        request = SpawnModel.Request()
        request.name = name
        request.model_xml = marker_xml
        request.initial_pose = pose
        request.reference_frame = "world" # Or "" for world, 'world' is common for Gazebo

        if not self.spawn_model_client.service_is_ready():
            self.get_logger().error(f"Service /gazebo/spawn_sdf_model not ready. Cannot spawn {name}.")
            return

        future = self.spawn_model_client.call_async(request)
        # Spin on this node until the future is complete or timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0) 

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f"Spawned marker {name} at x:{pose.position.x:.2f}, y:{pose.position.y:.2f}")
            else:
                self.get_logger().error(f"Failed to spawn marker {name}: {future.result().status_message}")
        else:
            if future.exception() is not None:
                self.get_logger().error(f"Spawn service call for {name} failed with exception: {future.exception()}")
            else:
                self.get_logger().error(f"Spawn service call for {name} timed out.")


    def visualize_start_stop_points(self, start_point, stop_point):
        """Visualize start (green) and stop (red) points in Gazebo"""
        # Start point marker
        start_pose = Pose()
        start_pose.position.x = float(start_point[0])
        start_pose.position.y = float(start_point[1])
        start_pose.position.z = 0.02 # Slightly above ground
        self.spawn_gazebo_marker(
            "start_point",
            start_pose,
            size=0.3,
            color=start_marker_color
        )
        
        # Stop point marker
        stop_pose = Pose()
        stop_pose.position.x = float(stop_point[0])
        stop_pose.position.y = float(stop_point[1])
        stop_pose.position.z = 0.02 # Slightly above ground
        self.spawn_gazebo_marker(
            "stop_point",
            stop_pose,
            size=0.3,
            color=stop_marker_color
        )

    def clear_existing_markers(self, prefixes=None, specific_names=None):
        if prefixes is None:
            prefixes = ["area_marker_", "area_1_corner_", "waypoint_"] # Common prefixes
        if specific_names is None:
            specific_names = ["start_point", "stop_point"]

        if not self.delete_model_client.service_is_ready():
            self.get_logger().error("Service /gazebo/delete_model not ready. Cannot clear markers.")
            return

        self.get_logger().info("Attempting to get model states for clearing markers...")
        model_states = self.wait_for_model_states(timeout_sec=5.0)

        if model_states is None:
            self.get_logger().warn("Could not get model states. Marker cleanup may be incomplete.")
            return

        deleted_count = 0
        models_to_delete = []
        for model_name in model_states.name:
            if model_name in specific_names:
                models_to_delete.append(model_name)
            else:
                for prefix in prefixes:
                    if model_name.startswith(prefix):
                        models_to_delete.append(model_name)
                        break
        
        if not models_to_delete:
            self.get_logger().info("No markers found matching specified prefixes/names to delete.")
            return

        self.get_logger().info(f"Found {len(models_to_delete)} markers to delete: {models_to_delete}")

        for model_name in models_to_delete:
            request = DeleteModel.Request()
            request.model_name = model_name
            
            future = self.delete_model_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

            if future.result() is not None:
                if future.result().success:
                    self.get_logger().info(f"Deleted marker {model_name}")
                    deleted_count += 1
                else:
                    # It's common for Gazebo to report failure if model already gone, treat as warning
                    self.get_logger().warn(f"Failed to delete marker {model_name} (or already deleted): {future.result().status_message}")
            else:
                if future.exception() is not None:
                     self.get_logger().warn(f"Delete service call for {model_name} failed with exception: {future.exception()}")
                else:
                    self.get_logger().warn(f"Delete service call for {model_name} timed out.")
        
        self.get_logger().info(f"Marker clearing process complete. Deleted {deleted_count} models.")


    def visualize_square_in_gazebo(self, corners, prefix="area_marker"):
        """Place markers at all corners of a square"""
        for i, (x,y) in enumerate(corners):
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.01  # Slightly above ground
            self.spawn_gazebo_marker(f"{prefix}_{i}", pose, size=0.2, color=(1.0,0.0,0.0,1.0)) # Default red

#################################################################################################

#------------------------ Waypoints Generation -------------------------------------------------#
# This function is non-ROS and remains unchanged.
# Note: The orientation tuple is (roll, pitch, yaw)
def generate_pattern(start, end, lane_spacing=None):
    """Generate lawnmower pattern waypoints"""
    x1, y1 = start
    x2, y2 = end
    direction = 1 if x2 >= x1 else -1 # >= to handle case where x1==x2 (vertical sweep first)
    
    # Initial yaw: 0 if moving along +X, PI if moving along -X
    initial_yaw = 0.0 if x2 >= x1 else 180
    waypoints = [((x1,y1,0.0),(0.0,0.0,initial_yaw))] 
    
    num_segments = 4 # Default number of lanes/segments
    lane_spacing = (y2-y1)/num_segments
    # if lane_spacing is None:
    #     if abs(y2 - y1) < 1e-6 : # If y1 and y2 are very close, avoid division by zero
    #         if num_segments > 0: # if user specified segments for a single line
    #             lane_spacing = 0 
    #         else: # if no segments, implies single line sweep
    #              num_segments = 0 # No turns, just start to end along X
    #              lane_spacing = 0
    #     else:
    #         lane_spacing = (y2-y1)/num_segments
    
    if abs(lane_spacing) < 1e-6 and num_segments > 0: # Effectively a single line sweep repeated
        num_segments = 0


    current_x = x1
    current_y = y1
    
    # Yaw for E-W movement: 0 (East), PI (West)
    # Yaw for N-S movement: PI/2 (North), -PI/2 (South)

    for i in range(num_segments + 1): # Iterate through each lane
        # Move horizontally
        target_x = x1 if direction == -1 else x2
        yaw_horizontal = 180 if direction == -1 else 0.0
        waypoints.append(((target_x, current_y, 0.0), (0.0, 0.0, yaw_horizontal)))
        current_x = target_x

        if i < num_segments: # If not the last segment, make a turn
            # Move vertically (lane change)
            next_y = current_y + lane_spacing
            yaw_vertical = 90 if lane_spacing > 0 else -90
            waypoints.append(((current_x, next_y, 0.0), (0.0, 0.0, yaw_vertical)))
            current_y = next_y
            
            # Flip direction for next horizontal sweep
            direction *= -1
        
    
    # # Ensure the last point is indeed the end point, with appropriate final orientation
    # final_yaw = 180 if direction == -1 else 0.0 # Aligned with the last horizontal segment
    # if waypoints[-1][0] != (x2,y2,0.0): # If last generated point is not end point
    #      # Add segment to reach y2 if current_y is not y2
    #     if abs(current_y - y2) > 1e-3: # If not at target y
    #         yaw_to_y2 = 90 if y2 > current_y else -90
    #         waypoints.append( ((current_x, y2, 0.0), (0.0,0.0, yaw_to_y2)) )
    #         current_y = y2
    #     # Add segment to reach x2 if current_x is not x2
    #     if abs(current_x - x2) > 1e-3: # If not at target x
    #         yaw_to_x2 = 0.0 if x2 > current_x else 90
    #         waypoints.append( ((x2, y2, 0.0), (0.0,0.0, yaw_to_x2)) )
    # else: # If last point IS the end point, update its orientation
    #     waypoints[-1] = ((x2,y2,0.0), (0.0,0.0,final_yaw))


    # # Remove duplicate consecutive waypoints (position and orientation)
    # unique_waypoints = []
    # if waypoints:
    #     unique_waypoints.append(waypoints[0])
    #     for i in range(1, len(waypoints)):
    #         # Check if position AND orientation are different enough
    #         pos_diff = np.linalg.norm(np.array(waypoints[i][0]) - np.array(unique_waypoints[-1][0])) > 1e-3
    #         orient_diff = np.linalg.norm(np.array(waypoints[i][1]) - np.array(unique_waypoints[-1][1])) > 1e-3
    #         if pos_diff or orient_diff:
    #              unique_waypoints.append(waypoints[i])
    
    return waypoints
#################################################################################################