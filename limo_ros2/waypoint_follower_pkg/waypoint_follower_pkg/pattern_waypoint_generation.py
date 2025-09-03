#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point # Twist removed as it's not used in this snippet
from gazebo_msgs.msg import ModelStates
# visualization_msgs.msg.Marker and std_msgs.msg.ColorRGBA removed as they are not used by SDF spawning
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SpawnModel, DeleteModel
import asyncio
import threading
import time

#------------------------ Function for reading the areas----------------------------------------#
# This function is to read areas following the structure by which they were generated
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


class GazeboVisualizer(Node):
    def __init__(self):
        super().__init__('gazebo_visualizer_node')
        
        # Service clients
        self.spawn_model_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_model_client = self.create_client(DeleteEntity, '/delete_entity')

        self.get_logger().info("Waiting for Gazebo services...")
        while not self.spawn_model_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().info('/spawn_entity service not available, waiting again...')
        while not self.delete_model_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().info('/delete_entity service not available, waiting again...')
        
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

    def wait_for_model_states(self, timeout_sec=20.0):
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
        
        request = SpawnEntity.Request()
        request.name = name
        request.xml = marker_xml
        request.initial_pose = pose
        request.reference_frame = "world" # Or "" for world, 'world' is common for Gazebo

        if not self.spawn_model_client.service_is_ready():
            self.get_logger().error(f"Service /spawn_entity not ready. Cannot spawn {name}.")
            return

        future = self.spawn_model_client.call_async(request)
        # Spin on this node until the future is complete or timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0) 

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

    def clear_all_markers(self):
        """
        Blindly delete any markers we might have spawned previously.
        We ignore failures so you always start fresh.
        """
        # 1) Build the list of names you want to wipe out
        names = ['start_point', 'stop_point']
        max_markers = 10  # adjust to >= the max number of area/waypoint markers you’ll ever spawn
        for i in range(max_markers):
            names += [
                f'area_marker_{i}',
                f'area_1_corner_{i}',
                f'waypoint_{i}',
            ]

        # 2) Fire off delete requests, but don’t error if they aren’t there
        for name in names:
            req = DeleteEntity.Request()
            req.name = name
            future = self.delete_model_client.call_async(req)
            # give it a quick spin so it actually goes out
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.1)
            # we don’t care if it failed—either it was gone or we removed it
    
    def clear_existing_markers(self, prefixes=None, specific_names=None):
        if prefixes is None:
            prefixes = ["area_marker_", "area_1_corner_", "waypoint_"] # Common prefixes
        if specific_names is None:
            specific_names = ["start_point", "stop_point"]

        if not self.delete_model_client.service_is_ready():
            self.get_logger().error("Service /gazebo/delete_model not ready. Cannot clear markers.")
            return

        self.get_logger().info("Attempting to get model states for clearing markers...")
        model_states = self.wait_for_model_states(timeout_sec=10.0)

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
            request = DeleteEntity.Request()
            request.name = model_name
            
            future = self.delete_model_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

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
# Note: The orientation tuple is (roll, pitch, yaw)
# HELPER FUNCTION: This creates the smooth 'Turn-Move-Turn' path.
def generate_path_segment(start_pose, end_pose, linear_step=0.25):
    """
    Generates a detailed list of waypoints for a 'Turn-Move-Turn' sequence.
    This creates smoother navigation by separating turning from moving.

    Args:
        start_pose (tuple): A tuple (x, y, yaw_degrees) for the starting pose.
        end_pose (tuple): A tuple (x, y, yaw_degrees) for the ending pose.
        linear_step (float): The distance between waypoints on the straight path.

    Returns:
        list: A list of (x, y, yaw_degrees) waypoints for the segment.
    """
    path = []
    start_x, start_y, start_yaw = start_pose
    end_x, end_y, end_yaw = end_pose

    # Calculate the angle and distance to the target position
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    distance = math.hypot(delta_x, delta_y)

    # Only generate points if there is a noticeable distance to travel
    if distance > 0.1:
        angle_to_target_rad = math.atan2(delta_y, delta_x)
        angle_to_target_deg = math.degrees(angle_to_target_rad)

        # 1. First Waypoint: Turn in place at the start position to face the target
        path.append((start_x, start_y, angle_to_target_deg))

        # 2. Intermediate Waypoints: Move in a straight line
        num_steps = int(distance / linear_step)
        for i in range(1, num_steps + 1):
            step_dist = i * linear_step
            intermediate_x = start_x + step_dist * math.cos(angle_to_target_rad)
            intermediate_y = start_y + step_dist * math.sin(angle_to_target_rad)
            path.append((intermediate_x, intermediate_y, angle_to_target_deg))

        # 3. Final Waypoint: Arrive at the destination
        path.append((end_x, end_y, angle_to_target_deg))

    # 4. Final Turn: Perform the final desired rotation at the end position
    # This is added regardless of distance to handle pure rotations.
    path.append((end_x, end_y, end_yaw))
    
    return path



def generate_pattern(start, end, corner_1, corner_4, pattern, lane_spacing=None, path_step_size = 0.20):
    
    def _densify(wps, step):
        """Interpolate positions only; keep yaw == A.yaw for all in-between points."""
        dense = [wps[0]]
        for A, B in zip(wps, wps[1:]):
            dx = B[0] - A[0]
            dy = B[1] - A[1]
            dist = math.hypot(dx, dy)
            N = max(1, math.ceil(dist / step))
            for i in range(1, N):
                t = i / N
                x = A[0] + t * dx
                y = A[1] + t * dy
                yaw = A[2]          # ← constant!
                dense.append((x, y, yaw))
            dense.append(B)        # B brings in the new yaw
        return dense
    if pattern == 'H_ZigZag':    
        """Generate horizontal lawnmower pattern waypoints"""
        x1, y1 = start
        x2, y2 = end
        direction = 1 if x2 > x1 else -1
        yaw = 90.0 if y2 > y1 else -90.0
        num_segments = 4 #Should be an even number
        lane_spacing = abs(y2-y1)/num_segments
        waypoints =  [(x1,y1,0.0)] if x2 > x1 else [(x1,y1,180.0)]
        current_y = y1
        current_x = x1
        
        for i in range(num_segments-1):
            waypoints.append((x2 if current_x == x1 else x1, current_y,0.0 if direction > 0 else 180.0))
            waypoints.append((x2 if current_x == x1 else x1, current_y,yaw))
            current_y += lane_spacing if y2 > y1 else -lane_spacing
            waypoints.append((x2 if current_x == x1 else x1, current_y, yaw))
            waypoints.append((x2 if current_x == x1 else x1, current_y, 180 if direction > 0 else 0.0))
            current_x = x2 if current_x == x1 else x1
            direction *= -1

        waypoints.append((x2 if current_x == x1 else x1, current_y,0.0 if direction > 0 else 180.0))
        waypoints.append((x2 if current_x == x1 else x1, current_y,yaw))
        current_y = y2
        waypoints.append((x2 if current_x == x1 else x1, current_y, yaw))
        waypoints.append((x2 if current_x == x1 else x1, current_y, 180 if direction > 0 else 0.0))
        direction *= -1
        waypoints.append(((x2,y2,0.0 if direction > 0 else 180)))

    elif pattern == 'V_ZigZag':
        """Generate vertical lawnmower pattern waypoints"""
        x1, y1 = start
        x2, y2 = end
        #-----------------
        y_c1 = corner_1[1]
        y_c4 = corner_4[1]
        #-----------------
        top = y1 > y_c1
        direction = 1 if x2 > x1 else -1
        yaw = -90.0 if top else 90.0
        num_segments = 3 #Should be an odd number
        lane_spacing = abs(x2-x1)/num_segments
        waypoints =  [(x1,y1,yaw)]
        current_x = x1
        
        for i in range(num_segments-1):
            waypoints.append((current_x, y_c1 if top else y_c4, yaw))
            waypoints.append((current_x, y_c1 if top else y_c4, 0.0 if direction > 0 else 180.0))
            current_x += lane_spacing if direction > 0 else -lane_spacing
            waypoints.append((current_x, y_c1 if top else y_c4, 0.0 if direction > 0 else 180.0))
            yaw *= -1
            waypoints.append((current_x, y_c1 if top else y_c4, yaw))
            top = not top
        
        waypoints.append((current_x, y_c1 if top else y_c4, yaw))
        waypoints.append((current_x, y_c1 if top else y_c4, 0.0 if direction > 0 else 180.0))
        current_x = x2
        waypoints.append((current_x, y_c1 if top else y_c4, 0.0 if direction > 0 else 180.0))
        yaw *= -1
        waypoints.append((current_x, y_c1 if top else y_c4, yaw))
        waypoints.append(((x2,y2,yaw)))
    
    elif pattern == 'Spiral':
        x_min, y_min = corner_1
        x_max, y_max = corner_4
        #-----------------------
        x_1, y_1 = start
        x_2, y_2 = end
        #-----------------------
        num_segments = 3 #Could be either odd or even
        lane_spacing = abs(x_2 - x_min)/num_segments
        #-----------------------
        vertical_direction = 1 if y_2 > y_1 else -1
        horizontal_direction = 1 if x_2 > x_1 else -1
        #-----------------------
        yaw = 90.0 if vertical_direction > 0 else -90.0

        waypoints = [(x_1,y_1,yaw)]
        current_x = x_1
        current_y = y_max if vertical_direction > 0 else y_min

        for i in range(num_segments):
            if vertical_direction > 0:
                current_y = y_max - i*lane_spacing
            else:
                current_y = y_min + i*lane_spacing
            waypoints.append((current_x, current_y, yaw))
            waypoints.append((current_x, current_y, 0.0 if horizontal_direction > 0 else 180.0))
            if horizontal_direction > 0:
                current_x = x_max - i*lane_spacing
            else:
                current_x = x_min + i*lane_spacing
            vertical_direction *= -1
            yaw *= -1
            waypoints.append((current_x, current_y, 0.0 if horizontal_direction > 0 else 180.0))
            horizontal_direction *= -1
            waypoints.append((current_x, current_y, yaw))
            if vertical_direction > 0:
                current_y = y_max - i*lane_spacing
            else:
                current_y = y_min + i*lane_spacing
            waypoints.append((current_x, current_y, yaw))
            waypoints.append((current_x, current_y, 0.0 if horizontal_direction > 0 else 180.0))
            if horizontal_direction > 0:
                current_x = x_max - (i+1)*lane_spacing
            else:
                current_x = x_min + (i+1)*lane_spacing
            waypoints.append((current_x, current_y, 0.0 if horizontal_direction > 0 else 180.0))
            vertical_direction *= -1
            yaw *= -1
            waypoints.append((current_x, current_y, yaw))
            horizontal_direction *=-1
        
        waypoints.append((x_2,y_2,yaw))       

    return _densify(waypoints, path_step_size)

#-------------------------Master function for generating smooth path----------------------------------
def create_detailed_path(actual_robot_pose, key_waypoints):
    """
    Builds a complete, detailed navigation path.
    """
    if not key_waypoints:
        return []

    full_path = []

    # Part 1: Generate the Entry Path from the robot's real pose
    first_target_pose = key_waypoints[0]
    entry_path = generate_path_segment(actual_robot_pose, first_target_pose)
    full_path.extend(entry_path)

    # Part 2: Generate paths between all the other key waypoints
    for i in range(len(key_waypoints) - 1):
        start_of_segment = key_waypoints[i]
        end_of_segment = key_waypoints[i+1]
        
        internal_path_segment = generate_path_segment(start_of_segment, end_of_segment)
        
        if len(internal_path_segment) > 1:
            full_path.extend(internal_path_segment[1:])

    return full_path
#################################################################################################
