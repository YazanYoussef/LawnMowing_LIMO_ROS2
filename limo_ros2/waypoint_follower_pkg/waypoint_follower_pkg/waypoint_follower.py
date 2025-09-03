#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Quaternion
import tf2_ros
from tf2_geometry_msgs import PoseStamped as TF2PoseStamped
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus
from std_msgs.msg import String
import math
import time
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import torch
import asyncio
import threading
import sys
import os
from rclpy.executors import SingleThreadedExecutor
from nav2_msgs.action import ComputePathToPose
from typing import List, Tuple, Optional
from waypoint_follower_pkg.pattern_waypoint_generation import GazeboVisualizer, parse_square_data ,generate_pattern, create_detailed_path

# Add TRATSS_Model directory to Python path so torch.load can find the modules
tratss_model_path = os.path.join(os.path.dirname(__file__), '..', 'TRATSS_Model')
if os.path.exists(tratss_model_path):
    sys.path.insert(0, os.path.abspath(tratss_model_path))

from waypoint_follower_pkg.TRATSS_Model.Model_testing import TRATSS_PLAN


def find_virtual_agent_location(initial_guess_m: Tuple[float, float],
                                target_points_m: List[Tuple[float, float]],
                                target_distances_m: List[float]) -> Tuple[float, float]:
    """
    Finds a virtual agent location that best satisfies pathfinding distance constraints.

    Args:
        initial_guess_m: The agent's real starting position (x, y) in meters.
        target_points_m: A list of real-world target points [(x1,y1), (x2,y2), ...].
        target_distances_m: A list of desired distances [d1, d2, ...] from the agent
                            to the corresponding target points.

    Returns:
        The optimized virtual agent location (x, y) in meters.
    """
    initial_guess = np.array(initial_guess_m)
    targets = np.array(target_points_m)
    distances = np.array(target_distances_m)

    # This is the "error function" we want to minimize.
    # It calculates the sum of squared differences between the ideal path distances
    # and the Euclidean distances from a candidate point `p`.
    def error_function(p):
        # p is a numpy array [x, y] representing the virtual agent's location
        euclidean_distances = np.sqrt(np.sum((targets - p)**2, axis=1))
        # Calculate the squared error
        errors = (euclidean_distances - distances)**2
        return np.sum(errors)

    # Run the optimizer
    result = minimize(
        fun=error_function,
        x0=initial_guess,
        method='L-BFGS-B'  # A robust optimization algorithm
    )

    if result.success:
        return tuple(result.x)
    else:
        # If optimization fails, return the initial guess as a fallback
        print(f"Warning: Virtual agent location optimization failed: {result.message}")
        return initial_guess_m

class WaypointFollowerNode(Node):
    def __init__(self):
        super().__init__('waypoint_follower_node')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.get_logger().info("WaypointFollowerNode initialized.")

    def wait_for_action_server(self, timeout_sec=10.0):
        """Waits for the action server to be available."""
        self.get_logger().info("Waiting for '/navigate_to_pose' action server...")
        start_time = time.time()
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            if time.time() - start_time > timeout_sec:
                self.get_logger().error(f"Action server not available after {timeout_sec} seconds!")
                return False
            self.get_logger().info("Action server not available, waiting again...")
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Action server '/navigate_to_pose' is available.")
        return True

    def send_goal_async(self, x, y, yaw_degrees):
        """Send goal asynchronously and return future for result"""
        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        
        yaw_radians = math.radians(yaw_degrees)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw_radians)
        pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        
        goal_msg.pose = pose
        self.get_logger().info(f"Sending goal: x={x}, y={y}, yaw={yaw_degrees}°")
        
        # Send the goal asynchronously
        return self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)

    
    def send_goal(self, x, y, yaw_degrees, executor, timeout_sec=300.0):
        """
        Enhanced goal sending with separated acceptance and execution monitoring
        """
        try:
            # Create and send goal
            goal_msg = NavigateToPose.Goal()
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            
            yaw_radians = math.radians(yaw_degrees)
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw_radians)
            pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            goal_msg.pose = pose
            
            self.get_logger().info(f"Sending goal: x={x:.3f}, y={y:.3f}, yaw={yaw_degrees}°")
            
            # Send goal and get future
            send_goal_future = self._action_client.send_goal_async(
                goal_msg, feedback_callback=self.feedback_callback)
            
            # Phase 1: Wait for goal acceptance with more aggressive spinning
            self.get_logger().info("Phase 1: Waiting for goal acceptance...")
            acceptance_timeout = 45.0
            start_time = time.time()
            spin_count = 0
            
            while not send_goal_future.done():
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed > acceptance_timeout:
                    self.get_logger().error(f"Goal acceptance timeout after {acceptance_timeout}s")
                    return False
                
                # More aggressive spinning
                executor.spin_once(timeout_sec=0.05)
                spin_count += 1
                
                # Log every 5 seconds during acceptance
                if spin_count % 100 == 0:  # Every ~5 seconds at 0.05s intervals
                    self.get_logger().info(f"Still waiting for acceptance... {elapsed:.1f}s elapsed")
            
            # Check if goal was accepted
            goal_handle = send_goal_future.result()
            if goal_handle is None:
                self.get_logger().error("Failed to get goal handle")
                return False
            
            if not goal_handle.accepted:
                self.get_logger().error("Goal rejected by navigation server")
                return False
            
            self.get_logger().info("✓ Goal ACCEPTED! Navigation started...")
            
            # Phase 2: Monitor execution
            get_result_future = goal_handle.get_result_async()
            execution_start = time.time()
            last_progress_log = execution_start
            
            while not get_result_future.done():
                current_time = time.time()
                execution_elapsed = current_time - execution_start
                
                # Progress logging every 20 seconds
                if current_time - last_progress_log >= 20.0:
                    remaining = timeout_sec - execution_elapsed
                    self.get_logger().info(f"🚀 Navigation progress: {execution_elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
                    last_progress_log = current_time
                
                # Check execution timeout
                if execution_elapsed > timeout_sec:
                    self.get_logger().error(f"Navigation timeout after {timeout_sec}s - canceling goal")
                    try:
                        cancel_future = goal_handle.cancel_goal_async()
                        # Wait up to 10 seconds for cancellation
                        cancel_deadline = time.time() + 10.0
                        while not cancel_future.done() and time.time() < cancel_deadline:
                            executor.spin_once(timeout_sec=0.1)
                        self.get_logger().info("Goal cancellation completed")
                    except Exception as e:
                        self.get_logger().error(f"Failed to cancel goal: {e}")
                    return False
                
                executor.spin_once(timeout_sec=0.1)
            
            # Check final result
            result = get_result_future.result()
            if result is None:
                self.get_logger().error("No result received from navigation")
                return False
            
            status = result.status
            execution_time = time.time() - execution_start
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"✅ Goal SUCCESS! Navigation completed in {execution_time:.1f}s")
                return True
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn(f"⚠️ Goal CANCELED after {execution_time:.1f}s")
                return False
            elif status == GoalStatus.STATUS_ABORTED:
                self.get_logger().error(f"❌ Goal ABORTED by navigation stack after {execution_time:.1f}s")
                return False
            else:
                self.get_logger().error(f"❌ Goal FAILED with status {status} after {execution_time:.1f}s")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Exception in send_goal_with_monitoring: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            return False

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Uncomment for more verbose feedback
        # self.get_logger().info(f"Distance remaining: {feedback.distance_remaining:.2f} m")
        pass

    def get_adjacency_matrix(self, num_areas, batch_size):
        Ai = np.ones((num_areas,num_areas))
        np.fill_diagonal(Ai,0)
        A = np.repeat(Ai[np.newaxis], batch_size,axis=0)
        return A.tolist()
    
    def init_planner_oracle(self, planner_topic: str = 'compute_path_to_pose'):
        # Use '/compute_path_to_pose' if your stack is fully rooted at '/'
        self._planner_client = ActionClient(self, ComputePathToPose, planner_topic)
        self.get_logger().info(f"Planner oracle binding to action: {planner_topic}")

    def wait_for_planner_server(self, timeout_sec: float = 10.0) -> bool:
        self.get_logger().info("Waiting for Nav2 planner 'compute_path_to_pose'...")
        start = time.time()
        while not self._planner_client.wait_for_server(timeout_sec=1.0):
            if time.time() - start > timeout_sec:
                self.get_logger().error(f"Planner action not available after {timeout_sec} s!")
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Planner action is available.")
        return True

    @staticmethod
    def _pose_stamped(x: float, y: float, frame: str = 'map') -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        return ps

    @staticmethod
    def _path_length(nav_path) -> float:
        poses = nav_path.poses
        if len(poses) < 2:
            return 0.0
        d = 0.0
        for a, b in zip(poses, poses[1:]):
            dx = b.pose.position.x - a.pose.position.x
            dy = b.pose.position.y - a.pose.position.y
            d += math.hypot(dx, dy)
        return d

    def nav2_path_length(self,
                         executor,
                         goal_xy: Tuple[float, float],
                         planner_id: str = '',
                         frame: str = 'map',
                         timeout_sec: float = 15.0) -> Optional[float]:
        """
        Computes path length using Nav2's planner action.
        The planner will use the robot's current pose as the start.
        This function uses the provided executor for proper ROS2 spinning.
        """
        goal = ComputePathToPose.Goal()
        goal.pose = self._pose_stamped(*goal_xy, frame)
        if planner_id:
            goal.planner_id = planner_id

        # Phase 1: Wait for goal acceptance
        send_fut = self._planner_client.send_goal_async(goal)
        start_time = time.time()
        while not send_fut.done():
            if time.time() - start_time > timeout_sec:
                self.get_logger().error(f"Planner goal acceptance timed out for goal {goal_xy}")
                return None
            executor.spin_once(timeout_sec=0.05)

        handle = send_fut.result()
        if not handle.accepted:
            self.get_logger().warn(f"Planner rejected path to goal={goal_xy}")
            return None

        # Phase 2: Wait for the result
        result_fut = handle.get_result_async()
        start_time = time.time()
        while not result_fut.done():
            if time.time() - start_time > timeout_sec:
                self.get_logger().error(f"Timed out waiting for planner result for goal {goal_xy}")
                handle.cancel_goal_async()  # Attempt to cancel
                return None
            executor.spin_once(timeout_sec=0.05)

        res = result_fut.result().result
        if res is None or not hasattr(res, 'path') or res.path is None:
            self.get_logger().warn(f"Planner returned no path for goal={goal_xy}")
            return None

        path_len = self._path_length(res.path)
        return path_len

class SimpleOdometryListener(Node):
    def __init__(self):
        super().__init__('simple_odometry_listener')
        
        # --- TF2 Setup ---
        # The buffer stores received transforms and the listener gets them over the wire
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # --- End TF2 Setup ---

        self.current_pose_odom = None # Store the latest pose in the odom frame
        self.pose_received = False
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odometry',
            self.odom_callback,
            10
        )
        
        self.get_logger().info("Odometry listener initialized, waiting for odometry data...")

    def odom_callback(self, msg: Odometry):
        # We just store the pose from odometry. The frame is msg.header.frame_id ('odom')
        self.current_pose_odom = msg.pose.pose
        
        if not self.pose_received:
            self.get_logger().info(f"First odometry received in frame '{msg.header.frame_id}'")
            self.pose_received = True

    def wait_for_odometry(self, timeout_sec=30.0):
        # (This function remains unchanged)
        self.get_logger().info("Waiting for odometry data...")
        start_time = time.time()
        while not self.pose_received and (time.time() - start_time < timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose_received:
            self.get_logger().info("Odometry data is available!")
            return True
        else:
            self.get_logger().error(f"No odometry data received after {timeout_sec} seconds!")
            return False

    def get_current_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        """
        Gets the robot's current pose and transforms it into the 'map' frame.
        Returns (x, y, yaw) in the map frame, or None if transformation fails.
        """
        if not self.pose_received or self.current_pose_odom is None:
            self.get_logger().warn("No odometry data available to transform!")
            return None

        # Create a PoseStamped message for the transformation
        pose_in_odom_frame = PoseStamped()
        pose_in_odom_frame.header.stamp = self.get_clock().now().to_msg()
        pose_in_odom_frame.header.frame_id = 'odom' # The frame this pose is currently in
        pose_in_odom_frame.pose = self.current_pose_odom

        target_frame = 'map'
        try:
            # Use the buffer to transform the pose
            transformed_pose = self.tf_buffer.transform(
                pose_in_odom_frame,
                target_frame,
                timeout=Duration(seconds=1.0) # Wait up to 1s for the transform
            )
            
            # Extract x, y, and yaw from the transformed pose
            x = transformed_pose.pose.position.x
            y = transformed_pose.pose.position.y
            q = transformed_pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            return (x, y, yaw)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Could not transform pose from 'odom' to 'map': {e}")
            return None

    def get_current_pose(self):
        # Keep this for backward compatibility or direct odom access if needed
        if self.current_pose_odom is None:
            return None
        
        q = self.current_pose_odom.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return (self.current_pose_odom.position.x, self.current_pose_odom.position.y, yaw)
    
    def wait_for_transform(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        """Waits for the transform from source_frame to target_frame to be available."""
        self.get_logger().info(f"Waiting for transform from '{source_frame}' to '{target_frame}'...")
        start_time = self.get_clock().now()
        
        while (self.get_clock().now() - start_time) < Duration(seconds=timeout_sec):
            try:
                # Check if the transform is available now. Time() means latest available.
                if self.tf_buffer.can_transform(target_frame, source_frame, Time()):
                    self.get_logger().info("✓ Transform is available.")
                    return True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                # This can happen if one of the frames doesn't exist yet at all
                self.get_logger().info(f"Waiting for frames to be published: {e}", throttle_duration_sec=1.0)

            # Spin the node to allow the TF listener's callback to process incoming transforms
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().error(f"Timeout waiting for transform from '{source_frame}' to '{target_frame}' after {timeout_sec}s.")
        return False

def main(args=None):
    rclpy.init(args=args)
    
    waypoint_node = WaypointFollowerNode()
    waypoint_node.init_planner_oracle('compute_path_to_pose') 
    odom_listener = SimpleOdometryListener()  
    visualizer = GazeboVisualizer()
    
    # Create executor and add nodes
    executor = SingleThreadedExecutor()
    executor.add_node(waypoint_node)
    executor.add_node(odom_listener)
    executor.add_node(visualizer)
    
    # Wait for odometry data to be available FIRST
    if not odom_listener.wait_for_odometry(timeout_sec=30.0):
        waypoint_node.get_logger().error("Failed to get odometry data. Exiting.")
        waypoint_node.destroy_node()
        odom_listener.destroy_node()
        visualizer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        return
    
    # Wait for action server to be available
    if not waypoint_node.wait_for_action_server(timeout_sec=30.0):
        waypoint_node.get_logger().error("Failed to connect to action server. Exiting.")
        waypoint_node.destroy_node()
        odom_listener.destroy_node()
        visualizer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        return
    
    if not waypoint_node.wait_for_planner_server(timeout_sec=30.0):
        waypoint_node.get_logger().error("Failed to connect to planner action. Exiting.")
        # ... (same cleanup as you do for failures)
        return
 
    # Wait for the crucial map-to-odom transform to be available from the localization system
    if not odom_listener.wait_for_transform('map', 'odom', timeout_sec=15.0):
        waypoint_node.get_logger().error("Failed to get map->odom transform. Is localization (e.g., AMCL) running? Exiting.")
        waypoint_node.destroy_node()
        odom_listener.destroy_node()
        visualizer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        return
    
    # Flag to track the overall mission status
    mission_failed = False
    
    try:
        file_path = '/home/navinst/Desktop/LIMO_Experiment/square_coordinates.txt'
        areas = parse_square_data(file_path)
        
        # Load model with explicit path handling
        model_path = '/home/navinst/Desktop/LIMO_Experiment/solution.pt'
        waypoint_node.get_logger().info(f"Loading model from: {model_path}")
        
        # Temporarily add the model directory to path
        model_dir = os.path.dirname(model_path)
        tratss_dir = '/home/navinst/limo_ros2/waypoint_follower_pkg/waypoint_follower_pkg/TRATSS_Model'
        
        if tratss_dir not in sys.path:
            sys.path.insert(0, tratss_dir)
            waypoint_node.get_logger().info(f"Added to Python path: {tratss_dir}")
        
        my_solution = torch.load(model_path, map_location='cpu')
        waypoint_node.get_logger().info("Model loaded successfully")
        
        solution_size = my_solution.size(1)
        online = True #This is to choose the execution mode: online: True, or offline:False
        visited_areas = [0]
        new_areas = [area for area in range(solution_size)]
        all_areas = [area for area in range(solution_size)]
        
        pub = waypoint_node.create_publisher(String, '/event_mark', 10)
        solution_update = False
        
        for i in range(solution_size):
            waypoint_node.get_logger().info(f"########## PROCESSING SOLUTION {i+1}/{my_solution.size(1)} ##########")
            
            if online: 
                current_A = [visited_areas[-1]]
                corners_of_areas = [[coord / 10.0 for corner in areas[str(a_idx + 1)]['corners'] for coord in corner] for a_idx in range(solution_size)]
                centers_of_areas = [[center / 10.0 for center in areas[str(a_idx + 1)]['center']] for a_idx in range(solution_size)]
                
                # Get current robot location 
                executor.spin_once(timeout_sec=0.1)  # Update odometry
                robot_pose = odom_listener.get_current_pose_in_map()
                
                if robot_pose is None:
                    waypoint_node.get_logger().error("Failed to get robot location in 'map' frame!")
                    mission_failed = True
                    break
                
                # ===== START: VIRTUAL AGENT & CURRENT AREA TRANSFORMATION =====

                start_xy_m = (float(robot_pose[0]), float(robot_pose[1]))

                # 1. Keep the planning graph with REAL, UNTRANSFORMED coordinates.
                centers_for_planning = deepcopy(centers_of_areas)
                corners_for_planning = deepcopy(corners_of_areas)

                if i != 0:
                    # 2. Gather path distance constraints from the corners of unvisited areas.
                    target_points_m = []
                    target_distances_m = []
                    waypoint_node.get_logger().info(f"Gathering path distances from corners of unvisited areas: {new_areas}")

                    for a_idx in new_areas:
                        key = str(a_idx + 1)
                        real_corners_m = [tuple(float(x) for x in pt) for pt in areas[key]['corners']]
                        for corner_m in real_corners_m:
                            Lp = waypoint_node.nav2_path_length(executor, corner_m, planner_id='GridBased')
                            if Lp is not None:
                                target_points_m.append(corner_m)
                                target_distances_m.append(Lp)
                            else:
                                target_points_m.append(corner_m)
                                target_distances_m.append(1000.0) # Large cost for unreachable corners
                            time.sleep(0.05)

                    # 3. Calculate the virtual agent location using the optimizer.
                    virtual_limo_location_m = start_xy_m
                    if target_points_m:
                        waypoint_node.get_logger().info("Optimizing virtual agent location...")
                        virtual_limo_location_m = find_virtual_agent_location(
                            start_xy_m, target_points_m, target_distances_m
                        )
                        waypoint_node.get_logger().info(f"Real Limo Loc: ({start_xy_m[0]:.2f}, {start_xy_m[1]:.2f})")
                        waypoint_node.get_logger().info(f"Virtual Limo Loc: ({virtual_limo_location_m[0]:.2f}, {virtual_limo_location_m[1]:.2f})")

                    # 4. Calculate the displacement vector between real and virtual locations (in meters).
                    delta_x_m = virtual_limo_location_m[0] - start_xy_m[0]
                    delta_y_m = virtual_limo_location_m[1] - start_xy_m[1]
                    waypoint_node.get_logger().info(f"Calculated virtual displacement (dx, dy) in meters: ({delta_x_m:.2f}, {delta_y_m:.2f})")

                    # 5. Apply this same displacement to the current area's coordinates.
                    if current_A:
                        current_area_idx = current_A[0]
                        waypoint_node.get_logger().info(f"Applying virtual displacement to current area: {current_area_idx + 1}")

                        # --- Shift the center of the current area ---
                        # Note: We apply the scaled-down delta to the scaled-down coordinates
                        original_center = centers_of_areas[current_area_idx]
                        shifted_center_x = original_center[0] + (delta_x_m / 10.0)
                        shifted_center_y = original_center[1] + (delta_y_m / 10.0)
                        centers_for_planning[current_area_idx] = [shifted_center_x, shifted_center_y]

                        # --- Shift the corners of the current area ---
                        original_corners_flat = corners_of_areas[current_area_idx]
                        shifted_corners_flat = []
                        # Iterate through the (x, y) pairs in the flat list
                        for k in range(0, len(original_corners_flat), 2):
                            shifted_x = original_corners_flat[k] + (delta_x_m / 10.0)
                            shifted_y = original_corners_flat[k+1] + (delta_y_m / 10.0)
                            shifted_corners_flat.extend([shifted_x, shifted_y])
                        corners_for_planning[current_area_idx] = shifted_corners_flat
                    
                    # 6. Set the final limo location for the planner.
                    limo_location = [virtual_limo_location_m[0] / 10.0, virtual_limo_location_m[1] / 10.0]

                else:
                    limo_location = [start_xy_m[0]/10.0,start_xy_m[1]/10.0]

                # ===== END: VIRTUAL AGENT & CURRENT AREA TRANSFORMATION =====

                waypoint_node.get_logger().info(f"Current robot location: x={robot_pose[0]:.3f}, y={robot_pose[1]:.3f}")
                
                try:
                    waypoint_node.get_logger().info("About to call TRATSS_PLAN...")
                    new_solution = TRATSS_PLAN(limo_location, corners_for_planning, centers_for_planning, current_A, visited_areas, new_areas)
                    waypoint_node.get_logger().info("TRATSS_PLAN completed successfully")
                except Exception as e:
                    waypoint_node.get_logger().error(f"Error in TRATSS_PLAN: {e}")
                    waypoint_node.get_logger().error(f"Error type: {type(e)}")
                    import traceback
                    waypoint_node.get_logger().error(f"Traceback: {traceback.format_exc()}")
                    raise e
                
            if online:#solution_update:
                new_area_index = new_solution.plan[0][0][0].item()
                if len(new_areas) == solution_size:
                    visited_areas = [new_area_index]
                else:
                    visited_areas.append(new_area_index)
                new_areas = [j for j in all_areas if j not in visited_areas]
                area_num = str(new_area_index+1)
                st_point = int(new_solution.plan[0][0][1])
                pattern_num = int(new_solution.plan[0][0][2])
                idx = st_point - 1
                waypoint_node.get_logger().info("A new plan has been obtained and will continue accordingly")
                pub.publish(String(data='New plan Obtained'))
            else:
                area_index = my_solution[0][i][0].item()
                area_num = str(area_index+1)
                st_point = int(my_solution[0][i][1])
                pattern_num = int(my_solution[0][i][2])
                idx = st_point - 1
            
            area_data = areas[area_num]
            start = area_data['corners'][idx]
            
            waypoint_node.get_logger().info(f"Solution: Area number is {area_num}, starting_point is {st_point}, and pattern is {pattern_num}")
            pub.publish(String(data=f'SOLUTION {i+1}/{my_solution.size(1)}, Area: {area_num} '))
            
            # Pattern logic 
            if pattern_num == 1:
                pattern = 'H_ZigZag'
                if (st_point == 1) or (st_point == 4):
                    stop = area_data['corners'][0] if idx > 0 else area_data['corners'][3]
                else:
                    stop = area_data['corners'][1] if idx > 1 else area_data['corners'][2]
            elif pattern_num == 2:
                pattern = 'V_ZigZag'
                if (st_point == 1) or (st_point == 2):
                    stop = area_data['corners'][0] if idx > 0 else area_data['corners'][1]
                else:
                    stop = area_data['corners'][2] if idx > 2 else area_data['corners'][3]
            else:
                pattern = 'Spiral'
                bottom_left_corner, top_right_corner = area_data['corners'][0], area_data['corners'][3]
                x_center = start[0] + ((top_right_corner[0] - bottom_left_corner[0]) / 2)
                y_center = start[1] + ((top_right_corner[1] - bottom_left_corner[1]) / 2)
                stop = (x_center, y_center)
            
            bottom_left_corner, top_right_corner = area_data['corners'][0], area_data['corners'][3]
            
            visualizer.clear_all_markers()
            visualizer.visualize_square_in_gazebo(area_data['corners'])
            visualizer.visualize_start_stop_points(start, stop)
            
            waypoints = generate_pattern(start, stop, bottom_left_corner, top_right_corner, pattern)
            
            # Navigate waypoints using executor
            for j, (x, y, yaw) in enumerate(waypoints):
                waypoint_node.get_logger().info(f"--- Navigating to Waypoint {j+1}/{len(waypoints)} ---")
                success = waypoint_node.send_goal(x, y, yaw, executor, timeout_sec=240.0)
                if not success:
                    waypoint_node.get_logger().error(f"Failed to reach waypoint {j+1}. ABORTING ENTIRE MISSION.")
                    mission_failed = True
                    break
            
            pub.publish(String(data=f'Finished SOLUTION {i+1}'))
            
            if mission_failed:
                break
        
        if not mission_failed:
            waypoint_node.get_logger().info("########## All solutions processed successfully. ##########")
    
    except KeyboardInterrupt:
        waypoint_node.get_logger().info("Navigation interrupted by user.")
    except Exception as e:
        waypoint_node.get_logger().error(f"An unhandled error occurred: {e}")
    finally:
        waypoint_node.get_logger().info("Shutting down nodes.")
        waypoint_node.destroy_node()
        odom_listener.destroy_node()
        visualizer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()