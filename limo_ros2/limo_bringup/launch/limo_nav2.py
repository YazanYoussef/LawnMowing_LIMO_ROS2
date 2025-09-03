# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is all-in-one launch script intended for use by nav2 developers."""

import os, datetime

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, Command
from launch_ros.actions import Node


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('limo_bringup')
    launch_dir = os.path.join(bringup_dir, 'launch')

    #Information needed for the distance logger and the path to save the obtained file
    ts = datetime.datetime.now().strftime('%d_%m_%Y_%H%M')
    distance_csv_path = os.path.join(bringup_dir, f'covered_distances/distance_{ts}.csv')
    events_csv_path = os.path.join(bringup_dir, f'covered_distances/event_info_{ts}.csv')
    distance_logger_location = os.path.join(launch_dir, 'distance_logger.py')

    # Create the launch configuration variables
    slam = LaunchConfiguration('slam')
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    autostart = LaunchConfiguration('autostart')

    # Launch configuration variables specific to simulation
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_simulator = LaunchConfiguration('use_simulator')
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')
    world = LaunchConfiguration('world')

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    # In case of the transforms (tf), currently, there doesn't seem to be a better alternative
    # https://github.com/ros/geometry2/issues/32
    # https://github.com/ros/robot_state_publisher/pull/30
    # TODO(orduno) Substitute with `PushNodeRemapping`
    #              https://github.com/ros2/launch_ros/issues/56
    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Declare the launch arguments needed for the distance logger node
    odom_arg = DeclareLaunchArgument('odom_topic', default_value='/odometry')
    csv_arg  = DeclareLaunchArgument('csv_path',  default_value=distance_csv_path)
    events_arg   = DeclareLaunchArgument('events_csv_path',  default_value=events_csv_path)
    event_t_arg  = DeclareLaunchArgument('event_topic',      default_value='/event_mark')
    logn_arg     = DeclareLaunchArgument('log_every_n',      default_value='1')      # 0 = no per-odom rows
    min_arg  = DeclareLaunchArgument('min_step_m', default_value='0.001')
    max_arg  = DeclareLaunchArgument('max_step_m', default_value='5.0')
    
    # Declare the launch arguments

    declare_use_amcl_cmd = DeclareLaunchArgument(
        'use_amcl',
        default_value='false', # Set to 'false' to use Gazebo ground truth
        description='Whether to launch AMCL for localization')
    
    use_amcl = LaunchConfiguration('use_amcl') # To use it below

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_namespace_cmd = DeclareLaunchArgument(
        'use_namespace',
        default_value='false',
        description='Whether to apply a namespace to the navigation stack')

    declare_slam_cmd = DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Whether run a SLAM')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(bringup_dir, 'maps', 'limo_free_space_map.yaml'),
        description='Full path to map file to load')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'param', 'navigation2.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_bt_xml_cmd = DeclareLaunchArgument(
        'default_bt_xml_filename',
        default_value=os.path.join(
            get_package_share_directory('nav2_bt_navigator'),
            'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
        description='Full path to the behavior tree xml file to use')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack')

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config_file',
        default_value=os.path.join(bringup_dir, 'rviz', 'nav2_default_view.rviz'),
        description='Full path to the RVIZ config file to use')

    declare_use_simulator_cmd = DeclareLaunchArgument(
        'use_simulator',
        default_value='True',
        description='Whether to start the simulator')

    declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
        'use_robot_state_pub',
        default_value='True',
        description='Whether to start the robot state publisher')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Whether to start RVIZ')

    declare_simulator_cmd = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Whether to execute gzclient)')

    declare_world_cmd = DeclareLaunchArgument(
        'world',
        # TODO(orduno) Switch back once ROS argument passing has been fixed upstream
        #              https://github.com/ROBOTIS-GIT/turtlebot3_simulations/issues/91
        # default_value=os.path.join(get_package_share_directory('turtlebot3_gazebo'),
        #                            'worlds/turtlebot3_worlds/waffle.model'),
        default_value=os.path.join(bringup_dir, 'worlds', 'map.world'),
        description='Full path to world model file to load')

    # Specify the actions

    #Use the gazebo launch file in the limo_description folder
    description_dir = get_package_share_directory('limo_description')
    launch_gazebo_rviz_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_dir, 'launch', 'gazebo_models_diff.launch.py')))


    # Calling and launching the distance logger function
    distance_recorder = ExecuteProcess(
        cmd=[
            'python3',
            distance_logger_location,
            '--ros-args',
            '-p', ['odom_topic:=',   LaunchConfiguration('odom_topic')],
            '-p', ['csv_path:=',     LaunchConfiguration('csv_path')],
            '-p', ['event_topic:=',     LaunchConfiguration('event_topic')],
            '-p', ['events_csv_path:=', LaunchConfiguration('events_csv_path')],
            '-p', ['log_every_n:=',     LaunchConfiguration('log_every_n')],
            '-p', ['min_step_m:=',   LaunchConfiguration('min_step_m')],
            '-p', ['max_step_m:=',   LaunchConfiguration('max_step_m')],
            '-p', 'use_sim_time:=true',
        ],
        output='screen'
    )
    
    # # urdf = os.path.join(bringup_dir, 'urdf', 'limo_four_diff.urdf')
    urdf_direct = get_package_share_directory('limo_description')
    urdf = os.path.join(urdf_direct, 'urdf', 'limo_four_diff.urdf')

    start_robot_state_publisher_cmd = Node(
        condition=IfCondition(use_robot_state_pub),
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=remappings,
        arguments=[urdf])

    rviz_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'rviz_launch.py')),
        condition=IfCondition(use_rviz),
        launch_arguments={'namespace': '',
                          'use_namespace': 'False',
                          'rviz_config': rviz_config_file}.items())

    bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'bringup_launch.py')),
        launch_arguments={'namespace': namespace,
                          'use_namespace': use_namespace,
                          'slam': slam,
                          'map': map_yaml_file,
                          'use_sim_time': use_sim_time,
                          'params_file': params_file,
                          'default_bt_xml_filename': default_bt_xml_filename,
                          'autostart': autostart,
                          'use_amcl': use_amcl}.items())


    # This node was added because we are using the groundtruth location published by Gazebo.
    # You will not need this node if you are to use AMCL
    static_tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'odom'], # x y z qx qy qz qw parent child
        parameters=[{'use_sim_time': use_sim_time}]
    )

    
    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(odom_arg)
    ld.add_action(csv_arg)
    ld.add_action(events_arg)
    ld.add_action(event_t_arg)
    ld.add_action(logn_arg)
    ld.add_action(min_arg)
    ld.add_action(max_arg)
    ld.add_action(distance_recorder)

    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_namespace_cmd)
    ld.add_action(declare_slam_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_bt_xml_cmd)
    ld.add_action(declare_autostart_cmd)

    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_use_simulator_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_world_cmd)

    ld.add_action(declare_use_amcl_cmd) # --- ADD TO LAUNCH DESCRIPTION ---

    ld.add_action(static_tf_map_to_odom) # Make sure this is added to the LaunchDescription

    # Add any conditioned actions
    # ld.add_action(start_gazebo_server_cmd)
    # ld.add_action(start_gazebo_client_cmd)

    # Add the actions to launch all of the navigation nodes
    ld.add_action(start_robot_state_publisher_cmd)
    # ld.add_action(waypoint_follower_node)
    ld.add_action(launch_gazebo_rviz_cmd)
    ld.add_action(rviz_cmd)
    ld.add_action(bringup_cmd)

    return ld
