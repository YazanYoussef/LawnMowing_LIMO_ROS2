
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command

from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('limo_bringup')
    launch_dir = os.path.join(bringup_dir, 'launch')

    namespace = LaunchConfiguration('namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    bt_xml_file = LaunchConfiguration('bt_xml_file')
    autostart = LaunchConfiguration('autostart')
    use_remappings = LaunchConfiguration('use_remappings')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')
    
    declare_map_yaml_cmd = DeclareLaunchArgument(
        name='map',
        default_value=os.path.join(get_package_share_directory('limo_bringup'), 'maps', 'limo_free_space_map.yaml'),
        description='Full path to map file to load'
    )

    declare_world = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            get_package_share_directory('limo_bringup'),
            'worlds', 'empty.world'
        ),
        description='Full path to world file to load'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'param', 'amcl_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')
    declare_bt_xml_cmd = DeclareLaunchArgument(
        'bt_xml_file',
        default_value=os.path.join(
            get_package_share_directory('nav2_bt_navigator'),
            'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
        description='Full path to the behavior tree xml file to use')
    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack')

    declare_use_remappings_cmd = DeclareLaunchArgument(
        'use_remappings', default_value= 'false',
        description='Arguments to pass to all nodes launched by the file')


    start_navigation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'limo_navigation.launch.py')),
        launch_arguments={'namespace': namespace,
                          'use_sim_time': use_sim_time,
                          'autostart': autostart,
                          'params_file': params_file,
                          'bt_xml_file': bt_xml_file,
                          'use_lifecycle_mgr': 'true', #'false',
                          'use_remappings': use_remappings,
                          'map_subscribe_transient_local': 'true'}.items())

    start_localization_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'limo_localization.launch.py')),
        launch_arguments={'namespace': namespace,
                          'map': map_yaml_file,
                          'use_sim_time': use_sim_time,
                          'autostart': autostart,
                          'params_file': params_file,
                          'use_lifecycle_mgr': 'true', #'false',
                          'use_remappings': use_remappings}.items())
                          
    start_lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': ['map',#'map_server',
                                    'amcl',
                                    'controller_server',
                                    'planner_server',
                                    'recoveries_server',
                                    'bt_navigator',
                                    'waypoint_follower']}])
    

    # Declare command-line arguments
    world       = LaunchConfiguration('world')

    # 1) Launch Gazebo (server + GUI) using the built-in launch script
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': world,
            'use_sim_time': use_sim_time
        }.items()
    )  # :contentReference[oaicite:0]{index=0}

    # 2) Generate and publish robot_description from your LIMO xacro
    xacro_file = os.path.join(
        get_package_share_directory('limo_description'),
        'urdf', 'limo_four_diff.xacro'
    )
    robot_description = Command(['xacro ', xacro_file])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )  # :contentReference[oaicite:1]{index=1}

    # 3) Spawn the LIMO into the running Gazebo instance
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic',  'robot_description',
            '-entity', 'limo',
            '-x',      '0.0',
            '-y',      '0.0',
            '-z',      '0.1'
        ],
        output='screen'
    )  # :contentReference[oaicite:2]{index=2}

    # 3) RViz2
    rviz_config_file = os.path.join(
        get_package_share_directory('limo_bringup'),
        'rviz','urdflimon_navigation.rviz'   # point to your .rviz file
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world)
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_bt_xml_cmd)
    ld.add_action(declare_use_remappings_cmd)
    ld.add_action(start_localization_cmd)
    ld.add_action(start_navigation_cmd)
    ld.add_action(start_lifecycle_manager_cmd)
    ld.add_action(gazebo)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_entity)
    ld.add_action(rviz)
    return ld
    

if __name__ == '__main__':
    generate_launch_description()
