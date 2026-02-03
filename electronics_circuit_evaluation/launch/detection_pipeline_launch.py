from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        # Image Publisher Node
        Node(
            package='electronics_circuit_evaluation',
            executable='image_publisher_node',
            name='image_publisher_node',
            output='screen'
        ),
        
        # Detection Node
        Node(
            package='electronics_circuit_evaluation',
            executable='detection_node',
            name='detection_node',
            output='screen'
        ),

        # Orientation Node
        Node(
            package='electronics_circuit_evaluation',
            executable='orientation_node',
            name='orientation_node',
            output='screen'
        ),
        
        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])
