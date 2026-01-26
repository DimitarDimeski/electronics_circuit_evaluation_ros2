from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='electronics_circuit_evaluation',
            executable='detection_node',
            name='detection_node'
        ),
        Node(
            package='electronics_circuit_evaluation',
            executable='orientation_node',
            name='orientation_node'
        ),
        Node(
            package='electronics_circuit_evaluation',
            executable='connectivity_node',
            name='connectivity_node'
        ),
        Node(
            package='electronics_circuit_evaluation',
            executable='evaluation_node',
            name='evaluation_node'
        ),
    ])
