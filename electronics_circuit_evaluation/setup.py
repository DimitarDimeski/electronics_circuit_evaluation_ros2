from setuptools import find_packages, setup

package_name = 'electronics_circuit_evaluation'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/evaluation_pipeline_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dimitar',
    maintainer_email='dimitar.dimeski23@gmail.com',
    description='Nodes for electronics circuit evaluation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = electronics_circuit_evaluation.detection_node:main',
            'orientation_node = electronics_circuit_evaluation.orientation_node:main',
            'connectivity_node = electronics_circuit_evaluation.connectivity_node:main',
            'evaluation_node = electronics_circuit_evaluation.evaluation_node:main',
            'image_publisher_node = electronics_circuit_evaluation.image_publisher_node:main',
        ],
    },
)
