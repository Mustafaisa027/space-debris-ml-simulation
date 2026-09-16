from setuptools import setup

package_name = 'zed_obstacle_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gelidonya_2',
    maintainer_email='gelidonya_2@example.com',
    description='ZED2i point cloud based obstacle map and sector avoidance nodes',
    license='MIT',
    tests_require=['pytest'],
   entry_points={
    'console_scripts': [
        'obstacle_map_node = zed_obstacle_map.obstacle_map_node:main',
        'sector_avoidance_node = zed_obstacle_map.sector_avoidance_node:main',
        'pointcloud_filter_node = zed_obstacle_map.pointcloud_filter_node:main',
        'obstacle_bridge_node = zed_obstacle_map.obstacle_bridge_node:main',
    ],
},
)
