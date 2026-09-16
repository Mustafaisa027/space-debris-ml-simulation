import os
from glob import glob
from setuptools import setup

package_name = 'mission_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gelidonya_2',
    maintainer_email='gelidonya_2@todo.todo',
    description='Gelidonya IDA gorev yoneticisi (3 parkur durum makinesi, '
                'kamikaze, IHA renk alici, log)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Gorev dugumleri
            'mission_manager_node = mission_control.mission_manager_node:main',
            'kamikaze_node = mission_control.kamikaze_node:main',
            'iha_color_receiver_node = mission_control.iha_color_receiver_node:main',
            # Veri teslimi (sartname Dosya 1/2/3)
            'mission_logger_node = mission_control.mission_logger_node:main',
            'blackbox_node = mission_control.blackbox_node:main',
            'telemetry_logger_node = mission_control.telemetry_logger_node:main',
            'video_recorder_node = mission_control.video_recorder_node:main',
            # Test / simulasyon
            'sim_mavros_node = mission_control.sim_mavros_node:main',
            'sim_target_node = mission_control.sim_target_node:main',
        ],
    },
)
