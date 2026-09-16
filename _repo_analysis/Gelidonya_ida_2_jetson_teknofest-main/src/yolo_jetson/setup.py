from setuptools import setup

package_name = 'yolo_jetson'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gelidonya_2',
    maintainer_email='gelidonya_2@todo.todo',
    description='ZED2i RGB uzerinde YOLO ile hedef tespiti (Parkur-3)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed_yolo_node = yolo_jetson.zed_yolo_node:main',
        ],
    },
)
