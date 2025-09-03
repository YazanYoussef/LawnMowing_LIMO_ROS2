from setuptools import find_packages, setup
from pathlib import Path

package_name = 'waypoint_follower_pkg'

# Collect ALL non-.py files under waypoint_follower_pkg/model/** to include in wheel
pkg_root = Path(__file__).parent / package_name
model_dir = pkg_root / 'TRATSS_Model'
extra_files = []
if model_dir.exists():
    for p in model_dir.rglob('*'):
        if p.is_file() and p.suffix not in ('.py', '.pyc', '.pyo'):
            # path must be relative to the package root for package_data
            extra_files.append(str(p.relative_to(pkg_root)))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={package_name: extra_files},
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='navinst',
    maintainer_email='20yy18@queensu.ca',
    description='Package for Gazebo visualization tasks',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'waypoint_follower = waypoint_follower_pkg.waypoint_follower:main',
        ],
    },
)
