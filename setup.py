from distutils.core import setup

setup(
    name='robot_demonstrator',
    version='1.0.0',
    url='https://github.com/MatthiasDR96/robot_demonstrator',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='Software for the robot demonstrator of KU Leuven Bruges',
    packages=['robot_demonstrator'],
    package_dir={'': 'src'},
    install_requires=['scipy', 'pyrealsense2', 'opencv-python', 'numpy', 'matplotlib', 'pandas']
)