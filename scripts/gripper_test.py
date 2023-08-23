# Imports
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from robot_demonstrator.Motion import *
from robot_demonstrator.transformations import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Define robot
robot = ABB_IRB1200("192.168.125.1")


for i in range(10):

    print("Pick")
    robot.con.set_dio(1)

    time.sleep(2)

    print("Place")
    robot.con.set_dio(0)

    time.sleep(2)
