from robot_demonstrator.transformations import *


# Convert to transformation matrix
T_bt = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                    [0, 0, -1, 60],
                    [ 0, 0, 0, 1]])

print(T_bt[:3,:3])

print(quat_from_r(T_bt[:3,:3]))

print(r_from_quat(1, 0, 0, 0))