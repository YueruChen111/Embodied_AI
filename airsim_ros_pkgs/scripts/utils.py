import uuid
import numpy as np
import math
from pyquaternion import Quaternion


def quaternionr2quaternion(quaternionr):
    return Quaternion(quaternionr.w_val, quaternionr.x_val, quaternionr.y_val, quaternionr.z_val)


def vector3r2numpy(vector):
    return np.array([vector.x_val, vector.y_val, vector.z_val])


def generate_unique_token():
    """
    return a 32 bit unique token
    """
    return uuid.uuid4().hex


def euler2quaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5 * np.pi / 180.0)
    sy = np.sin(yaw * 0.5 * np.pi / 180.0)
    cp = np.cos(pitch * 0.5 * np.pi / 180.0)
    sp = np.sin(pitch * 0.5 * np.pi / 180.0)
    cr = np.cos(roll * 0.5 * np.pi / 180.0)
    sr = np.sin(roll * 0.5 * np.pi / 180.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


def quaternion2euler(rotation):
    w, x, y, z = rotation[0], rotation[1], rotation[2], rotation[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def test():
    yaw, pitch, roll = 12, 13, 14
    q = Quaternion(euler2quaternion(yaw, pitch, roll))
    print(roll * np.pi / 180, pitch * np.pi / 180, yaw * np.pi / 180)
    print(quaternion2euler(list(q)))


if __name__ == '__main__':
    test()