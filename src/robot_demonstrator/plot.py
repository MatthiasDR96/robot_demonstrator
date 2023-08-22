# Imports
import matplotlib.pyplot as plt
import numpy as np


def plot_frame(x, y, z, r, ax):
    axis_length = 100
    pose_ix = np.dot(np.array(r), np.array([axis_length, 0, 0]))
    pose_iy = np.dot(np.array(r), np.array([0, axis_length, 0]))
    pose_iz = np.dot(np.array(r), np.array([0, 0, axis_length]))
    ax.plot([x, pose_ix[0] + x], [y, pose_ix[1] + y], [z, pose_ix[2] + z], 'r', linewidth=2)
    ax.plot([x, pose_iy[0] + x], [y, pose_iy[1] + y], [z, pose_iy[2] + z], 'g', linewidth=2)
    ax.plot([x, pose_iz[0] + x], [y, pose_iz[1] + y], [z, pose_iz[2] + z], 'b', linewidth=2)


def plot_frame_t(t, ax, text=''):
    axis_length = 100
    r = t[0:3, 0:3]
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    pose_ix = np.dot(r, np.array([axis_length, 0, 0]))
    pose_iy = np.dot(r, np.array([0, axis_length, 0]))
    pose_iz = np.dot(r, np.array([0, 0, axis_length]))
    ax.plot(x + [0, pose_ix[0]], y + [0, pose_ix[1]], z + [0, pose_ix[2]], 'r', linewidth=2)
    ax.plot(x + [0, pose_iy[0]], y + [0, pose_iy[1]], z + [0, pose_iy[2]], 'g', linewidth=2)
    ax.plot(x + [0, pose_iz[0]], y + [0, pose_iz[1]], z + [0, pose_iz[2]], 'b', linewidth=2)
    pose_t = np.dot(r, np.array([0.3 * axis_length, 0.3 * axis_length, 0.3 * axis_length]))
    ax.text(x + pose_t[0], y + pose_t[1], z + pose_t[2], text, fontsize=11)


def plot_transf_p(tref, ttransf, ax):
    t = np.dot(tref, ttransf)
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    x0 = tref[0][3]
    y0 = tref[1][3]
    z0 = tref[2][3]
    ax.plot([x0, x], [y0, y], [z0, z], 'k:', linewidth=1)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]
    y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]
    z_mean = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])
