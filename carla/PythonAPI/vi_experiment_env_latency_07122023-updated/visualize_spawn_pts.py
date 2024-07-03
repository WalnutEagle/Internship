import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import global_planner

PATH = Path(
    '/home/tzm/Projects/2_Lab/carla_910/PythonAPI/vi_experiment_env/Town05_paths/')

data = np.load(PATH/'Town05_spawn_points.npy', allow_pickle=True)
path = np.load(PATH/'path_points_t05_7_100_1.npy', allow_pickle=True)

planner = global_planner.Planner(
    '/home/tzm/Projects/2_Lab/carla_910/PythonAPI/vi_experiment_env/Town05_paths/path_points_t05_7_100_1.npy')

fig, ax = plt.subplots()

ax.plot(planner.path_pts[:,2], 'x')
ax.plot(58,planner.path_pts[58,2], 'rx')
ax.plot(61,planner.path_pts[61,2], 'rx')

fig2, ax2 = plt.subplots()
ax2.plot(planner.path_pts[:,0], planner.path_pts[:,1], '.-')
ax2.plot(planner.path_pts[58,0], planner.path_pts[58,1], 'rx')
ax2.plot(planner.path_pts[61,0], planner.path_pts[61,1], 'rx')



print(planner.path_pts[58,0], planner.path_pts[58,1])
print(planner.path_pts[61,0], planner.path_pts[61,1])
# plt.plot(data[:, 0], data[:, 1], '.')
# plt.plot(path[:, 0], path[:, 1], 'rx-')
# plt.plot(-111.55669403076172, -78.61675262451172, 'bx')
# plt.plot(path[59, 0], path[59, 1], 'bo')
# plt.plot(path[61, 0], path[61, 1], 'ro')
# for idx, d in enumerate(data):
#     plt.text(d[0], d[1], str(idx), color="red", size=7)
plt.show()
