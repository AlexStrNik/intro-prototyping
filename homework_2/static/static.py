import itertools
import mujoco
import mujoco.viewer
import numpy as np
import tqdm
import pandas as pd

model = mujoco.MjModel.from_xml_path('../../homework_1/robot_arm.xml')
data = mujoco.MjData(model)

angles = np.arange(0, 360, 25)
configurations = list(itertools.product(angles, repeat=model.nq))
pos_list = [np.deg2rad(config) for config in configurations]

joints = ['pitch', 'roll', 'yaw', 'elbow']
results = {
    key: [] for key in joints
}

for pos in tqdm.tqdm(pos_list):
    data.qpos[:] = pos
    data.qvel[:] = 0
    data.qacc[:] = 0

    mujoco.mj_inverse(model, data)
    mujoco.mj_forward(model, data)

    if data.contact:
        pass

    for torque_ind, torque in enumerate(data.qfrc_inverse):
        results[joints[torque_ind]].append(torque)

df = pd.DataFrame(results, columns=joints)
df.reset_index(drop=True, inplace=True)
df.to_csv('static.csv', index=False)
