import pandas as pd
import mujoco
import numpy as np
from scipy.linalg import pinv
import time

amplitude = 0.4
max_velocity = 0.4


def cartesian_trajectory(t):
    x0, y0, z0 = 0.5, 1, 0.5
    x = x0 + amplitude * np.sin(t)
    z = z0 + amplitude * np.sin(t + np.pi)

    return np.array([x, y0, z])


model = mujoco.MjModel.from_xml_path("../../homework_1/robot_arm.xml")
data = mujoco.MjData(model)

end_effector_id = model.body('tool').id

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))


def inverse_kinematics(cartesian_position, end_effector_id, max_iterations=100, tolerance=1e-4):
    error = np.subtract(cartesian_position, data.body(end_effector_id).xpos)

    mujoco.mj_jac(model, data, jacp, jacr, cartesian_position, end_effector_id)

    n = jacp.shape[1]
    I = np.identity(n)
    product = jacp.T @ jacp + 0.15 * I

    if np.isclose(np.linalg.det(product), 0):
        j_inv = np.linalg.pinv(product) @ jacp.T
    else:
        j_inv = np.linalg.inv(product) @ jacp.T

    delta_q = j_inv @ error

    # Compute next step
    q = data.qpos.copy()
    q += 0.5 * delta_q

    return q


joint_positions = []
joint_velocities = []
joint_accelerations = []
cartesian_positions = []
cartesian_velocities = []
cartesian_accelerations = []
torques = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        cartesian_position = cartesian_trajectory(data.time)

        joint_position = inverse_kinematics(
            cartesian_position, end_effector_id
        )

        data.qpos[:] = joint_position
        mujoco.mj_step(model, data)

        joint_positions.append(data.qpos[:].copy())
        joint_velocities.append(data.qvel[:].copy())
        joint_accelerations.append(data.qacc[:].copy())
        cartesian_positions.append(cartesian_position)

        cartesian_velocities.append(np.zeros(3))
        cartesian_accelerations.append(np.zeros(3))
        torques.append(data.qfrc_actuator[:].copy())

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

data = {
    "time": time,
    "joint_positions": joint_positions,
    "joint_velocities": joint_velocities,
    "joint_accelerations": joint_accelerations,
    "cartesian_positions": cartesian_positions,
    "cartesian_velocities": cartesian_velocities,
    "cartesian_accelerations": cartesian_accelerations,
    "torques": torques
}

df = pd.DataFrame(data)
