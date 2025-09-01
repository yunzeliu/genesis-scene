import numpy as np
import genesis as gs
import argparse

def quat_geodesic_angle(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    d = abs(np.dot(q1, q2))
    d = np.clip(d, 0.0, 1.0)
    d = 2.0 * np.arccos(d)
    return np.min([d, 2 * np.pi - d])

def quat_to_rotmat(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion in [w, x, y, z] format.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Notes
    -----
    Assumes right-handed coordinate system and active rotation.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Quaternion must be shape (4,), given {}".format(q.shape))

    # Normalize to guard against drift
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        raise ValueError("Zero-norm quaternion is not valid")
    q = q / np.sqrt(n)

    w, x, y, z = q

    # Precompute products
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

    return R

class TaskEnv:
    def __init__(self):
        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=5e-3,
                    substeps=15,
                ),
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(3, -1, 1.5),
                    camera_lookat=(0.0, 0.0, 0.0),
                    camera_fov=30,
                    max_FPS=60,
                ),
                show_viewer=True
            )

        # self.franka = self.scene.add_entity(
        #         gs.morphs.URDF(file="embodiments/franka-panda/panda.urdf", fixed=True, merge_fixed_links=False),
        #         material=gs.materials.Rigid(coup_friction=1.0),
        #     )
        # self.hand_link_name = "panda_hand"
        self.franka = self.scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.1)),
                material=gs.materials.Rigid(coup_friction=1.0),
            )
        self.hand_link_name = "hand"

        self.plane = self.scene.add_entity(
                gs.morphs.Plane(),
            )
        
        self.targ_pos = np.array([0.45, -0.05, 0.1])
        self.targ_quat = np.array([1.0, 0.0, 0.0, 1.0])
        self.targ_quat /= np.linalg.norm(self.targ_quat)

        self.t_shape = self.scene.add_entity(gs.morphs.Mesh(file = "T-shape-modified.obj", pos=(0.65, -0.05, 0.1), scale=0.5), surface=gs.surfaces.Default(color=(0.6, 0.7, 0.8)))
        self.marker = self.scene.add_entity(gs.morphs.Mesh(file = "T-shape-modified.obj", pos=self.targ_pos - np.asarray([0.0, 0.0, 0.049]), quat=self.targ_quat, scale=0.5, collision=False, fixed=True), surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)))
        self.table = self.scene.add_entity(gs.morphs.Box(size=(2.0, 2.0, 0.1), pos=(0.0, 0.0, 0.05), fixed=True), material=gs.materials.Rigid(friction=1.0), surface=gs.surfaces.Default(color=(0.8, 0.7, 0.6)))

        self.key_point_poses = [
            np.array([0.0, 0.075, 0.025]),
            np.array([0.05, 0.075, 0.025]),
            np.array([0.075, 0.05, 0.025]),
            np.array([0.05, 0.025, 0.025]),
            np.array([0.025, 0.0, 0.025]),
            np.array([0.025, -0.05, 0.025]),
            np.array([0.0, -0.075, 0.025]),
            np.array([-0.025, -0.05, 0.025]),
            np.array([-0.025, 0.0, 0.025]),
            np.array([-0.05, 0.025, 0.025]),
            np.array([-0.075, 0.05, 0.025]),
            np.array([-0.05, 0.075, 0.025]),
        ]

        self.key_points = [
            self.scene.add_entity(gs.morphs.Sphere(radius=0.01, collision=False, fixed=True), surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0), opacity=0.5)) for kp in self.key_point_poses
        ]

        self.targ_key_points = [
            self.scene.add_entity(gs.morphs.Sphere(radius=0.01, collision=False, fixed=True), surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0), opacity=0.5)) for kp in self.key_point_poses
        ]
        self.targ_point = self.scene.add_entity(gs.morphs.Sphere(radius=0.1, collision=False, fixed=True), surface=gs.surfaces.Default(color=(1.0, 1.0, 0.0), opacity=0.5))

        self.scene.build()

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)

        # Optional: set control gains
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        self.end_targ_pos = np.array([0.45, 0.0, 0.3])

        self.end_effector = self.franka.get_link(self.hand_link_name)
        # move to pre-grasp pose
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.end_targ_pos,
            quat=np.array([0, 1, 0, 0]),
        )
        qpos[-2:] = 0.0
        self.franka.set_dofs_position(qpos[:-2], self.motors_dof)
        self.franka.set_dofs_position(qpos[-2:], self.fingers_dof)

        # grasp with 1N force
        self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
        self.franka.control_dofs_force(np.array([-1, -1]), self.fingers_dof)
        # franka.control_dofs_position(np.array([0, 0]), fingers_dof) # you can also use position control

    def combined_dist(dpos, dquat):
        return dpos * dpos * 100 + dquat * dquat

    def targ_dist(self):
        pos=self.targ_pos
        quat=self.targ_quat

        tpos = self.t_shape.get_pos().cpu().numpy()
        tquat = self.t_shape.get_quat().cpu().numpy()
        dpos = np.linalg.norm(pos - tpos)
        dquat = quat_geodesic_angle(quat, tquat)
        return dpos, dquat
    
    def reward(self):
        dpos, dquat = self.targ_dist()
        return TaskEnv.combined_dist(dpos, dquat)

    def step(self):
        # get R from t_shape.get_quat with numpy functions
        R = quat_to_rotmat(self.t_shape.get_quat().cpu().numpy())
        targ_R = quat_to_rotmat(self.targ_quat)
        for i in range(12):
            self.key_points[i].set_pos(self.t_shape.get_pos().cpu().numpy() + R @ self.key_point_poses[i])
            self.targ_key_points[i].set_pos(self.targ_pos + targ_R @ self.key_point_poses[i])
        self.targ_point.set_pos(self.end_targ_pos)
        self.scene.step()
        r = self.targ_dist()
        print("targ dist:", r)
        return r

    def run(self):
        for i in range(100):
            self.step()

        i = 0
        stage = 0
        stage_step = 0
        last_dpos, last_dquat = self.targ_dist()

        # lift
        while last_dpos > 0.005 or last_dquat > 0.05:
            if stage == 0:
                tpos = self.t_shape.get_pos().cpu().numpy()
                R = quat_to_rotmat(self.t_shape.get_quat().cpu().numpy())
                targ_R = quat_to_rotmat(self.targ_quat)
                targ_dist = -1.0
                sel_targ_key_pos = None
                sel_cur_key_pos = None
                for i in range(12):
                    cur_key_pos = tpos + R @ self.key_point_poses[i]
                    targ_key_pos = self.targ_pos + targ_R @ self.key_point_poses[i]
                    dist = np.linalg.norm(cur_key_pos - targ_key_pos)
                    if dist > targ_dist:
                        targ_dist = dist
                        sel_targ_key_pos = targ_key_pos
                        sel_cur_key_pos = cur_key_pos
                self.push_direction = sel_targ_key_pos - sel_cur_key_pos
                self.push_direction /= np.linalg.norm(self.push_direction)
                start_point = sel_cur_key_pos - 0.05 * self.push_direction
                start_point[2] = 0.3
                print(f"start_point: {start_point}")
                self.end_targ_pos = start_point
                stage = 1
                stage_step = 0
            elif stage == 1:
                if stage_step == 50:
                    self.end_targ_pos[2] = 0.245
                    stage = 2
                    stage_step = 0
            elif stage == 2:
                if stage_step == 50:
                    stage = 3
                    stage_step = 0
            elif stage == 3:
                if self.reward() < 1.0:
                    self.end_targ_pos += 0.001 * self.push_direction
                else:
                    self.end_targ_pos += 0.005 * self.push_direction
            elif stage == 4:
                if stage_step == 50:
                    stage = 0
                    stage_step = 0
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=self.end_targ_pos,
                quat=np.array([0, 1, 0, 0]),
            )
            self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
            self.step()
            i += 1
            stage_step += 1

            new_dpos, new_dquat = self.targ_dist()
            if stage == 3 and stage_step > 30 and TaskEnv.combined_dist(new_dpos, new_dquat) >= TaskEnv.combined_dist(last_dpos, last_dquat):
                self.end_targ_pos[2] = 0.3
                self.end_targ_pos -= self.push_direction * 0.03
                stage = 4
                stage_step = 0

            last_dpos, last_dquat = new_dpos, new_dquat
            print(f"stage: {stage}, step: {stage_step}, reward: {self.reward()}")

        for i in range(100):
            self.step()


def main():
    env = TaskEnv()
    env.run()

if __name__ == "__main__":
    main()