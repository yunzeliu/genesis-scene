import numpy as np
import genesis as gs
import argparse

def quat_geodesic_angle(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    d = abs(np.dot(q1, q2))
    d = np.clip(d, 0.0, 1.0)
    return 2.0 * np.arccos(d)

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

        self.t_shape = self.scene.add_entity(gs.morphs.Mesh(file = "T-shape-modified.obj", pos=(0.65, -0.05, 0.1), scale=0.5), surface=gs.surfaces.Default(color=(0.6, 0.7, 0.8)))
        self.marker = self.scene.add_entity(gs.morphs.Mesh(file = "T-shape-modified.obj", pos=(0.45, -0.05, 0.051), quat=(1, 0, 0, 1), scale=0.5, collision=False, fixed=True), surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)))
        self.table = self.scene.add_entity(gs.morphs.Box(size=(2.0, 2.0, 0.1), pos=(0.0, 0.0, 0.05), fixed=True), material=gs.materials.Rigid(friction=1.0), surface=gs.surfaces.Default(color=(0.8, 0.7, 0.6)))

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

        self.end_effector = self.franka.get_link(self.hand_link_name)
        # move to pre-grasp pose
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([0.75, -0.03, 0.235]),
            quat=np.array([0, 1, 0, 0]),
        )
        qpos[-2:] = 0.0
        self.franka.set_dofs_position(qpos[:-2], self.motors_dof)
        self.franka.set_dofs_position(qpos[-2:], self.fingers_dof)

        # grasp with 1N force
        self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
        self.franka.control_dofs_force(np.array([-1, -1]), self.fingers_dof)
        # franka.control_dofs_position(np.array([0, 0]), fingers_dof) # you can also use position control

    def step(self):
        self.scene.step()
        pos=np.array([0.45, -0.05, 0.1])
        quat=np.array([1.0, 0.0, 0.0, 1.0])
        quat /= np.linalg.norm(quat)

        tpos = self.t_shape.get_pos().cpu().numpy()
        tquat = self.t_shape.get_quat().cpu().numpy()
        dpos = pos - tpos
        dquat = quat_geodesic_angle(quat, tquat)
        print ("pos diff:", np.linalg.norm(dpos), "quat diff:", dquat)

    def run(self):
        for i in range(100):
            self.step()

        # lift
        for i in range(500):
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([0.75 - 0.0005 * i, -0.03, 0.235]),
                quat=np.array([0, 1, 0, 0]),
            )
            self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
            self.step()

        for i in range(100):
            self.step()


def main():
    env = TaskEnv()
    env.run()

if __name__ == "__main__":
    main()