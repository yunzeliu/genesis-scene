import numpy as np
import genesis as gs
import argparse

gs.init(backend=gs.gpu)

scene = gs.Scene(
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

# franka = scene.add_entity(
#         gs.morphs.URDF(file="embodiments/franka-panda/panda.urdf", fixed=True, merge_fixed_links=False),
#         material=gs.materials.Rigid(coup_friction=1.0),
#     )
# hand_link_name = "panda_hand"
franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=1.0),
    )
hand_link_name = "hand"

plane = scene.add_entity(
        gs.morphs.Plane(),
    )

t_shape = scene.add_entity(gs.morphs.Mesh(file = "T-shape-modified.obj", pos=(0.75, -0.05, 0.0), scale=0.5))

scene.build()

if True:
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    print(franka)

    end_effector = franka.get_link(hand_link_name)
    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.64, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.03
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)

    # grasp with 1N force
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-1, -1]), fingers_dof)
    # franka.control_dofs_position(np.array([0, 0]), fingers_dof) # you can also use position control

    for i in range(100):
        scene.step()

    # lift
    for i in range(300):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.64 + 0.0005 * i, 0.0, 0.135]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        scene.step()

    for i in range(100):
        scene.step()
