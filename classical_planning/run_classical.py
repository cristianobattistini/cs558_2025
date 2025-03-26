import pybullet as p
import pybullet_data
import time
import os


# 1) Connect to PyBullet
p.connect(p.GUI)

# 2) Add PyBullet's default data (plane, etc.) to search path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 5) Load the ground plane
plane_id = p.loadURDF("plane.urdf")


# 3) The directory of this script: "CS558_2025/classical_planning"
current_dir = os.path.dirname(__file__)

# 4) Add your "assets/a1" folder to PyBullet's search path
#    so that any relative paths (like "../meshes/") in the URDF can be resolved.
p.setAdditionalSearchPath(os.path.join(current_dir, '../assets/a1'))


# 6) Construct the path to a1.urdf (inside "assets/a1/urdf/")
a1_urdf_path = os.path.join(current_dir, '../assets/a1/urdf/a1.urdf')

# 7) Load the A1 robot at the start position/orientation
start_position = [0, 0, 0.48]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(a1_urdf_path, start_position, start_orientation)

# 8) Create a dynamic obstacle (cube)
col_cube_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
vis_cube_id = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1]
)
cube_id = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=col_cube_id,
    baseVisualShapeIndex=vis_cube_id,
    basePosition=[2, 2, 0.5]
)

# 9) Set an initial velocity for the obstacle
p.resetBaseVelocity(cube_id, linearVelocity=[-1, 0, 0])

# 10) Run the simulation
for _ in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

