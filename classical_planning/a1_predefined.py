import pybullet as p
import time
import pybullet_data


# class Dog:
p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)              # Hide side debug sliders
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

# Add PyBullet's default data (plane, etc.) to search path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.81)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("../assets/a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)

#enable collision between lower legs
for j in range (p.getNumJoints(quadruped)):
        print(p.getJointInfo(quadruped,j))

lower_legs = [2,5,8,11]
for l0 in lower_legs:
    for l1 in lower_legs:
        if (l1>l0):
            enableCollision = 1
            print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
            p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

jointIds=[]
paramIds=[]

maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)



joints=[]

while(1):
    with open("mocap.txt","r") as filestream:
        for line in filestream:
            maxForce = p.readUserDebugParameter(maxForceId)
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            joints=currentline[2:14]
            for j in range (12):
                targetPos = float(joints[j])
                p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)

            p.stepSimulation()
            for lower_leg in lower_legs:
                pts = p.getContactPoints(quadruped,-1, lower_leg)

            time.sleep(1./500.)

