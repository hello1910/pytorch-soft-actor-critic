  
"""tack of blocks on a tabletop is pushed by a gripper."""
from gym.utils import seeding
from gym import spaces
import pickle
import gym
import os
import numpy as np
import pybullet as p
import pybullet_data
from rotations import *
import math



def stable_func(row):
    #print(row.shape)
    first=False
    second=False
    w_and_l=row[:,0:2]
    x_and_y=row[:,3:5]
    #checking (1,2)
    #x range
    botx=(x_and_y[0,0]- w_and_l[0,0]/2, x_and_y[0,0] + w_and_l[0,0]/2)
    boty=(x_and_y[0,1]- w_and_l[0,1]/2, x_and_y[0,1] + w_and_l[0,1]/2)

    top_x=x_and_y[1,0]
    top_y=x_and_y[1,1]


    if not (botx[1]>top_x>botx[0]) or not (boty[1]> top_y>boty[0]):
        first=True


    botx=(x_and_y[1,0]- w_and_l[1,0]/2, x_and_y[1,0] + w_and_l[1,0]/2)
    boty=(x_and_y[1,1]- w_and_l[1,1]/2, x_and_y[1,1] + w_and_l[1,1]/2)

    top_x=x_and_y[2,0]
    top_y=x_and_y[2,1]

    if not (botx[1]>top_x>botx[0]) or not (boty[1]> top_y>boty[0]):
        second=True
    #print(first,second)
    return first,second

class BlockStackPushEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, num_blocks_in_stack=3, num_distractor_blocks=0, episode_length=10, sim_steps_per_action=5, 
                 goal_threshold=0.005, seed=0, pos_only=False, camera_vantage='side'):
        self.num_blocks_in_stack = num_blocks_in_stack
        self.num_distractor_blocks = num_distractor_blocks
        self.episode_length = episode_length
        self.sim_steps_per_action = sim_steps_per_action
        self.goal_threshold = goal_threshold
        self.camera_vantage = camera_vantage
        self._seed = seed
        self.pos_only = pos_only
        self.seed(seed=seed)

        self.setup()

    def setup(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_initial_state(self):
        """The state is num_blocks x num_attributes. Blocks in stack are first, then distractors."""
        state = []

        # For now, constant orientation (quaternion) for all blocks.
        orn_x, orn_y, orn_z, orn_w = 0., 0., 0., 1.

        # Block stack blocks. For now, always start the block stack at (0, 0, 0).
        x, y = 0., 0.
        previous_block_top = 0.
        for _ in range(self.num_blocks_in_stack):
            w, l, h, mass, friction = self.sample_block_static_attributes()
            z = previous_block_top + h/2.
            previous_block_top += h
            attributes = [w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction]
            state.append(attributes)

        # Distractor blocks. Sampled away from stack.
        for _ in range(self.num_distractor_blocks):
            distance_from_stack = self.np_random.uniform(0.2, 0.5)
            angle_from_stack = self.np_random.uniform(0, 2*np.pi)
            x, y = distance_from_stack * np.cos(angle_from_stack), distance_from_stack * np.sin(angle_from_stack)
            w, l, h, mass, friction = self.sample_block_static_attributes()
            z = h/2.
            attributes = [w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction]
            state.append(attributes)
        state=pickle.load(open("save_numpy_array.p","rb"))[1]
        # Goal
        distance_from_stack = self.np_random.uniform(0.1, 0.3)
        angle_from_stack = self.np_random.uniform(-np.pi/4, np.pi/4)
        x, y = distance_from_stack * np.cos(angle_from_stack), distance_from_stack * np.sin(angle_from_stack)
        z = 0
        goal = [x, y, z, orn_x, orn_y, orn_z, orn_w]
        goal[0]=0.60
        goal[1]=0.30
        return np.array(state, dtype=np.float32),np.array(goal, dtype=np.float32)

    def sample_block_static_attributes(self):
        w, l, h = self.np_random.normal(0.1, 0.01, size=(3,))
        mass = self.np_random.uniform(0.05, 0.2)
        friction = self.np_random.uniform(0.1, 1.0)

        return w, l, h, mass, friction

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        raise NotImplementedError()

    def set_state(self, state, goal):
        raise NotImplementedError()
    

    def get_state(self):
        raise NotImplementedError()



class PybulletBlockStackPushEnv(BlockStackPushEnv):

    def setup(self):
        self.physics_client_id = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=self.physics_client_id)

        # Load the constant objects
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", (0, 0, -1), (0., 0., 0., 1.), useFixedBase=True, physicsClientId=self.physics_client_id)
        p.loadURDF("table/table.urdf", (0., 0., -0.75), (0., 0., 0., 1.), useFixedBase=True, physicsClientId=self.physics_client_id)
        self.gripper_start_pos = (-0.3, 0, -0.1)
        self.gripper_orientation = (0., 1., 0., 1.)
        self.gripper_id = p.loadURDF("pr2_gripper.urdf", self.gripper_start_pos, physicsClientId=self.physics_client_id)
        self.gripper_constraint_id = p.createConstraint(self.gripper_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], 
            self.gripper_start_pos, childFrameOrientation=self.gripper_orientation, physicsClientId=self.physics_client_id)

        # Set gravity
        p.setGravity(0., 0., -10., physicsClientId=self.physics_client_id)

        # Record the initial state so we can reset to it later without having to reload the constant objects
        self.initial_state_id = p.saveState(physicsClientId=self.physics_client_id)
        self.block_ids = []
        self.goal_block_id = None
    def disconnect(self):
        p.disconnect()
    def set_state(self, state, goal):
        """The state is num_blocks x num_attributes."""

        # Blocks are always recreated on reset because their size, mass, friction changes
        self.static_block_attributes = {}
        self.block_ids = []

        for block_state in state:
            block_id = self.create_block(block_state)
            self.block_ids.append(block_id)

        # Create the goal
        x, y, z, orn_x, orn_y, orn_z, orn_w = goal
        self.goal_state = np.array([x, y, z, orn_x, orn_y, orn_z, orn_w])
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=(1, 0, 0, 0.75), 
            physicsClientId=self.physics_client_id)
        self.goal_block_id = p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=[x, y, z], 
            baseOrientation=[orn_x, orn_y, orn_z, orn_w], physicsClientId=self.physics_client_id)

        # Let the world run for a bit
        for _ in range(40):
            p.stepSimulation(physicsClientId=self.physics_client_id)
    def get_full_gripper(self,state):
       o1,p1=self.get_gripper_state()
       inty=np.array([0,0,0,0,0])
       o1=o1.reshape((3,1))
       p1=p1.reshape((4,1))
       inty=inty.reshape((5,1))
       ans=np.vstack((np.vstack((o1,p1)),inty))
       ans=ans.reshape((1,12))
       return np.vstack((state,ans))
    def get_state(self):
        """The state is num_blocks x num_attributes. Blocks in stack are first, then distractors."""
        state = []

        for block_id in self.block_ids:
            block_attributes = self.get_block_attributes(block_id)
            if self.pos_only:
                state.append(block_attributes[3:6])
            else:
                state.append(block_attributes)                
   #     print(np.array(state).shape, "shapeinstate")
        return self.get_full_gripper(np.array(state, dtype=np.float32)) , self.goal_state.copy()

    def get_gripper_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.gripper_id, physicsClientId=self.physics_client_id)

        return np.array(pos, dtype=np.float32),np.array(orn,dtype=np.float32)

    def get_block_attributes(self, block_id):
        w, l, h, mass, friction = self.static_block_attributes[block_id]

        (x, y, z), (orn_x, orn_y, orn_z, orn_w) = p.getBasePositionAndOrientation(block_id, 
            physicsClientId=self.physics_client_id)

        attributes = [w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction]
        return attributes

    def reset(self):
        self.episode_idx = 0

        # TODO: time whether this is actually faster than just rebuilding the whole scene
        for block_id in self.block_ids:
            p.removeBody(block_id, physicsClientId=self.physics_client_id)
        if self.goal_block_id is not None:
            p.removeBody(self.goal_block_id, physicsClientId=self.physics_client_id)

        p.restoreState(stateId=self.initial_state_id, physicsClientId=self.physics_client_id)
        p.changeConstraint(self.gripper_constraint_id, self.gripper_start_pos, maxForce=500, physicsClientId=self.physics_client_id)

        initial_state,goal= self.sample_initial_state()
        self.set_state(initial_state, goal)

        action_list=[[0.117706667, -4.81789825e-06, 0.0], [0.053038837, -4.99072241e-06, 0.0], [0.130527135, -3.23587253e-06, 0.0], [0.0210778331, 1.30573375e-06, 0.0], [0.134124202, 2.0407941e-05, 0.0], [0.0219372237, 4.95898039e-06, 0.0], [0.0579635579, 1.63016369e-05, 0.0], [0.0557842973, 1.76268891e-05, 0.0], [0.028251502, 9.63342528e-06, 0.0], [0.0824221889, 2.92031703e-05, 0.0], [0.0435994307, 1.57883286e-05, 0.0], [0.129925998, 4.78468989e-05, 0.0]]
        for i in range(11):
           _,_,_,_=self.step(action_list[i])
        return self.get_state()

    def reset_and_set_state(self, state, goal, gripper_state):
        # TODO: time whether this is actually faster than just rebuilding the whole scene
        for block_id in self.block_ids:
            p.removeBody(block_id, physicsClientId=self.physics_client_id)
        if self.goal_block_id is not None:
            p.removeBody(self.goal_block_id, physicsClientId=self.physics_client_id)

        p.restoreState(stateId=self.initial_state_id, physicsClientId=self.physics_client_id)

        p.changeConstraint(self.gripper_constraint_id, gripper_state, maxForce=500, physicsClientId=self.physics_client_id)
        self.set_state(state, goal)

    def step(self, action):
        self.episode_idx += 1

        incremental_action = np.array(action) / self.sim_steps_per_action

        for _ in range(self.sim_steps_per_action):
            self.apply_action(incremental_action)
            p.stepSimulation(physicsClientId=self.physics_client_id)
            state, goal = self.get_state()
        reward = self.get_reward(state, goal)
        done = (self.episode_idx >= self.episode_length)

        return state, reward, done, {'goal' : goal}

    def inside_step(self, action):
        incremental_action = np.array(action) / self.sim_steps_per_action

        for _ in range(self.sim_steps_per_action):
            self.apply_action(incremental_action)
            p.stepSimulation(physicsClientId=self.physics_client_id)
            state, goal = self.get_state()
        reward = self.get_reward(state, goal)
        done = False#(self.episode_idx >= self.episode_length)
        
        return state, reward, done, {'goal' : goal}

    def get_reward(self, state, goal):
        if self.pos_only:
            x, y, z = state[0]
        else:
            w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction = state[0]
        goal_x, goal_y,z, orn_x, orn_y, orn_z, orn_w  = goal
        distance = (x - goal_x)**2 + (y - goal_y)**2
        # print("Distance to goal:", distance)
        #tilt=np.abs(math.degrees(quat2euler(state[2][6:10])[2])) + np.abs(math.degrees(quat2euler(state[1][6:10])[2])) + np.abs(math.degrees(quat2euler(state[0][6:10])[2]))
        additional_penalty=0
        if stable_func(state) != (False, False):
           additional_penalty=100000
        return -1*distance + -1*additional_penalty      #-1*tilt*0.0005
        #if distance < self.goal_threshold:
        #    return 1.
        #return 0.

    def render(self, mode='human', close=False):
        base_pos=[0,0,0]

        if self.camera_vantage == 'side':
            distance, yaw, pitch, roll = 1.5, 155., -25., 0
        elif self.camera_vantage == 'birdseye':
            distance, yaw, pitch, roll = 1.5, 90., -90., 0
        else:
            raise Exception("Unrecognized camera camera_vantage. Options are 'side' and 'birdseye.")

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
            physicsClientId=self.physics_client_id)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(2048//4)/(1536//4),
            nearVal=0.1, farVal=100.0,
            physicsClientId=self.physics_client_id)

        (_, _, px, _, _) = p.getCameraImage(
        width=(2048//4), height=(1536//4), viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physics_client_id
            )

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def create_block(self, attributes, color=(0., 0., 1., 1.)):
        w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction = attributes

        # Create the collision shape
        half_extents = [w/2., l/2., h/2.]
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.physics_client_id)

        # Create the visual_shape
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, 
            physicsClientId=self.physics_client_id)

        # Create the body
        block_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id, 
            baseVisualShapeIndex=visual_id, basePosition=[x, y, z], baseOrientation=[orn_x, orn_y, orn_z, orn_w],
            physicsClientId=self.physics_client_id)
        p.changeDynamics(block_id, -1, lateralFriction=friction, physicsClientId=self.physics_client_id)

        # Cache block static attributes
        self.static_block_attributes[block_id] = (w, l, h, mass, friction)

        return block_id

    def apply_action(self, action):
        pos, _ = p.getBasePositionAndOrientation(self.gripper_id, physicsClientId=self.physics_client_id)
        new_pos = np.add(pos, action)
        p.changeConstraint(self.gripper_constraint_id, new_pos, maxForce=500, physicsClientId=self.physics_client_id)


def create_block_stack_push_env(backend='pybullet', *args, **kwargs):
    if backend == 'pybullet':
        return PybulletBlockStackPushEnv(*args, **kwargs)
    # TODO
    raise NotImplementedError()
