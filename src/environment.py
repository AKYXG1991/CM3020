import os
import pybullet as p
import pybullet_data
import math
import random


def make_mountain(pid, num_rocks=100, max_size=0.25, arena_size=10, mountain_height=5):
    def gaussian(x, y, sigma=arena_size/4):
        return mountain_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

    for _ in range(num_rocks):
        x = random.uniform(-1 * arena_size/2, arena_size/2)
        y = random.uniform(-1 * arena_size/2, arena_size/2)
        z = gaussian(x, y)
        size_factor = 1 - (z / mountain_height)
        size = random.uniform(0.1, max_size) * size_factor
        orientation = p.getQuaternionFromEuler([
            random.uniform(0, 3.14),
            random.uniform(0, 3.14),
            random.uniform(0, 3.14),
        ])
        rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size], physicsClientId=pid)
        rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=pid)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation, physicsClientId=pid)


def make_arena(pid, arena_size=10, wall_height=1):
    wall_thickness = 0.5
    floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], physicsClientId=pid)
    floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1], physicsClientId=pid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness], physicsClientId=pid)

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], physicsClientId=pid)
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1], physicsClientId=pid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2], physicsClientId=pid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2], physicsClientId=pid)

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], physicsClientId=pid)
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1], physicsClientId=pid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2], physicsClientId=pid)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2], physicsClientId=pid)


def load_sandbox(pid, arena_size=20):
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    shapes_path = os.path.join(os.path.dirname(__file__), 'shapes')
    p.setAdditionalSearchPath(shapes_path)
    p.setGravity(0, 0, -10, physicsClientId=pid)
    make_arena(pid, arena_size=arena_size)
    p.loadURDF('gaussian_pyramid.urdf', (0, 0, -1), p.getQuaternionFromEuler((0, 0, 0)), useFixedBase=1, physicsClientId=pid)


