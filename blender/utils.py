import random

import bpy
from mathutils import *


def preprocess_args(argv):
    return argv[argv.index('--') + 1:] if '--' in argv else []

def parse_interval(interval):
    if '[' not in interval:
        # Assume it's a single number
        return float(interval)

    # Assume it's an interval
    interval_type = interval[:interval.index('[')]
    interval = interval[interval.index('['):]

    if interval_type not in ('uniform', 'linspace'):
        raise ValueError('Unrecognised interval type: {}'.format(interval_type))

    if len(interval) < 3 or not (interval[0] == '[' and interval[-1] == ']' and ',' in interval):
        raise ValueError('Could not parse interval: {}'.format(interval))
    a, b = interval[1:-1].split(',')
    try:
        a = float(a)
        b = float(b)
    except:
        raise ValueError('Could not parse interval: {}'.format(interval))
    if a >= b:
        raise ValueError('Invalid interval: {}'.format(interval))
    return a, b, interval_type

def renderToFile(filename, width, height, mode='RGB'):
    print(width, height)
    """Save the scene as a png file"""
    bpy.data.scenes['Scene'].render.filepath = filename
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.resolution_x = width 
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.image_settings.color_mode = mode
    bpy.ops.render.render(write_still=True)

class Camera:
    def __init__(self, camera):
        self.camera = camera

    def setLocation(self, r):
        if type(r) is not Vector:
            r = Vector(r)
        self.camera.location = r

    def setRotation(self, q):
        if type(q) is not Quaternion:
            q = Quaternion(q)
        self.camera.rotation_euler = q.to_euler()

    def look_at(self, point):
        # Point the camera's '-Z' and use its 'Y' as up
        q = (point - self.camera.location).to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = q.to_euler()

    def getLocation(self):
        return self.camera.location

    def getRotation(self):
        return self.camera.rotation_euler.to_quaternion()

    def getPoseString(self, decimals=6):
        r = [round(v, decimals) for v in self.getLocation()]
        q = [round(v, decimals) for v in self.getRotation()]
        return '{} {} {} {} {} {} {}'.format(r[0], r[1], r[2], q[0], q[1], q[2], q[3])
