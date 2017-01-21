import bpy
from mathutils import *


class Camera:
    def __init__(self, width, height, mode='RGB'):
        self.scene = bpy.data.scenes["Scene"]
        self.camera = self.scene.camera
        self._setup(width, height, mode)

    def _setup(self, width, height, mode):
        self.scene.render.resolution_percentage = 100
        self.scene.render.resolution_x = width 
        self.scene.render.resolution_y = height
        self.scene.render.image_settings.color_mode = mode

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

    def takePicture(self, filename):
        """Save the scene as a png file"""
        self.scene.render.filepath = filename
        bpy.ops.render.render(write_still=True)