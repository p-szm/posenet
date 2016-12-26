import bpy
from mathutils import *

camera = bpy.data.objects['Camera']
camera.location = (7.564687, -1.99391, 1.403622)
camera.rotation_euler = Quaternion((0.608239, 0.509721, 0.392023, 0.465347)).to_euler()