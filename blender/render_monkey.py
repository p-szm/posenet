import math
import random

import bpy
from mathutils import Vector


def renderToFile(filename, width, height):
    bpy.data.scenes['Scene'].render.filepath = filename
    bpy.context.scene.render.resolution_x = width 
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.ops.render.render(write_still=True)

def makeMaterial(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT' 
    mat.diffuse_intensity = 1.0 
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    return mat
 
def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

def setCamera(tx, ty, tz, rx, ry, rz):
    scene = bpy.data.scenes["Scene"]

    # Set camera rotation in euler angles
    scene.camera.rotation_mode = 'XYZ'
    scene.camera.rotation_euler[0] = rx*(math.pi/180.0)
    scene.camera.rotation_euler[1] = ry*(math.pi/180.0)
    scene.camera.rotation_euler[2] = rz*(math.pi/180.0)

    # Set camera translation
    scene.camera.location.x = tx
    scene.camera.location.y = ty
    scene.camera.location.z = tz

def look_at(obj_camera, point):
    loc_camera = obj_camera.location
    direction = point - loc_camera

    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def delete_object(name):
    try:
        bpy.data.objects[name].select = True
        bpy.ops.object.delete()
    except:
        print('No object with name {} found'.format(name))


# Empty the scene
delete_object('Cube')

# Define materials
red = makeMaterial('Red', (1,0,0), (1,1,1), 1)

# Create Suzanne
bpy.ops.mesh.primitive_monkey_add(radius=2, location=(0,0,0), rotation=(0,0,math.pi/2))
monkey = bpy.data.objects['Suzanne']
monkey.modifiers.new('My SubDiv', 'SUBSURF')
monkey.select = True
bpy.ops.object.shade_smooth()
setMaterial(monkey, red)

camera = bpy.data.objects['Camera']
N = 100

with open('train2.txt', 'w') as f:
    f.write('Suzanne Synthetic Dataset V1\n')
    f.write('ImageFile, Camera Position [X Y Z W P Q R]\n\n')
    for i in range(N):
        r = random.uniform(6, 10)
        theta = random.uniform(math.pi/3, 2*math.pi/3)
        phi = random.uniform(-math.pi/4, math.pi/4)
        x = r*math.sin(theta)*math.cos(phi)
        y = r*math.sin(theta)*math.sin(phi)
        z = r*math.cos(theta)
        setCamera(x, y, z, 0, 0, 0)
        look_at(camera, bpy.data.objects["Suzanne"].location+Vector((random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1))))
        q = camera.rotation_euler.to_quaternion()
        filename = 'train2/image{0:04d}.png'.format(i)
        renderToFile(filename, 1000, 800)
        f.write('{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f}\n'.format(filename, x, y, z, q[0], q[1], q[2], q[3]))
