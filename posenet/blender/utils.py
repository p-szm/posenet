import bpy


def preprocess_args(argv):
    return argv[argv.index('--') + 1:] if '--' in argv else []

def use_gpu():
    bpy.context.scene.cycles.device = 'GPU'
