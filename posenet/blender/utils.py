def preprocess_args(argv):
    return argv[argv.index('--') + 1:] if '--' in argv else []