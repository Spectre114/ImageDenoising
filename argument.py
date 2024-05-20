import easydict

def parse():
    '''
    Add arguments.
    '''
    args = easydict.EasyDict({
        "root_dir": '../dataset/BSDS300/images',
        "output_dir": '../checkpoints/',
        "num_epochs": 200,
        "D": 6,
        "C": 64,
        "plot": False,
        "model": 'dudncnn',
        "lr": 1e-3,
        "image_size": (180, 180),
        "test_image_size": (320, 320),
        "batch_size": 4,
        "sigma": 30
    })

    return args
