from os import listdir, mkdir, sep
from os.path import join, exists, splitext

def list_images_with_name(directory):
    images = []
    names = []
    for file in listdir(directory):
        name = file
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        names.append(name)
    return images, names