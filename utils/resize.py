from PIL import Image
def resize(image):
    SIZE = (256, 256)
    return image.resize(SIZE, Image.Resampling.BOX)