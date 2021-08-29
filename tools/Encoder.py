import base64
# convert image data to base64string


def encodeImagetoBase64(image):
    with open(image,'rb') as f:
        img_data = base64.b64encode(f.read())
        f.close()
    return img_data
