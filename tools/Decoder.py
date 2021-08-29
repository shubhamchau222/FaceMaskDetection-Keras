# this file decodes the base64 to Image
import base64
import os
def Decoder(b64string , fileName):
    image_data = base64.b64decode(b64string)
    storage_path = os.path.join('research' , fileName )
    print(storage_path)
    with open(file=storage_path ,mode='wb') as f:
        f.write(image_data)
        f.close()



