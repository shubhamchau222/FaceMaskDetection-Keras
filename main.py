from flask import request , jsonify ,Flask
from flask_cors import CORS, cross_origin
import os
from research.obj import Muliclassobj
from tools.Decoder import Decoder

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
app = Flask(__name__)
CORS(app)

class configure:
    def __init__(self):
        self.model_path  = 'research/faster_rcnn_inception_v2_coco'
        self.filename = "inputImage.jpg"
        self.objectDetection = Muliclassobj(self.filename , self.model_path)

@app.route('/predict' , methods=['POST'])
@cross_origin()
def prediction():
    image = request.json['image']
    print(image)
    Decoder(image,clapp.filename)
    result=clapp.objectDetection.getPrediction()
    return jsonify(result)

if __name__ == '__main__':
    clapp = configure()
    app.run(host='127.0.0.1', port=5002, debug=True)  # to run on web browser
    #app.run(host='0.0.0.0', port=7000, debug=True)




