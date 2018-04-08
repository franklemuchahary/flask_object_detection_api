from flask import Flask, render_template
from flask_restful import Resource, Api, reqparse
import werkzeug
import os
import json

from ImgDetWeb import img_det_web
import ObjDetML

app = Flask(__name__)
api = Api(app)

app.register_blueprint(img_det_web)

UPLOAD_FOLDER = 'static/img'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')


@app.route('/')
def home():
	return render_template('html/home.html')

class PhotoUpload(Resource):
    decorators=[]

    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No File Found',
                    'status':'error'
                    }
        photo = data['file']

        if photo:
        	#check for number of files in directory and rename file numerically
       		num_of_files_in_dir = len([name for name in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, name))])
        	final_image_path = UPLOAD_FOLDER+'/image'+str(num_of_files_in_dir)+'.jpg'
        	filename = 'image'+str(num_of_files_in_dir)+'.jpg'
        	photo.save(os.path.join(UPLOAD_FOLDER,filename))
        	#call ML function and store results in a dictionary
        	result_dict = ObjDetML.object_detection_funct(os.path.join(UPLOAD_FOLDER,filename), True)
        	return {
                	'predictions':result_dict,
                    'message':'Photo Uploaded and Predictions Made',
                    'status':'success'
                    }
        return {
                'data':'',
                'message':'Something Went Wrong',
                'status':'error'
                }



#api.add_resource(Home, '/')
api.add_resource(PhotoUpload, '/obj_detection_api')

if __name__ == '__main__':
	app.run(debug=True)