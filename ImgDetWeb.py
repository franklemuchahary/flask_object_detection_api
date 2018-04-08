from flask import Blueprint, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import ObjDetML
import subprocess


UPLOAD_FOLDER = '/opt/REST_API/static/web_img_uploads'
UPLOAD_FOLDER_VIDEO = '/opt/REST_API/static/web_vid_uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
ALLOWED_EXTENSIONS_VIDEO = set(['mp4'])


img_det_web = Blueprint('img_det_web', __name__)

@img_det_web.route("/ImgDetWeb")
def upload_image_web():
    return render_template('html/upload_image.html')


'''
The Code Below Deals with Image Uploads
'''
@img_det_web.route("/ImgWebUpload", methods = ['GET', 'POST'])
def upload_file_from_web():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        #if everything is alright upload    
        if file and allowed_file(file.filename, False):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            #call ML function
            result_file_path = obj_det_ml(file_path)
            return render_template('html/show_result_image.html', result=result_file_path)


'''
The Code Below Deals with Video Uploads
'''
@img_det_web.route("/VidWebUpload", methods = ['GET', 'POST'])
def upload_video_from_web():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        #if everything is alright upload    
        if file and allowed_file(file.filename, True):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER_VIDEO, filename)
            file.save(file_path)
	    subprocess.call(['chmod', '0777', file_path])
            #call ML function
            result_file_path = video_obj_det_ml(file_path)
            #result_file_path = file_path
            return render_template('html/show_result_video.html', result=result_file_path)




def allowed_file(filename, video=False):
	if video == True:
		return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VIDEO
	else:
		return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''
The Code Below Deals with Machine Learning Stuff
'''
def obj_det_ml(file_path):
	return ObjDetML.object_detection_funct(file_path, False)

def video_obj_det_ml(file_path):
	return ObjDetML.object_detection_funct(file_path, False, True)
