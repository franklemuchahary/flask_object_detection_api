import numpy as np
import os
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import json

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def object_detection_funct(image_path, api_call=True, video_file=False):
    #initialize model graph path and labels map path
    path_name = '/opt/graph_def'
    path_to_frozen_det_graph = path_name + '/frozen_inference_graph.pb'
    path_to_labels = os.path.join('/opt/models/research/object_detection/data', 
                                  'mscoco_label_map.pbtxt')
    num_of_classes = 90
    
    #load the frozen tensorflow model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_det_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    #load the label maps for example 1=person, 2=cat etc. etc.
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_of_classes, 
                                                            use_display_name=True)
    category_map = label_map_util.create_category_index(categories)
    
    #actual prediction for single image
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            if video_file==False:
                image = Image.open(image_path)
                #convert image into numpy array using the function written
                image_np = load_image_into_numpy_array(image)
            
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                feed_dict={image_tensor: image_np_expanded})
            
        
                #decode the predictions and convert it into a dictionary
                output_dict = {'class':classes[0], 'score':scores[0]}
            
                if api_call == True:
                    #return final dictionary
                    scores_predicted = output_dict['score'][output_dict['score'] > 0.4]
                    classes_predicted = output_dict['class'][0:len(scores_predicted)].astype(np.int32)
                    results_dict = dict()
                    for i in range(0,len(classes_predicted)):
                        results_dict[str(i)+'_'+str(category_map[classes_predicted[i]]['name'])] = str(round(
                            scores_predicted[i]*100,2))+" %"
                    return results_dict
                else:
                    #save image with boxes and return image path
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), 
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores), category_map,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)
                    DIR = '/static/img_results'
		    DIR2 = '/opt/REST_API/static/img_results'
                    num_of_files_in_dir = len([name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, 
                                                                                                           name))])
                    final_image_path = DIR2+'/image'+str(num_of_files_in_dir)+'.png'
		    img_path_to_return = DIR+'/image'+str(num_of_files_in_dir)+'.png'
                    result_img = Image.fromarray(image_np, 'RGB')
                    result_img.save(final_image_path)

		    scores_predicted = output_dict['score'][output_dict['score'] > 0.4]
                    classes_predicted = output_dict['class'][0:len(scores_predicted)].astype(np.int32)
                    results_dict = dict()
                    for i in range(0,len(classes_predicted)):
                        results_dict[str(i)+'_'+str(category_map[classes_predicted[i]]['name'])] = str(round(
                            scores_predicted[i]*100,2))+" %"

                    return img_path_to_return, results_dict
            else:
                reader = imageio.get_reader(image_path)
                fps = reader.get_meta_data()['fps']
                VID_OP_DIR = '/static/vid_results'
		VID_OP_DIR2 = '/opt/REST_API/static/vid_results'
                num_of_files_in_dir = len([name for name in os.listdir(VID_OP_DIR2) if os.path.isfile(os.path.join(VID_OP_DIR2, 
                                                                                                           name))])
                final_video_path = VID_OP_DIR2+'/output_video'+str(num_of_files_in_dir)+'.mp4'
		vid_path_to_return = VID_OP_DIR+'/output_video'+str(num_of_files_in_dir)+'.mp4'
                writer = imageio.get_writer(final_video_path, fps = fps)
                for i, frame in enumerate(reader):
                    image_np = frame
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                feed_dict={image_tensor: image_np_expanded})
                    
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), 
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores), category_map,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)
                    writer.append_data(image_np)
                writer.close()
                return vid_path_to_return
