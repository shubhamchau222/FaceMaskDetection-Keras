import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util
from tools.Encoder import encodeImagetoBase64


class Muliclassobj:
    def __init__(self,imagePath,modelPath):
        sys.path.append('..')
        self.model_name=modelPath
        self.image_name=imagePath
        current_wd = os.getcwd()
        self.path_to_ckpt = os.path.join(current_wd,self.model_name,"frozen_inference_graph.pb")
        self.path_to_labels= os.path.join(current_wd,'research/data','labelmap.pbtxt')
        self.path_to_image = os.path.join(current_wd,'research',self.image_name)
        self.num_classes = 10
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map , max_num_classes=self.num_classes , use_display_name=True)
        self.category_index=label_map_util.create_category_index(self.categories)
        self.class_names_mapping = {
                                    1 : 'without_mask',
                                    2 : 'with_mask',
                                    3 : 'mask_weared_incorrect'
                                     }
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def getPrediction(self):
        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(self.path_to_image)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        result = scores.flatten()
        res = []
        for idx in range(0, len(result)):
            if result[idx] > .55:
                res.append(idx)

        top_classes = classes.flatten()
        # select only those classes whose score is > 0.55 ( threshold )
        res_list = [top_classes[i] for i in res]
        class_final_names = [self.class_names_mapping[x] for x in res_list]
        top_scores = [e for l2 in scores for e in l2 if e > 0.30]

        new_scores = scores.flatten()
        new_boxes = boxes.reshape(300, 4)
        # get all boxes from an array
        max_boxes_to_draw = new_boxes.shape[0]
        # this is set as a default but feel free to adjust it to your needs
        min_score_thresh = .30

        listOfOutput = []
        for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
            valDict = {}
            valDict["className"] = name
            valDict["confidence"] = str(score)
            if new_scores is None or new_scores[i] > min_score_thresh:
                val = list(new_boxes[i])
                valDict["yMin"] = str(val[0])
                valDict["xMin"] = str(val[1])
                valDict["yMax"] = str(val[2])
                valDict["xMax"] = str(val[3])
                listOfOutput.append(valDict)

        # below method accept the Prediction and convert the predicted image to the base64 string
        # we'll receive the base64 image string in output so that we can visualize it manually
        # Img--> base64_string
        # to predict the Image
        vis_util.visualize_boxes_and_labels_on_image_array(
             image,
             np.squeeze(boxes),
             np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
           min_score_thresh=0.60)
        output_fileName = 'Output.jpg'
        cv2.imwrite(output_fileName,image) # write the prediction image to Given Image file
        opencodebase64 = encodeImagetoBase64('output.jpg')  # encoder : convert predicted image to base64
        listOfOutput.append({'image':opencodebase64.decode('utf-8')})

        return listOfOutput















