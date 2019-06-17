import numpy as np
import os
import sys
import cv2 as cv
from scipy.io import loadmat
import os
import shutil
from tqdm import tqdm
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot, plot
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from resnet_152 import resnet152_model


###################    CLASSIFICATION FUNCTIONS ######################

def load_pretrained_model(model_weights_path = 'models/model.96-0.89.hdf5'):
    """
    Function:
        - loads a pretrained ResNet-152 model
    Argument:
        - model_weights_path (str): the relative path to model weights
    Returns:
        - model" the pretrained ResNet-152 model
    """
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    model = resnet152_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model

def rank_predictions(probabilities, classnames, top=5):
    """
    Function:
        - Filter the probabilities array outputed by the prediction step to get the top n predictions
    Arguments:
        - probabilities (np.ndarray) : 196-dimension probabilities array outputed by the prediction step
        - classnames (np.array) : 196-dimension array of classnames
        - top (int) : number of top predictions to get
    Returns
        - ranked predictions (dict): a dictionary whose keys are integers (1,2,3...`top`) and values are the classnumber, classname and probability corresponding to that rank
    """
    ranked_predictions = {}


    top_indices = np.argsort(-probabilities)[0,:top]
    top_probabilities = probabilities[0, top_indices]
    top_classes = classnames[top_indices]

    for i in range(top):
        ranked_predictions[i+1] = {"classnumber":top_indices[i] + 1,
                                   "classname":top_classes[i][0],
                                   "probability" : top_probabilities[i]}
    return ranked_predictions

def predict_images_in_a_directory(directory, classnames, model, prediction_output = "classname", verbose = True):
    """
    Function:
        - Predict images in a directory
    Arguments:
        - directory (str): relative path to the directory containing the images
        - classnames (np.array) : 196-dimension array of classnames
        - model: the keras model to be used fo prediction
        - prediction_output (str): choose from "classname" or "classnumber"
        - verbose (bool): if True, prints the predicted class and the real class as the prediction is running
    Returns:
        - predicted_labels (list): a list of predictions
    """
    # Get the complete paths to the images
    files = os.listdir(directory)
    paths = []
    for file in files:
        path = os.path.join(directory, file)
        paths.append(path)

    number_samples = len(paths)
    predicted_labels = []


    for i, path in tqdm(enumerate(paths)):

        # Preprocess the image
        img = cv.imread(path)
        img = cv.resize(img, (224, 224), cv.INTER_CUBIC)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        # Get the probabilities of all class
        probabilities = model.predict(img)
        # Get the top probability
        top = 1
        predicted_classname = rank_predictions(probabilities, classnames, top)[top]["classname"]
        predicted_classnumber = rank_predictions(probabilities, classnames, top)[top]["classnumber"]

        if prediction_output == "classname":
            predicted_labels.append(predicted_classname)
        else:
            predicted_labels.append(predicted_classnumber)

        if verbose:
            print(f"Predicting sample {i + 1} out of {number_samples}")
            print(f"Prediction:{predicted_classname}")
            print()
            print("=======================================")

    return predicted_labels

def predict_images_in_a_nested_directory(sorted_directory, classnames, model, prediction_output = "classname", verbose = True):
    """
    Function:
        - Predict images in a nested folder where images are placed in subfolders whose names are the classnames of the images of
    Arguments:
        - sorted_directory (str): relative path to the sorted_directory containing the images
        - classnames (np.array) : 196-dimension array of classnames
        - model: the keras model to be used fo prediction
        - prediction_output (str): choose from "classname" or "classnumber"
        - verbose (bool): if True, prints the predicted class and the real class as the prediction is running
    Returns:
        - predicted_labels (list): a list of predictions
        - real_labels (list): a list of real labels
    """
    # Get the complete paths to the images
    paths = []
    for root, directories, files in os.walk(sorted_directory, topdown=False):
        for file in files:
            paths.append(os.path.join(root, file))

    number_samples = len(paths)
    predicted_labels = []
    real_labels = []


    for i, path in tqdm(enumerate(paths)):
        # Preprocess the image
        img = cv.imread(path)
        img = cv.resize(img, (224, 224), cv.INTER_CUBIC)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        # Get the probabilities of all class
        probabilities = model.predict(img)
        # Get the top probability
        top = 1
        predicted_classname = rank_predictions(probabilities, classnames, top)[top]["classname"]
        predicted_classnumber = rank_predictions(probabilities, classnames, top)[top]["classnumber"]
        if prediction_output == "classname":
            predicted_labels.append(predicted_classname)
        else:
            predicted_labels.append(predicted_classnumber)

        # Get the real classname from folder structure
        real_classname = path.split(os.path.sep)[-2]
        # The presence of / confuses the slitting process
        if real_classname == "V Cargo Van Minivan 2012":
            real_classname = "Ram C/V Cargo Van Minivan 2012"

        real_classnumber = np.where(classnames == real_classname)[0] + 1

        if prediction_output == "classname":
            real_labels.append(real_classname)
        else:
            real_labels.append(real_classnumber)

        if verbose:
            print(f"Predicting sample {i + 1} out of {number_samples}")
            print(f"Prediction:{predicted_classname}")
            print(f"Real class:{real_classname}")
            print()
            print("=======================================")

    return predicted_labels, real_labels

def calculate_accuracy(y_pred, y_real):
    """
    Function: Calculate the accuracy of a model by comparing its predictions and real labels
    Arguments:
        - y_pred (list) = list of predictions
        - y_real(list) = list of real labels
    Returns:
        - accuracy (float): number of correct predictions divided by number of total predictions
    """

    assert len(y_pred) == len(y_real)


    total_prediction = len(y_pred)
    correct_prediction = 0


    for i in range(total_prediction):
        if y_pred[i] == y_real[i]:
            correct_prediction += 1
    accuracy = correct_prediction/total_prediction

    return accuracy

def draw_confusion_matrix(real_classnames,predicted_classnames, classnames, save_as_html = False):

    """
    Function:
        - Draw an interactive plotly confusion matrix that compares the predictions and real values
    Arguments:
        - real_classnames (list) : a list of the correct labels
        - prediction_classnames (list): a list of prediced labels
        - classnames (np.array) : 196-dimension array of classnames

    """
    # Get the list of classnames
    classnames_list = [classname[0] for classname in classnames]
    for index, name in enumerate(classnames_list):
        if name == "Ram C/V Cargo Van Minivan 2012":
            classnames_list[index] = "Ram CV Cargo Van Minivan 2012"

    # Get the numpy confusion matrix
    cm = confusion_matrix(real_classnames, predicted_classnames)
    cm_normalize = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    # Get the hovertext
    hovertext_normalized = list()
    for yi, yy in enumerate(classnames_list):
        hovertext_normalized.append(list())
        for xi, xx in enumerate(classnames_list):
            hovertext_normalized[-1].append(f'Predicted car model: {xx}<br />Actual car model: {yy}<br />Normalized count: {cm_normalize[yi][xi]:.3f}')


    # Draw the confusion matrix
    trace = go.Heatmap(x = classnames_list,
                       y = classnames_list,
                       z = cm_normalize,
                       hoverinfo="text",
                       text = hovertext_normalized,
                      colorscale=[[0.0, 'rgb(0,0,0)'],
                                  [1.0, 'rgb(255,255,255)']],
                      showscale = False)
    data=[trace]
    layout = go.Layout(title = "Normalized confusion Matrix (hover over to see values)",
                      xaxis = dict(title = "Predicted car model",  showticklabels=False),
                      yaxis = dict(title = "Actual car model", showticklabels=False),
                      width=1000,
                      height=1000,
                      margin=go.layout.Margin(
                                                l=50,
                                                r=50,
                                                b=100,
                                                t=100,
                                                pad=4
                                            ))
    figure = go.Figure(data = data, layout = layout)
    if save_as_html:
        plot(figure)
    else:
        iplot(figure)

def safe_predictions_as_txt(predictions, filename = "predictions.txt"):
    """
    Function:
        - Save a list of prediction as a txt file
    Arguments:
        - predictions (list): list of predictions
        - filename (str): name of the file in which the predictions will be saved.
    """
    outF = open(filename, "w")
    for line in predictions:
    # write line to output file
        outF.write(str(line))
        outF.write("\n")
    print(f"Saved predictions as {filename}")
    outF.close()

def predict_from_path(path, model):
    """
    Function:
        - Gives the probabilities from the prediction of one single file
    Arguments:
        - path (str): relative path to the image to be predicted
        - model: model to be used for prediction
    Returns:
        - probabilities (np.ndarray): probabilities of each class

    """
    # Preprocess the image
    img = cv.imread(path)
    img = cv.resize(img, (224, 224), cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # Get the probabilities of all class
    probabilities = model.predict(img)
    return probabilities

def draw_probability_confusion_matrix(main_matrix, classnames, save_as_html = False):

    """
    Function:
        - Draw an interactive plotly probability confusion matrix that gives conditional probabilities of each prediction
    Arguments:
        - main_matrix (np.ndarray) : a 196 * 196 numpy array
        - classnames (np.ndarray) : 196-dimension array of classnames

    """
    # Get the list of classnames
    classnames_list = [classname[0] for classname in classnames]
    for index, name in enumerate(classnames_list):
        if name == "Ram C/V Cargo Van Minivan 2012":
            classnames_list[index] = "Ram CV Cargo Van Minivan 2012"

    # Get the hovertext
    hovertext = list()
    for yi, yy in enumerate(classnames_list):
        hovertext.append(list())
        for xi, xx in enumerate(classnames_list):
            hovertext[-1].append(f"Probability of the prediction being '{xx}' given real label being {yy}:<br />Probability: {main_matrix[yi][xi]:.3f}")


    # Draw the confusion matrix
    trace = go.Heatmap(x = classnames_list,
                       y = classnames_list,
                       z = main_matrix,
                       hoverinfo="text",
                       text = hovertext_normalized,
                      colorscale=[[0.0, 'rgb(0,0,0)'],
                                  [1.0, 'rgb(255,255,255)']],
                      showscale = False)
    data=[trace]
    layout = go.Layout(title = "Normalized confusion Matrix (hover over to see values)",
                      xaxis = dict(title = "Predicted car model",  showticklabels=False),
                      yaxis = dict(title = "Actual car model", showticklabels=False),
                      width=1000,
                      height=1000,
                      margin=go.layout.Margin(
                                                l=50,
                                                r=50,
                                                b=100,
                                                t=100,
                                                pad=4
                                            ))
    figure = go.Figure(data = data, layout = layout)
    if save_as_html:
        plot(figure)
    else:
        iplot(figure)


################    OBJECT DETECTION FUNCTIONS     ###################

PATH_TO_FROZEN_GRAPH = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03' + \
    '/frozen_inference_graph.pb'


def build_localization_model(path_to_frozen_graph=PATH_TO_FROZEN_GRAPH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    """
    Function:
        - Transforms the box masks back to full image masks.
        - Embeds masks in bounding boxes of larger masks whose shapes correspond to image shape.

    Arguments:
        - box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
        - boxes: A tf.float32 tensor of size [num_masks, 4] containing the box corners.
                 Row i contains [ymin, xmin, ymax, xmax] of the box corresponding to mask i.
                 Note that the box corners are in normalized coordinates.
        - image_height: Image height. The output mask will have the same height as the image height.
        - image_width: Image width. The output mask will have the same width as the image width.

    Returns:
        - A tf.float32 tensor of size [num_masks, image_height, image_width].
    """

    def reframe_box_masks_to_image_masks_default():
        """
        Function: The default function when there are more than 0 box masks.
        """
        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat([tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        return tf.image.crop_and_resize(image=box_masks_expanded,
                                        boxes=reverse_boxes,
                                        box_ind=tf.range(num_boxes),
                                        crop_size=[image_height, image_width],
                                        extrapolation_value=0.0)

    image_masks = tf.cond(tf.shape(box_masks)[0] > 0,
                          reframe_box_masks_to_image_masks_default,
                          lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
    return tf.squeeze(image_masks, axis=3)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def predict_bounding_box(image, graph):
    """
    Function:
        - Uses the object detection model to localize the most prominently displayed car in an image
    Arguments:
        - image (np.ndarray): a numpy representation of the car image
        - graph: Tensorflow graph containing the object detection model
    Returns:
    if model detects a car:
        - best_bounding_box_coordinates (tuple): a tuple of coordinates corresponding to the location of the most prominently displayed car.
                                                The bounding box coordinates (ymin,xmin, ymax,xmax) are specified in normalized coordinates.
        - best_bounding_box_probability (float): the confidence score of the best_bounding_box actually containing a car
    if the model doesn't detect any car:
        - None, None
    """


    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image, axis=0)

    # Actual detection step
    output_dict = run_inference_for_single_image(image_expanded, graph)

    try:
        # Get bounding boxes that capture cars or car-like objects
        threshold = 0.5
        car_indexes = ((output_dict["detection_classes"] == 3) + (
            output_dict["detection_classes"] == 8))*(output_dict["detection_scores"] > threshold)
        car_bounding_boxes = output_dict["detection_boxes"][car_indexes]

        def calculate_area(bounding_box):
            """Calculate an area enclosed by the coordinates of bounding_box"""
            ymin, xmin, ymax, xmax = bounding_box
            area = (ymax - ymin)*(xmax - xmin)
            return area

        # Get bounding box with the largest area, which should correspond to the most prominently displayed car in the image
        best_bounding_box_index = np.array(
            [(calculate_area(car_bounding_box)) for car_bounding_box in car_bounding_boxes]).argmax()
        best_bounding_box_coordinates = output_dict["detection_boxes"][car_indexes][best_bounding_box_index]
        best_bounding_box_probability = output_dict["detection_scores"][car_indexes][best_bounding_box_index]

        return best_bounding_box_coordinates, best_bounding_box_probability

    except:

        # If the above filtering step encounters errors, that means there is no car in the picture
        print("The model doesn't detect any car in the picture!")
        print("The function will return `None`.")
        return None, None

def filter_and_preprocess_images_using_model(source_directory,target_directory, graph):
    """
    Function:
        - Using the provided object detection model, crop and resize images in the source_directory and move them into the target_directory

    Arguments:
        - graph: Tensorflow graph containing the object detection model
        - source_directory: a nested directory containing subfolders whose names are classnames; these subfolders contain images to be preprocessed
        - target_directory: a nested directory into which the preprocess images will be moved

    """


    if not os.path.exists(target_directory):
        os.mkdir(target_directory)


    for root, directories, files in tqdm(os.walk(source_directory, topdown=False)):
        for i, file in tqdm(enumerate(files)):
            source_path = os.path.join(root, file)
            classname = source_path.split(os.path.sep)[-2]
            print(classname, i)
            target_path = os.path.join(target_directory, classname, file)
            if os.path.exists(target_path):
                print(f"{target_path} already exists")
            else:
                try:
                    # Read the original image
                    original_image_bgr = cv.imread(source_path)
                    original_image_rgb = cv.cvtColor(original_image_bgr, cv.COLOR_BGR2RGB)

                    # Find its bounding box
                    best_bounding_box_coordinates, best_bounding_box_probability = predict_bounding_box(original_image_rgb)

                    # If a car is detected and if its relative area is more than 10% of the image area:
                    if best_bounding_box_probability is not None and calculate_area(best_bounding_box_coordinates) > 0.1:

                        # Crop it and resize it
                        cropped_image_rgb = crop_image_with_bounding_box(original_image_rgb, best_bounding_box_coordinates)
                        cropped_image_bgr = cv.cvtColor(cropped_image_rgb, cv.COLOR_RGB2BGR)
                        resized_image_bgr = cv.resize(cropped_image_bgr, (224, 224))

                        # Make the correct subfolder (whose name is the image's classname) if it hasn't existed yet
                        # Put the cropped and resized image in that subfolder
                        target_subfolder = os.path.join(target_directory,classname)
                        if not os.path.exists(target_subfolder):
                            os.mkdir(target_subfolder)
                        cv.imwrite(target_path, resized_image_bgr)
                except:
                    continue

def detect_crop_and_resize_images_using_model(graph, source_directory="./cars_test/cars_test", target_directory="./cars_test_preprocessed"):
    """
    Function:
        - Given images of cars in a directory, use the object detection model, to localize the cars
        - Crop and resize images in the source_directory and place the preprocessed images in the target_directory
    Arguments:
        - graph: Tensorflow graph containing the object detection model
        - source_directory (str): relative path of the directory containing the images to be preprocessed
        - target_directory (str): relative path of the directory into which the preprocessed images are to be placed
    """

    image_list = os.listdir(source_directory)
    length_image_list = len(image_list)
    print(f"There are {length_image_list:,} images to preprocess in `{source_directory}`.")
    print("Detecting..")

    # If target_directory doesn't exist, create it
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    # Preprocess images in the source_path and place them in destination_path
    for index, filename in tqdm(enumerate(image_list)):
        print(f"Processing {filename}, which is image {index + 1} out of {length_image_list}")

        original_image_bgr = cv.imread(os.path.join(source_directory, filename))
        original_image_rgb = cv.cvtColor(original_image_bgr, cv.COLOR_BGR2RGB)
        try:
            best_bounding_box_coordinates, best_bounding_box_probability = predict_bounding_box(original_image_rgb, graph)


            cropped_image = crop_image_with_bounding_box(original_image_rgb, best_bounding_box_coordinates)
            resized_image = cv.resize(cropped_image, (224, 224))
            resized_image_bgr = cv.cvtColor(resized_image, cv.COLOR_RGB2BGR)

            cv.imwrite(os.path.join(target_directory, filename), resized_image_bgr)
        except:
            resized_image_bgr = cv.resize(original_image_bgr, (224, 224))
            cv.imwrite(os.path.join(target_directory, filename), resized_image_bgr)
    print(f"{length_image_list:,} images have been preprocessed and placed in `{target_directory}`")
