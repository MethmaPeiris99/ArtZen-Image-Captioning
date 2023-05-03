from imageai.Detection import ObjectDetection
import os
from PIL import Image
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
import collections
import csv


def convert_rgb_to_names(rgb_tuple):
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'


def filter_colours(input_data):
    filtered_colors = []

    for colour in input_data:
        filtered_colors.append(colour[1])

    return filtered_colors


def get_colour_names(detected_colours):
    counter = collections.Counter(detected_colours)  # Retrieve the colours detected in the image by removing the duplicates
    common_colours = counter.most_common(5)  # Store the most common colours 5 colours out of the detected colours
    common_colour_names = []

    for colour in common_colours:
        common_colour_names.append(colour[0])

    return common_colour_names

def build_image_caption(count_objects, detected_colours):
    if len(count_objects) == 0:
        return f"{detected_colours}, no objects"
    else:
        for object_name, object_count in count_objects.items():
            if object_count > 1:
                return f"{detected_colours}, {object_count} {object_name}s"
            else:
                return f"{detected_colours}, {object_count} {object_name}"


execution_path = os.getcwd()
detector = ObjectDetection()  # Create an instance of ObjectDetection()
detector.setModelTypeAsRetinaNet()  # Set the detection type of the ObjectDetection instance
detector.setModelPath(os.path.join(  # Set the pre-trained model path of the ObjectDetection instance
    execution_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth")
)
detector.loadModel()  # Load the ObjectDetection model

detectedColours = []  # Declare an array to store all the detected objects of an image
detectedObjects = []  # Declare an array to store all the detected colours of an image
image_caption_data = []

image_folder_directory = r"D:\retinanet_resnet50_fpn_coco-eeacb38b\test-paintings"

columnNames = ['Image File', 'Colours', 'Objects', 'Image Caption']

with open('wikiart-captions.csv', 'w') as wikiArtCaptionFile:
    csvWriter = csv.writer(wikiArtCaptionFile)
    csvWriter.writerow(columnNames)

    for image in os.listdir(image_folder_directory):
        # Clear the arrays where detected colours, detected objects are stored in each iteration
        detectedColours.clear()
        detectedObjects.clear()
        image_caption_data.clear()

        detections = detector.detectObjectsFromImage(
            input_image=os.path.join(execution_path + r"\test-paintings", image))
        img = Image.open(r"test-paintings\\" + image)
        image_colors = img.convert('RGB').getcolors(img.size[0] * img.size[1])
        image_colors = filter_colours(image_colors)

        for detectedColour in image_colors:
            detectedColours.append(convert_rgb_to_names(detectedColour))

        for detectedObject in detections:
            detectedObjects.append(detectedObject['name'])

        countObjects = collections.Counter(detectedObjects)
        print("OBJECT COUNT: ",countObjects)

        detected_colour_names = get_colour_names(detectedColours)
        image_file_name = image
        image_caption = build_image_caption(countObjects, detected_colour_names)
        image_caption_data = [image_file_name, detected_colour_names, detectedObjects, image_caption]
        print(image_caption_data, "\n")

        csvWriter.writerow(image_caption_data)
    print("END")