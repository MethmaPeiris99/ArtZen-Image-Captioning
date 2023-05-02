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


execution_path = os.getcwd()
detector = ObjectDetection()  # Create an instance of ObjectDetection()
detector.setModelTypeAsRetinaNet()  # Set the detection type of the ObjectDetection instance
detector.setModelPath(os.path.join(  # Set the pre-trained model path of the ObjectDetection instance
    execution_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth")
)
detector.loadModel()  # Load the ObjectDetection model

detectedColours = []  # Declare an array to store all the detected objects of an image
detectedObjects = []  # Declare an array to store all the detected colours of an image

image_folder_directory = r"C:\Users\USER\Downloads\retinanet_resnet50_fpn_coco-eeacb38b\animal-painting"

for image in os.listdir(image_folder_directory):

    detections = detector.detectObjectsFromImage(
        input_image=os.path.join(execution_path + r"\animal-painting", image))
    img = Image.open(r"animal-painting\\" + image)
    image_colors = img.convert('RGB').getcolors(img.size[0] * img.size[1])
    image_colors = filter_colours(image_colors)

    for detectedColour in image_colors:
        detectedColours.append(convert_rgb_to_names(detectedColour))

    for detectedObject in detections:
        detectedObjects.append(detectedObject['name'])

    counter = collections.Counter(detectedColours)  # Retrieve the colours detected in the image by removing the duplicates
    print("OUTPUT----------------")
    print(type(counter.most_common(5)[0]))
    print(counter.most_common(5)[0][0])
    print(img, str(counter.most_common(5)), detectedObjects, "\n")

    columnNames = ['Image File', 'Colours', 'Objects']
    with open('wikiart-captions.csv', 'w') as wikiArtCaptionFile:
        image_file_name = image
        detected_colours = detectedColours
        csvWriter = csv.writer(wikiArtCaptionFile)
        csvWriter.writerow(columnNames)
        csvWriter.writerow()
        wikiArtCaptionFile.write(img.filename + '- ')
        wikiArtCaptionFile.write(str(counter.most_common(5)))
        wikiArtCaptionFile.write(" | ")
        for detectedObject in detections:
            wikiArtCaptionFile.write(str(detectedObject["name"]) + ",")
        wikiArtCaptionFile.write("\n")
