from io import BytesIO
import os
from PIL import Image, ImageDraw
from unittest import result
import requests

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

local_image = ".\images\street.jpeg"
remote_image = "https://raw.githubusercontent.com/Azure-Sample/cognitive-services-sample-data-files/master/ComputerVision/Images/objects.jpg"
image_features = ['objects', 'tags']

subscription_key = "a6128fa681ff428ea629bb25b4dbef5d"
endpoint = "https://objectdetection45.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

def drawRectangle(object, draw):
    rect = object.rectangle
    left = rect.x
    right = left + rect.w
    top = rect.y
    bottom = top + rect.h
    coordinates = ((left, top), (right,bottom))
    draw.rectangle(coordinates, outline = 'red')


def getObjects(results, draw):
    print('OBJECTS DETECTED:')
    if len(results.objects) == 0:
        print('No objects detected')
    else:
        for object in results.objects:
            print('object at location {}, {}, {},{}'.format(
                object.rectangle.x, object.rectangle.x+object.rectangle.w,
                object.rectangle.y, object.rectangle.y+ object.rectangle.h
            ))

            drawRectangle(object, draw)
            print()
            print('Bounding boxes frawn around objects, see popup')
        print()


def getTags(results):
    print('TAGS:')
    if (len(results.tags) == 0):
        print('No tags detected')
    else:
        for tag in results.tags:
            print(" '{}' with confidence {:.2f}%".format(
                tag.name, tag.confidence*100
            ))
    print()


    '''
    Analyze Image
    '''
    print('Analyze local image')
    print()

    local_image_object = open(local_image, 'rb')
    image_1 = Image.open(local_image)
    draw = ImageDraw.Draw(image_1)

    results_local = computervision_client.analyze_image(local_image_object, image_features)

    getObjects(results_local, draw)
    getTags(results_local)

    image_1.show()
    print()



    '''
    Detecting Objects
    '''

    print('Analyze remote image')
    print()

    results_remote = computervision_client.analyze_image(remote_image, image_features)

    object_image = requests.request.get(remote_image)
    image_r = Image.open(BytesIO(object_image.content))
    draw = ImageDraw.Draw(image_r)

    getObjects(results_remote, draw)
    getTags(results_remote)

    image_r.show()