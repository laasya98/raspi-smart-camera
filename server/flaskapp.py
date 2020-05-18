from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import logging
import boto3
from keras.models import load_model
import numpy as np
from flask import Flask
import cv2
import time
from face_classification.src.image_emotion_gender_demo_modified import demo_emotion


app = Flask(__name__)

ACCESS_KEY_ID = 'AKIA56JHVDHF24AYYC6Q'
ACCESS_SECRET_KEY = '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'
BUCKET_NAME = 'raspi-smart-camera'

jpg = ".jpg"

def run_mask_model(image_str):
    net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
    print("downloaded the model")

    # im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
    #                         'gluoncv/detection/biking.jpg?raw=true',
    #                         path='biking.jpg')

    im_fname = "images/" + image_str + jpg
    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]
    width, height = orig_img.shape[1], orig_img.shape[0]
    masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
    orig_img = utils.viz.plot_mask(orig_img, masks)

    print("finished mask classification, making plot")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                            class_names=net.classes, ax=ax)
    print("Plotted the mask model output")

    plt.savefig("images/" + image_str + "_mask" + jpg, bbox_inches='tight')
    print("End of MaskRCNN")

def run_emotion_model(image_str):
    demo_emotion(image_str)

def classify(image_str):
    start = time.time()
    run_mask_model(image_str)
    end1 = time.time()
    print("Execution time for emotion: " + str(end1-start))
    run_emotion_model(image_str)
    end2 = time.time()
    print("Execution time for mask: " + str(end2-start))

def test_upload():
    local_filename = "biker_test.jpg"
    s3_file_name = "biker_test_server.jpg"
    #note the s3 filename/path is set differently and has to be listed manually

    data = open(local_filename, 'rb')

    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
    )
    try:
        s3.Bucket(BUCKET_NAME).put_object(Key="test_folder/" + s3_file_name, Body=data)
        logging.info("Successfully uploaded file {} to S3 bucket {}/{}.".format(local_filename, BUCKET_NAME, s3_file_name))

    except Exception as e:
        print("Error: could not upload file:" + local_filename + " to s3:" + str(e))

    print ("Upload Done")

def test_download(image_str):
    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
    )

    s3_file_name = "upload_folder/" + str(image_str)
    local_download_path = "images/" + image_str #include the file name

    # Image download
    s3.Bucket(BUCKET_NAME).download_file(s3_file_name, local_download_path); # Change the second part
    # This is where you want to download it too.
    # I believe the semicolon is there on purpose
    print ("Download Done")

def download(image_str):
    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
    )

    s3_file_name = "upload_folder/" + str(image_str) + jpg
    local_download_path = "images/" + image_str + jpg #include the file name

    try:
    # Image download
        s3.Bucket(BUCKET_NAME).download_file(s3_file_name, local_download_path); # Change the second part
        logging.info("Successfully uploaded file {} to S3 bucket {}/{}.".format(local_download_path, BUCKET_NAME, s3_file_name))
    except Exception as e:
        print("Error: could not upload file:" + local_download_path + " to s3:" + str(e))

def upload(image_str):
    local_filename1 = "images/" + image_str + "_mask.jpg"
    local_filename2 = "images/" + image_str + "_emotion.jpg"

    s3_filename1 = image_str + "_mask.jpg"
    s3_filename2 = image_str + "_emotion.jpg"

    #note the s3 filename/path is set differently and has to be listed manually

    data1 = open(local_filename1, 'rb')
    data2 = open(local_filename2, 'rb')

    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
    )
    try:
        s3.Bucket(BUCKET_NAME).put_object(Key="test_folder/" + s3_filename1, Body=data1)
        logging.info("Successfully uploaded file {} to S3 bucket {}/{}.".format(local_filename1, BUCKET_NAME, s3_filename1))
    except Exception as e:
        print("Error: could not upload file:" + local_filename1 + " to s3:" + str(e))

    try:
        s3.Bucket(BUCKET_NAME).put_object(Key="test_folder/" + s3_filename2, Body=data2)
        logging.info("Successfully uploaded file {} to S3 bucket {}/{}.".format(local_filename2, BUCKET_NAME, s3_filename2))
    except Exception as e:
        print("Error: could not upload file:" + local_filename2 + " to s3:" + str(e))

    print ("Upload Done")

# doesn't work
# def detect_emotion(face_image_str):
#     img = cv2.imread("images/"+str(face_image_str))
#     # data = open("images/"+str(face_image_str), 'rb')
#     model = load_model("face_and_emotion_detection/emotion_detector_models/model_v6_23.hdf5")
#     predicted_class = np.argmax(model.predict(img))
#     print ("predicted class" + str(predicted_class))
#     return predicted_class

@app.route("/")
def hello():
    return "<h1 style='color:blue'>hiiii!</h1>"

@app.route('/classify/<input_str>')
def classify_image(input_str):
    # don't put .jpg in the name, i'll add it myself
    download(input_str) # downloads the original image from the upload folder in the bucket
    # test_upload()
    classify(input_str) # run the image through both models and save them
    upload(input_str) # upload the completed images into the processed folder _emotion.jpg and _mask.jpg
    return "classified and uploaded image: " + str(input_str)

@app.route('/download/<input_str>')
def download_image(input_str):
    print(input_str)
    test_download(input_str)
    return "tried to download image: " + str(input_str)

@app.route('/emotion/<input_str>')
def check_emotion(input_str):
    emotion = detect_emotion(input_str)
    return emotion

if __name__ == '__main__':
    app.run(host='0.0.0.0')

