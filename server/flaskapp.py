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

ACCESS_KEY_ID = ''
ACCESS_SECRET_KEY = ''
BUCKET_NAME = 'raspi-smart-camera'

jpg = ".jpg"

def run_mask_model(image_str):
    net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
    print("downloaded the model")
    print("Starting Inference")

    im_fname = "images/" + image_str + jpg
    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]
    width, height = orig_img.shape[1], orig_img.shape[0]
    masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
    orig_img = utils.viz.plot_mask(orig_img, masks)

    print("finished mask classification, making plot")

    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                            class_names=net.classes, ax=ax)
    print("Plotted the mask model output")
    # fig.set_size_inches(w,h)

    ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(orig_img, aspect='auto')

    plt.savefig("images/" + image_str + "_mask" + jpg, bbox_inches='tight', pad_inches=0)
    print("End of MaskRCNN")

def run_emotion_model(image_str):
    demo_emotion(image_str)

def classify(image_str):
    print("Starting Classify")
    start = time.time()
    run_mask_model(image_str)
    end1 = time.time()
    print("Execution time for emotion: " + str(end1-start))
    run_emotion_model(image_str)
    end2 = time.time()
    print("Execution time for mask: " + str(end2-end1))

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

def upload_mask(image_str):
    local_filename1 = "images/" + image_str + "_mask.jpg"

    s3_filename1 = image_str + "_mask.jpg"

    #note the s3 filename/path is set differently and has to be listed manually

    data1 = open(local_filename1, 'rb')

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

    print ("Upload Mask Done: " + image_str)

def upload_emotion(image_str):
    local_filename2 = "images/" + image_str + "_emotion.jpg"

    s3_filename2 = image_str + "_emotion.jpg"

    #note the s3 filename/path is set differently and has to be listed manually

    data2 = open(local_filename2, 'rb')

    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
    )
    try:
        s3.Bucket(BUCKET_NAME).put_object(Key="test_folder/" + s3_filename2, Body=data2)
        logging.info("Successfully uploaded file {} to S3 bucket {}/{}.".format(local_filename2, BUCKET_NAME, s3_filename2))
    except Exception as e:
        print("Error: could not upload file:" + local_filename2 + " to s3:" + str(e))

    print ("Upload Emotion Done: " + image_str)


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Welcome to raspi-smart-camera!</h1><h2>Endpoints:</h2><h3>classify, emotion, mask</h3>"

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
    print("downloading image: " + input_str)
    test_download(input_str)
    return "tried to download image: " + str(input_str)

@app.route('/emotion/<input_str>')
def emotion(input_str):
    print("GET Request for /emotion on image: " + input_str)
    download(input_str)
    run_emotion_model(input_str)
    upload_emotion(input_str)
    return ("Finished Executing Emotion")

@app.route('/mask/<input_str>')
def mask(input_str):
    print("GET Request for /mask on image: " + input_str)
    download(input_str)
    run_mask_model(input_str)
    upload_mask(input_str)
    return "Finished Executing Mask"

if __name__ == '__main__':
    app.run(host='0.0.0.0')

