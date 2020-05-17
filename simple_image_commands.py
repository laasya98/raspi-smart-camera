# https://github.com/keithweaver/python-aws-s3

import boto3
import logging
# from botocore.client import Config

ACCESS_KEY_ID = 'AKIA56JHVDHF24AYYC6Q'
ACCESS_SECRET_KEY = '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'
BUCKET_NAME = 'raspi-smart-camera'

def test_upload(local_filename = "/home/pi/raspi-smart-camera/img0.jpg", s3_file_name = "test_img.jpg"):
    
    #note the s3 filename/path is set differently and has to be listed manually
    data = open(local_filename, 'rb')
    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        # config=Config(signature_version='s3v4')
    )
    try:
        s3.Bucket(BUCKET_NAME).put_object(Key="upload_folder/" + s3_file_name, Body=data)
        logging.info("Successfully uploaded file {} to S3 bucket {}/{}.".format(local_filename, BUCKET_NAME, s3_file_name))

    except Exception as e:
        print("Error: could not upload file:" + local_filename + " to s3:" + str(e))

    print ("Upload Done")

def test_download(s3_file_name = "test_folder/test_img.jpg", local_download_path = "test_img.jpg" ):
    # S3 Connect
    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        # config=Config(signature_version='s3v4')
    )

    #include the file name

    # Image download
    s3.Bucket(BUCKET_NAME).download_file(s3_file_name, local_download_path); # Change the second part
    # This is where you want to download it too.
    # I believe the semicolon is there on purpose

    print ("Download Done")

# I guess this doesn't really need to be a lambda?
# 1. From Wheesh, on button press, send image using above logic (test_upload)
# 2. Then, from Wheesh, invoke a lambda sitting in AWS that takes the sent filename
#    and runs some ML on the file and place it somewhere in an s3 bucket
# 3. Wheesh makes a request for that file, as it knows the filename (test_download)
# 4. In case it takes some time for the ML model to run, if Wheesh can't find the file, do a wait() and try again
# 5. Finally, store the downloaded file somewhere and show blit it


# test_upload()
# test_download()
