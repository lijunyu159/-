import os
import cv2
import json
import pandas as pd
import numpy as np
from glob import glob 
from tqdm import tqdm
from IPython import embed
import base64
from labelme import utils
image_path = "D:/Mask_RCNN/mask_rcnn/samples/balloon/balloon/transform/"
csv_file = "D:/Mask_RCNN/mask_rcnn/samples/balloon/balloon/via_project_9Jun2020_16h29m_csv.csv"
annotations = pd.read_csv(csv_file,header=None).values
total_csv_annotations = {}
for annotation in annotations:
    key = annotation[0].split(os.sep)[-1]
    value = np.array([annotation[1:]])
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
    else:
        total_csv_annotations[key] = value
for key,value in total_csv_annotations.items():
    # key是文件名
    try:
        height,width = cv2.imread(image_path+key).shape[:2]
    except:
        print(image_path+key)
    labelme_format = {
    "version":"4.4.0",
    "flags":{},
    "imagePath":key,
    "imageHeight":height,
    "imageWidth":width
    }
    with open(image_path+key,"rb") as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    #img = utils.img_b64_to_arr(imageData)
    labelme_format["imageData"] = imageData
    # labelme_format["imageWidth"]=width
    # labelme_format["version"]="4.4.0"
    shapes = []
    for shape in value:
        # print(shape[4])
        coodinate=eval(shape[4])
        try:
            label = eval(shape[-1])["category"]
        except:
            print(key)
        s = {"label":label,"flags":{},"group_id":None,"shape_type":"polygon"}
        points=[]
        if "all_points_x" in coodinate:
            for i in range(len(coodinate["all_points_x"])):
                if len(points)<=i:
                    points.append([coodinate["all_points_x"][i],coodinate["all_points_y"][i]])
            s["points"] = points
            shapes.append(s)
    labelme_format["shapes"] = shapes
    # labelme_format["imageHeight"]=height
    # labelme_format["imagePath"]=key
    # labelme_format["flags"]={}
    json.dump(labelme_format,open("%s%s"%(image_path,key.replace(".jpg",".json")),"w"),ensure_ascii=False, indent=2)
