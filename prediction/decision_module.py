import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json

import detectron2
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.event_output import output_json

from predictor import VisualizationDemo
import pyrealsense2 as rs
import numpy as np
from detectron2.gui.graphical_user_interface import gui_config,  read_event


import pymodbus
from pymodbus.client.sync import ModbusTcpClient

'''
            Decision module
'''
def set_modbus(value):
    client = ModbusTcpClient('127.0.0.1')
    client.write_coil(1, value)
    result = client.read_coils(1,1)
    print(result.bits[0])



def setup_cfg():
    config_file = "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" 
    cfg.MODEL.WEIGHTS = "../detectron2/data/output/model_final.pth" # Windows doesn't support training.
    cfg.SOLVER.IMS_PER_BATCH = 1 # 16
    cfg.SOLVER.BASE_LR = 0.02 #0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 (15000: 9hours) iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8192   # 128 faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 33  

    cfg.freeze()
    
    return cfg

def metadataset():
    config_path     = os.path.abspath(__file__ + "/..//../")
    for d in ["train_coco","val_coco"]:    
        dicts = detectron2.data.datasets.load_coco_json(config_path + "/detectron2/data/melbourne/" +d + "/annotations.json", config_path + "/detectron2/data/melbourne/" + d, dataset_name=None, extra_annotation_keys=None)
        # DatasetCatalog.register("melbourne_" + d, lambda d=d: dicts)
        MetadataCatalog.get("melbourne_" + d).set(thing_classes=['_background_', 'suitcase','soft_bag','wheel','extended_handle','person','tray','upright_suitcase','spilled_bag','sphere_bag',
    'documents','bag_tag','strap_around_bag','stroller','golf_bag','surf_equipment','sport_equipment','music_equipment',
    'plastic_bag','shopping_bag','wrapped_bag','umbrella','storage_container','box','big_wheel','laptop_bag','tube','pet_container',
    'ski_equipment','tripod','child_safety_car_seat','tool_box','very_small_parcel','bingo_sticker'])
    melbourne_metadata = MetadataCatalog.get("melbourne_val_coco")
    return melbourne_metadata

def config_camera():
        # ...from Camera 1
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device('920312073100')
    config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # ...from Camera 2
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device('913522070674')
    config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming from both cameras
    pipeline_1.start(config_1)
    pipeline_2.start(config_2)

    return pipeline_1, pipeline_2


def mask_detection(pipeline, demo, metadata):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return
    # Convert images to numpy arrays
    img = np.asanyarray(color_frame.get_data())
    start_time = time.time()
    obj_predictions, visualized_output = demo.run_on_image(img)
    predictions = obj_predictions["instances"].to("cpu")
    img_result = visualized_output.get_image()[:, :, ::-1]
    
    
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    # labels = _create_text_labels(classes, scores, metadata.get("thing_classes", None))

    import datetime
    datetime_object = datetime.datetime.now()
    print(datetime_object)
    path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(path, str(datetime_object) + '.jpg')
    events = output_json(filename, classes, scores, boxes)
    return events, img_result


if __name__ == "__main__":
    
        from tkinter import *
        from tkinter import ttk
        from tkinter import messagebox
        import tkinter as tk

        import settings

        pipeline1, pipeline2 = config_camera()

        melbourne_metadata = metadataset()
        cfg = setup_cfg()
        demo = VisualizationDemo(cfg)

        gui_root = Tk()
        bhs = gui_config(gui_root)
        result = gui_root.update()
        
        try: 
            while(True):

                if settings.wait_for_user == True:
                    if settings.check_button == True:
                        #start sensor2
                        settings.wait_for_user = False
                        settings.check_button = False
                        settings.camera = "camera2"

                    elif settings.force_in_button == True:
                        settings.convoyable = True
                        settings.force_in_button = False
                        # send START
                        print("START")
                        # start sensor1
                        settings.camera == "camera1"
                        settings.wait_for_user = False
                    else:
                        # print("wait for user")
                        gui_root.update()
                else:

                    # sensor1
                    if settings.camera == "camera1":
                        pipeline = pipeline1
                    else:
                        pipeline = pipeline2
                    print(settings.camera)

                    event_dl, img_result = mask_detection(pipeline, demo, melbourne_metadata)

                    # event_list_pose, pose = pose_dim_estimation(portrait,profile,aligned_depth_frame, color_frame,
                    #                        depth_intrin,depth_to_color_extrin,width,height)
                    # event_list = event_list_detectrion + event_list_pose
                    event_list = event_dl
                    # test code
                    # event_list = []
                    # event_list.append("test1")
                    # event_list.append("test2")


                    if len(event_list) ==0:

                        # stop_conveyor = set_modbus(False) # temporaly define True: Start,  False: Stop,  and get the feedback
                        print("STOP")
                        read_event(bhs, gui_root, event_list, img_result)
                        settings.wait_for_user = True

                        # save the failure case
                        settings.camera = "camera1"
                        continue

                    else:
                        # all good for sensor1, pass
                        print("Analyzing")
                        if cv2.waitKey(1)== 27:
                            break 
                        # display "Analysing.........."

                # # camera2 check, if need re-positioning, update GUI & start re-check
                
                    
        
        finally:
            cv2.destroyAllWindows()
