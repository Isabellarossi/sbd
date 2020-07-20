import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

import detectron2
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
import pyrealsense2 as rs
import numpy as np
from detectron2.gui.graphical_user_interface import gui_config


import pytest

'''
python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
	[--other-options]
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
'''



# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    config_path     = os.path.abspath(__file__ + "/..//../")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        
        default     = os.path.join(config_path + "/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml"),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser




# def test_global():

#     settings.init()          # Call only once
#     print(setting.mylist)
def add_events(events):
    # settings.init()   
    for event in events:
        settings.event_list.append(event)

    return settings.event_list[0]

def camera2_check():
    # camera2 check, if need re-positioning, update GUI & start re-check
    # check it if need to run the prediction.
    conveyable = True

    # run prediction
    frames = pipeline_2.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return
    # Convert images to numpy arrays
    img = np.asanyarray(color_frame.get_data())
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)


    cv2.imshow("Back camera", visualized_output.get_image()[:, :, ::-1])
    return conveyable


def test_add_events():
    import settings
    settings.init()          # Call only once
    events = ["Wheel at front","The bag is upright"]
    # settings.event_list.append("event")
    assert settings.event_list == ["event"]

def capital_case(x):
    return x.capitalize()

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    if args.input:
        print("input image")
    else: # args.webcam:
        print("get webcam.......................")
        num_camera = 2
        assert args.input is None, "Cannot have both --input and --webcam!"

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

        from detectron2.utils.visualizer import Visualizer, _create_text_labels
        from detectron2.data import DatasetCatalog, MetadataCatalog
        config_path     = os.path.abspath(__file__ + "/..//../")
        for d in ["train_coco","val_coco"]:    
            dicts = detectron2.data.datasets.load_coco_json(config_path + "/detectron2/data/melbourne/" +d + "/annotations.json", config_path + "/detectron2/data/melbourne/" + d, dataset_name=None, extra_annotation_keys=None)
            # DatasetCatalog.register("melbourne_" + d, lambda d=d: dicts)
            MetadataCatalog.get("melbourne_" + d).set(thing_classes=['_background_', 'suitcase','soft_bag','wheel','extended_handle','person','tray','upright_suitcase','spilled_bag','sphere_bag',
        'documents','bag_tag','strap_around_bag','stroller','golf_bag','surf_equipment','sport_equipment','music_equipment',
        'plastic_bag','shopping_bag','wrapped_bag','umbrella','storage_container','box','big_wheel','laptop_bag','tube','pet_container',
        'ski_equipment','tripod','child_safety_car_seat','tool_box','very_small_parcel','bingo_sticker'])
        melbourne_metadata = MetadataCatalog.get("melbourne_val_coco")


        from tkinter import *
        from tkinter import ttk
        from tkinter import messagebox
        import tkinter as tk

        
        gui_root = Tk()
        gui_config(gui_root)
        print("get pipeline.......................")
        try: 
            while(True):
                frames = pipeline_1.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # Convert images to numpy arrays
                img = np.asanyarray(color_frame.get_data())
                start_time = time.time()
                obj_predictions, visualized_output = demo.run_on_image(img)
                predictions = obj_predictions["instances"].to("cpu")
                
                
                

                boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
                scores = predictions.scores if predictions.has("scores") else None
                classes = predictions.pred_classes if predictions.has("pred_classes") else None
                labels = _create_text_labels(classes, scores, melbourne_metadata.get("thing_classes", None))
                           
                cv2.imshow("front camera", visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(1) == 27:
                    break  # esc to quit


                # TODO 
                # event_list, pose = pose_dim_estimation(portrait,profile,aligned_depth_frame, color_frame,
                #                        depth_intrin,depth_to_color_extrin,width,height)
                event_list = ["Wheel at front","The bag is upright"]


                from detectron2.event_output import output_json
                # mock image
                image_file = "/home/don/code/BagAnalysis/3DImaging/GPU_based_solution/Deeplearning/detectron2/detectron2/data/melbourne/train_melb_mask/img_860.jpg"
                # event_detected = output_json(image_file, classes, scores, boxes)


                # check event
                
                # send STOP
                from plc.Client import send_stop
                # send_stop()
                # and wait 2 sec for server message
                # from plc.Server import 
                # wait GUI operatioin

                
                # Update GUI
                result = gui_root.update()
                print(result)
                print("gui updated ..............................................")



                # # camera2 check, if need re-positioning, update GUI & start re-check
                # if num_camera == 2:
                #     # check it if need to run the prediction.

                #     # run prediction
                #     frames = pipeline_2.wait_for_frames()
                #     color_frame = frames.get_color_frame()
                #     if not color_frame:
                #         continue
                #     # Convert images to numpy arrays
                #     img = np.asanyarray(color_frame.get_data())
                #     start_time = time.time()
                #     predictions, visualized_output = demo.run_on_image(img)

                #     cv2.imshow("Back camera", visualized_output.get_image()[:, :, ::-1])

                    
        
        finally:
            cv2.destroyAllWindows()
            if num_camera == 1:
                pipeline.stop()
            elif num_camera ==2:
                # Stop streaming
                pipeline_1.stop()
                pipeline_2.stop()



