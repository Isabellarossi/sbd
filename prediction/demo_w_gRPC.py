import argparse
import glob
import multiprocessing as mp
import os, time, enum, json, sys
from os import listdir
import cv2, torch, tqdm
import random, re

import detectron2
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.event_output import output_json

from projects.EfficientDet.test_folder import test




from demo.predictor import VisualizationDemo
import pyrealsense2 as rs
import numpy as np


# from data_transmission import server
import demo.settings as settings

COCO_CLASSES = ['suitcase','soft_bag','wheel','extended_handle','person','tray','upright_suitcase','spilled_bag','sphere_bag',
'documents','bag_tag','strap_around_bag','stroller','golf_bag','surf_equipment','sport_equipment','music_equipment',
'plastic_bag','shopping_bag','wrapped_bag','umbrella','storage_container','box','big_wheel','laptop_bag','tube','pet_container',
'ski_equipment','tripod','child_safety_car_seat','tool_box','very_small_parcel','bingo_sticker']

colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
          (80, 7, 65), (10, 102, 25), (90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17),
          (82, 144, 81), (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60),
          (16, 179, 33), (112, 69, 165), (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34),
          (53, 34, 86), (141, 35, 190), (6, 171, 8), (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181),
          (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), (168, 33, 45), (28, 136, 135), (86, 91, 108),
          (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), (44, 65, 179), (1, 33, 26),
          (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112), (22, 157, 50),
          (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33),
          (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91),
          (31, 39, 184), (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130),
          (31, 139, 34), (66, 6, 127), (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3),
          (129, 87, 143), (49, 87, 40), (128, 62, 120), (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108),
          (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2), (62, 128, 96), (21, 3, 150), (0, 6, 95),
          (2, 20, 184), (122, 37, 185)]






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

class Opt():
    image_size = 1280
    cls_threshold = 0.8
    nms_threshold = 0.8
    # pretrained_model = "/home/don/code/BagAnalysis/3DImaging/GPU_based_solution/Deeplearning/detectron2/projects/EfficientDet/trained_models/efficient_model_418.pth"

def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection")
    parser.add_argument("--image_size", type=int, default=1280, help="The common width and height for all images")
    parser.add_argument("--cls_threshold", type=float, default=0.8)
    parser.add_argument("--nms_threshold", type=float, default=0.8)
    parser.add_argument("--pretrained_model", type=str, default="/home/don/code/BagAnalysis/3DImaging/GPU_based_solution/Deeplearning/detectron2/projects/EfficientDet/trained_models/efficient_model_440.pth")
    parser.add_argument("--input", type=str, default="test_videos/input.mp4")
    parser.add_argument("--output", type=str, default="test_videos/output.mp4")

    args = parser.parse_args()
    return args

def metadataset():
    config_path     = os.path.abspath(__file__ + "/..//../")
    for d in ["train_coco","val_coco"]:    
        dicts = detectron2.data.datasets.load_coco_json(config_path + "/data/melbourne/" +d + "/annotations.json", config_path + "/data/melbourne/" + d, dataset_name=None, extra_annotation_keys=None)
        # DatasetCatalog.register("melbourne_" + d, lambda d=d: dicts)
        MetadataCatalog.get("melbourne_" + d).set(thing_classes=['_background_', 'suitcase','soft_bag','wheel','extended_handle','person','tray','upright_suitcase','spilled_bag','sphere_bag',
    'documents','bag_tag','strap_around_bag','stroller','golf_bag','surf_equipment','sport_equipment','music_equipment',
    'plastic_bag','shopping_bag','wrapped_bag','umbrella','storage_container','box','big_wheel','laptop_bag','tube','pet_container',
    'ski_equipment','tripod','child_safety_car_seat','tool_box','very_small_parcel','bingo_sticker'])
    melbourne_metadata = MetadataCatalog.get("melbourne_val_coco")
    return melbourne_metadata
    
melbourne_metadata = metadataset()

def config_camera():
        # ...from Camera 1
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device('920312073100')
    config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    # ...from Camera 2
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device('913522070674')
    # config_2.enable_device('913522070714')
    config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

    # Start streaming from both cameras
    profile_1 = pipeline_1.start(config_1)
    profile_2 = pipeline_2.start(config_2)



    return pipeline_1, pipeline_2, profile_1, profile_2

# align_to = rs.stream.color
# align = rs.align(align_to)


# pipeline1, pipeline2, profile_1, profile_2 = config_camera()


cfg = setup_cfg()
demo = VisualizationDemo(cfg)


def get_frame(pipeline):


    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = frames.get_color_frame()
    if not color_frame:
        return

        # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    # Intrinsics & Extrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

    return color_frame


def test_imge_generator():
    if os.name == 'nt':
            print("windows")
            image_folder = "C:\\code\\BagAnalysis\\3DImaging\\CPU_based_solution\\Deeplearning\\models\\research\\object_detection\\test_images_benchmark_new"
    else:
        print("linux")            
        image_folder = "/home/don/tools/models/research/object_detection/test_images_benchmark_new/"
    # for f in random.sample(listdir(image_folder),1):
    f = random.sample(listdir(image_folder),1)
    image_file = os.path.join(image_folder, f[0])
    print("generator: ...... ", image_file)
    cv_img = cv2.imread(image_file)
    return cv_img


RECORD_LENGTH = 40
HEIGHT_CONVEYER = 0 #0.4 # from 0.25-0.4m

def crop_depth_data(depth_image, xmin_depth,xmax_depth, ymin_depth, ymax_depth):

    depth_org = depth_image[xmin_depth:xmax_depth, ymin_depth:ymax_depth].astype(float)

    # Get data scale from the device and convert to meters
    sensor = profile_1.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()
    depth = depth_org * depth_scale
    dist, _, _, _ = cv2.mean(depth)
    print("Detected a {0} {1:.3} meters away.".format("object ", dist))
    return dist

def depth_detect(depth, color):

    # ROI dept 
    xmin_depth = 200
    xmax_depth = 300
    ymin_depth = 200
    ymax_depth = 600
    dist = crop_depth_data(depth, xmin_depth,xmax_depth, ymin_depth, ymax_depth)
    if dist < 1.6 - HEIGHT_CONVEYER and dist > 0.0:
        print("Object is detected %.2f meter away." % dist)
        depth_detected = True
    else:
        # print("wait for objects.")
        depth_detected = False
    cv2.rectangle(color, (int(xmin_depth), int(ymin_depth)), (int(xmax_depth), int(ymax_depth)), (255, 255, 255), 2)

    cv2.imwrite("hahaha.jpg",depth)
    # cv2.imshow('COLOR IMAGE', color)
    
    # cv2.waitKey(10)
    # cv2.destroyAllWindows
    return depth_detected
    



def mask_detection(color_frame, demo, metadata):
    # Convert images to numpy arrays
    
    if settings.systemID == "BHS":

        cv_img = test_imge_generator()
        img = np.asanyarray(cv_img)

    elif settings.systemID == "SBD":
        img = np.asanyarray(color_frame.get_data())


    else:
        print("systemID error")

    start_time = time.time()
    
    if settings.prediction_model == "maskrcnn":
    
        # mask rcnn
        obj_predictions, visualized_output = demo.run_on_image(img)
        predictions = obj_predictions["instances"].to("cpu")
        img_result = visualized_output.get_image()[:, :, ::-1]
        
        
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        # labels = _create_text_labels(classes, scores, metadata.get("thing_classes", None))

        cv2.imwrite(image_file, img_result)
    
    elif settings.prediction_model == "efficientdet":

        # efficientDet

        opt = get_args()

        scores, classes, boxes = efficientDet_pred(cv_img, opt)
        for box_id in range(boxes.shape[0]):
            pred_prob = float(scores[box_id])
            if pred_prob < opt.cls_threshold:
                break
            pred_label = int(classes[box_id])
            xmin, ymin, xmax, ymax = boxes[box_id, :]
            color = colors[pred_label]
            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(COCO_CLASSES[pred_label] + ' : %.3f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(cv_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            cv2.putText(
                cv_img, COCO_CLASSES[pred_label] + ' : %.3f' % pred_prob,
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)
        # cv2.imwrite(os.path.join('demo', "demo_result.jpg"), cv_img)
        cv2.imshow("result", cv_img)
        cv2.waitKey(100)
        cv2.destroyAllWindows
    else:
        print("model error!")


    import datetime
    datetime_object = datetime.datetime.now()
    print(datetime_object)
    path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(path, str(datetime_object) + '.jpg')
    resResponse = output_json(filename, classes, scores, boxes)

    return resResponse





def AI_callback():
            start_time = time.time()
            print(settings.systemID, settings.prediction_model)
            
            
            # sensor1
            if settings.cameraId == 1:
                pipeline = pipeline1
                profile = profile_1
            else:
                pipeline = pipeline2
                profile = profile_2

            frames = pipeline.wait_for_frames()
            dl_frame = frames.get_color_frame()

            # point cloud
            aligned_frames = align.process(frames)
  
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            if (not color_frame) and (not aligned_depth_frame):
                return
            

            # depth detection
            color = np.asarray(frames[rs.stream.depth].get_data())
            depth = np.asarray(frames[rs.stream.depth].get_data())
            depth_detected = depth_detect(depth, color)
            
            # print("depth detection:.........", depth_detected)
            
            if not depth_detected:
                res1 = AnalyseResponse_data()
                res1.status = 1
                # print("nothing detected, res value...",res)
                return res1


                            # Intrinsics & Extrinsics
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)
            
            # 1st detect
            res1 = mask_detection(dl_frame, demo, melbourne_metadata)


            # 2nd detect
            frames_2 = pipeline.wait_for_frames()
            dl_frame_2 = frames_2.get_color_frame()

            
            dl_done_time = time.time()
            
            from pointcloud.point_interface  import pose_dim_estimation
            width = 640
            height = 480

            dimension_calculation_module = True
            orientation_detector_module = True
            upright_detector_module = True
            out_of_conveyor_detector_module = True

            box_point_calculation = True


            point_cloud_module = [dimension_calculation_module, orientation_detector_module,upright_detector_module,
                                  out_of_conveyor_detector_module]

            pose, event_list_pose = pose_dim_estimation(settings.portrait,profile,aligned_depth_frame, color_frame,
                                   depth_intrin,depth_to_color_extrin,width,height, point_cloud_module, box_point_calculation)
            stop_time = time.time()
            print("pose_dim_estimation time:", stop_time - start_time)
            
            dimension = pose[6]
            over_sized_event = event_list_pose[0]
            upright_event = event_list_pose[1]
            out_of_conveyor_event = event_list_pose[2]
            wrong_orientation_event = event_list_pose[3]

            event_list_pose = []
            event_list = []

            res1.length = int(dimension[0] * 1000)
            res1.width = int(dimension[1] *1000)
            res1.height = int(dimension[2] * 1000)

            
            num_suitcase = sum(['suitcase' in x for x in res1.classifications ])


            if over_sized_event == True:
                res1.rejectReasons.append("unauthorized object")
            elif upright_event == True and num_suitcase > 0:
                res1.rejectReasons.append("upright position")
                print("object", res1.classifications)
                
            elif out_of_conveyor_event == True:
                res1.rejectReasons.append("out of conveyor")
            elif wrong_orientation_event == True and num_suitcase > 0:
                res1.rejectReasons.append("wrong orientation")
                print("object", res1.classifications)
                

            
            settings.event_list = res1.rejectReasons


            stop_time = time.time()
            res1.cameraId = settings.cameraId

            res1.processingTime = int((stop_time - start_time) * 1000)
        
            print("total time:", stop_time - start_time)
            
            
            return res1

def efficientDet_pred(cv_img, opt):

    model = torch.load(opt.pretrained_model).module
    if torch.cuda.is_available():
        model.cuda()

    start_time = time.time()
    # image_expanded = np.expand_dims(image, axis=0)

    image = cv_img
    output_image = np.copy(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    height, width = image.shape[:2]
    image = image.astype(np.float32) / 255
    image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
    if height > width:
        scale = opt.image_size / height
        resized_height = opt.image_size
        resized_width = int(width * scale)
    else:
        scale = opt.image_size / width
        resized_height = int(height * scale)
        resized_width = opt.image_size

    image = cv2.resize(image, (resized_width, resized_height))

    new_image = np.zeros((opt.image_size, opt.image_size, 3))
    new_image[0:resized_height, 0:resized_width] = image
    new_image = np.transpose(new_image, (2, 0, 1))
    new_image = new_image[None, :, :, :]
    new_image = torch.Tensor(new_image)
    if torch.cuda.is_available():
        new_image = new_image.cuda()
    with torch.no_grad():
        scores, labels, boxes = model(new_image)
        boxes /= scale
        '''
    # filter for result
    dataframe = list(zip(boxes, labels, scores))
    dataframe = list(filter(lambda x: x[2] >= args.threshold*100, dataframe))

    if(len(dataframe) > 0):
        bboxes, labels, bbox_scores = list(zip(*dataframe))
    else:
        bboxes, labels, bbox_scores = [], [], []
        '''

    if boxes.shape[0] == 0:
        return

    for box_id in range(boxes.shape[0]):
        pred_prob = float(scores[box_id])
        if pred_prob < opt.cls_threshold:
            break
        pred_label = int(labels[box_id])
        xmin, ymin, xmax, ymax = boxes[box_id, :]
        color = colors[pred_label]
        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
        text_size = cv2.getTextSize(COCO_CLASSES[pred_label] + ' : %.3f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
        cv2.putText(
            output_image, COCO_CLASSES[pred_label] + ' : %.3f' % pred_prob,
            (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            (255, 255, 255), 1)


    stop_time = time.time()
    print("efficientDet prediction time.......", (stop_time - start_time))
    return scores, labels, boxes

class AnalyseResponse_data():
    class Status(enum.Enum): 
        Unknown = 0
        No_Bag = 1
        Cleared_Bag = 2
        Not_Cleared_Bag = 3
    # status = Status(1)
    status = 1
    rejectReasons = []
    cameraId = 0

    # timestamp = google.protobuf.Timestamp()
    processingTime = 6
    length = 7
    width = 8
    height = 9

    placementStatus = False
    placementDetails = 0
    orientationStatus = False
    orientaionDetails = 0
    classifications=[]
    components = []
    boundingBoxX1 = 0
    boundingBoxY1 = 0
    boundingBoxX2 = 0
    boundingBoxY2 = 0
    objectId = ""



def sbd_pred(cv_img):
    '''
    print("sbd prediction start........")
    cv_img = test_imge_generator()
    start_time = time.time()
    print(settings.systemID, settings.prediction_model)
    opt = get_args()

    print("1")
    scores, classes, boxes = efficientDet_pred(cv_img, opt)
    print("2")
    for box_id in range(boxes.shape[0]):
        pred_prob = float(scores[box_id])
        if pred_prob < opt.cls_threshold:
            break
        pred_label = int(classes[box_id])
        xmin, ymin, xmax, ymax = boxes[box_id, :]
        color = colors[pred_label]
        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), color, 2)
        text_size = cv2.getTextSize(COCO_CLASSES[pred_label] + ' : %.3f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(cv_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
        cv2.putText(
            cv_img, COCO_CLASSES[pred_label] + ' : %.3f' % pred_prob,
            (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            (255, 255, 255), 1)
    # cv2.imwrite(os.path.join('demo', "demo_result.jpg"), cv_img)
    print("3")


    res1.processingTime = int((stop_time - start_time) * 1000)

    # cv2.destroyAllWindows
    print("total time:", stop_time - start_time)
    return res1
    '''

    # mask rcnn
    obj_predictions, visualized_output = demo.run_on_image(cv_img)
    predictions = obj_predictions["instances"].to("cpu")
    img_result = visualized_output.get_image()[:, :, ::-1]
    
    
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    # labels = _create_text_labels(classes, scores, metadata.get("thing_classes", None))
    print("classes.shape",classes.shape)
    a = torch.from_numpy(classes)
    print(a)


    # COCO_CLASSES
    if classes is not None:
        for i in range(len(classes)):
            if scores[i]>0.9:
                print(classes[i])
                if classes[0][i] == 0 : # suitcase
                    print("suitcase detected")
                if classes[0][i] == 3 : # extended handle
                    print("suitcase detected")                    


    cv2.imwrite("image_sbd.jpg", img_result)

    # result = 
    return True

if __name__ == "__main__":
    # AI_callback()
    # efficientDet_pred()
    cv_img = cv2.imread("/home/don/Pictures/suitcase.jpg")
    res = sbd_pred(cv_img)

