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
from event_output import output_json

from projects.EfficientDet.test_folder import test


from predictor import VisualizationDemo
import numpy as np

# from data_transmission import server
import settings as settings

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
    config_file = "../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" 
    cfg.MODEL.WEIGHTS = "../data/output/model_final.pth" # Windows doesn't support training.
    cfg.SOLVER.IMS_PER_BATCH = 1 # 16
    cfg.SOLVER.BASE_LR = 0.02 #0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 (15000: 9hours) iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8192   # 128 faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 33  

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4 # args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # args.confidence_threshold

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

cfg = setup_cfg()
demo = VisualizationDemo(cfg)




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
    labels = _create_text_labels(classes, scores, melbourne_metadata.get("thing_classes", None))
    print("classes.shape, labels",classes.shape, labels)
    # a = torch.from_numpy(classes)
    # print(a)

    objects = []
    # COCO_CLASSES
    if classes is not None:
        for i in range(len(classes)):
            if scores[i]>0.9:
                label = labels[i].split(" ")[0]
                objects.append(label)
                print(label)
                if label == "suitcase":
                    print("suitcase detected")
                elif label == "tray":
                    print("tray")            
                elif label == "soft_bag":
                    print("soft bag")             
                elif label == "extended_handle":
                    response.flags.append(0)                     

    num_tray = sum(['tray' in x for x in objects ]) # tricky: count "tray:xxxx", not just "tray", so not use: num_tray = objects.count("tray")
    num_suitcase = sum(['suitcase' in x for x in objects ])
    num_soft_bag = sum(['soft_bag' in x for x in objects ])
    response = AnalysisResponse()

    if (num_suitcase + num_soft_bag) > 1: # multi bags
        response.flags.append(1)
    
    if num_tray >1:
        response.result = 2 #TubDetected
    elif num_soft_bag > 1 and num_tray ==0: 
        response.result = 3 # TubRequired
    elif num_suitcase > 0 :
        response.result = 1 # NoTubRequired
    else:
        response.result = 0
    print(response.result,response.flags)

    cv2.imwrite("image_sbd.jpg", img_result)

    return response

class AnalysisResponse():
	result = 0
	flags = []

if __name__ == "__main__":
    # AI_callback()
    # efficientDet_pred()
    cv_img = cv2.imread("/home/don/Pictures/suitcase.jpg")
    res = sbd_pred(cv_img)

