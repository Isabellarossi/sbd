# 1 modified the convertor code to fix the bounding box issue
Here is only one bbox field in the coco converted json though there are more than one bbox in the image. But the number of segmentations in the coco converted json matches with the non coco json file.

Solution: replaced lines 109 to 125 in the original labelme2coco.py file with the following (please build from source: https://github.com/wkentaro/labelme/), renaming labels to person-1, person-2, etc, before adding them to the coco annotations file.

```
'''
        masks = {}                                     # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            shape_type = shape.get('shape_type', None)
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if label in masks:
                masks[label] = masks[label] | mask
            else:
                masks[label] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[label].append(points)
        '''
```

# 2 To convert the labelme json file to coco formate. Please check the path of labelme first and then run the commands in path: "BagAnalysis/3DImaging/GPU_based_solution/Deeplearning/detectron2/detectron2/data":

For train dataset
```
python /home/don/tools/labelme/examples/instance_segmentation/labelme2coco.py melbourne/train_melb_mask melbourne/train_coco --labels melbourne/labels.txt
```
for val dataset
```
python /home/don/tools/labelme/examples/instance_segmentation/labelme2coco.py melbourne/val_melb_mask melbourne/val_coco --labels melbourne/labels.txt
```

