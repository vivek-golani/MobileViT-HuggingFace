import os
import pdb
import torch
import argparse
import evaluate
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import VOCSegmentation
from transformers import MobileViTForSemanticSegmentation, MobileViTFeatureExtractor, MobileViTV2ForSemanticSegmentation, AutoImageProcessor

def eval(args):
    #hardcoded values
    num_classes = 21
    list_seg_pred = []
    list_seg_gt = []

    # dataset
    if args['dataset'] == 'voc':
        voc_dataset = VOCSegmentation(root="./data/VOC", year="2012", image_set="val", download=True)
    else:
        print(f'Dataset not supported yet')

    # model
    model = args['model'].split('_')[0]
    variant = args['model'].split('_')[1]
    if model == 'mobilevit':
        model_name = f'apple/deeplabv3-mobilevit-{variant}'
        model = MobileViTForSemanticSegmentation.from_pretrained(model_name)
        feature_extractor = MobileViTFeatureExtractor.from_pretrained(model_name)
    elif model == 'mobilevitv2':
        model_name = 'apple/mobilevitv2-1.0-imagenet1k-256'
        model = MobileViTV2ForSemanticSegmentation.from_pretrained(model_name)
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    else:
        print(f'Model not supported yet')

    # evaluation metric
    mean_iou = evaluate.load("mean_iou")

    # Num params in model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Num Params: {pytorch_total_params}')

    # Iterate over images and store logits from model
    for cnt, (img, label) in enumerate(voc_dataset):
        if cnt%200==0:
            print(f'Num samples seen: {cnt}')

        orig_shape = img.size

        # Basic augmentations - [crop, normalize by /255, RGB to BGR]
        inputs = feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            pred = outputs['logits']
            pred = F.interpolate(pred, size=(orig_shape[1], orig_shape[0]), mode='bilinear', align_corners=False)
            pred = pred.argmax(1).numpy()
            pred = np.squeeze(pred, axis=0)

        list_seg_pred.append(pred)

        label = np.array(label)
        list_seg_gt.append(label)
    
    # calculate miou
    ret_metrics = mean_iou.compute(predictions=list_seg_pred, references=list_seg_gt, num_labels=num_classes, ignore_index=255)
    print(f'Model: {args["model"]}\nDataset: {args["dataset"]}\nResult: {ret_metrics}')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Huggingface MobileViT torch model."
    )
    parser.add_argument('-d', '--dataset', default="voc", type=str, required=False, help='Name of the dataset') # options: 'zebra', 'voc'
    parser.add_argument('-m', '--model', default="mobilevit_small", type=str, required=False, choices=['mobilevit_xx-small', 'mobilevit_x-small', 'mobilevit_small'], help='Flavors of MobileViT models')
    parser.add_argument('-r', '--resolution', default=512, type=int, required=False, help='Image resolution of the model.')
    parser.add_argument('-t', '--task', default='segment', type=str, required=False, choices=['classify', 'segment'], help='Downstream task')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    eval(args)