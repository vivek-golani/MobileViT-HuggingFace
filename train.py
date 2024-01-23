import os
import pdb
import yaml
import torch
import shutil
import argparse
import evaluate
import numpy as np
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from PIL import Image as image

import data.torch as ptu
from data.loader import Loader
from data.BaseZebra import BaseZebraDataset
from visualize import visualize_dataset

from datasets import Dataset, Features, Image
from transformers.integrations import TensorBoardCallback
from transformers.models.mobilevit.modeling_mobilevit import MobileViTConvLayer
from transformers import MobileViTImageProcessor, MobileViTForSemanticSegmentation, TrainingArguments, Trainer, ProgressCallback

def load_config():
    return yaml.load(open(Path(__file__).parent / 'config.yml', 'r'), Loader=yaml.FullLoader)

def load_dataset(args, cfg):    
    dataset = args['dataset']
    if dataset == 'zebra':
        dataset_kwargs=dict(
            image_size=cfg[dataset]['image_size'],
            crop_size=args['resolution'],
            normalization='mobilevit',
        )
        train_dataset = BaseZebraDataset(split='train', **dataset_kwargs)
        dataset_kwargs['batch_size'] = 1
        val_dataset = BaseZebraDataset(split='val', **dataset_kwargs)

    return train_dataset, val_dataset

def train(model, processor, dataset, train_dataset, val_dataset, cfg):
    base_dir = Path(__file__).parent
    output_dir = os.path.join(str(base_dir / 'saved_models'), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    logdir = os.path.join(str(base_dir / 'logs'), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    def compute_metrics(eval_pred):
        metric = evaluate.load("mean_iou")
        with torch.no_grad():
            logits, labels = eval_pred
            print(f'shapes: log: {logits.shape}, lab: {labels.shape}')
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()

            metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=cfg[dataset]['num_classes'],
                ignore_index=0,
                reduce_labels=False,
            )
            
            # add per category metrics as individual key-value pairs
            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update({f"accuracy_{i}": v for i, v in enumerate(per_category_accuracy)})
            metrics.update({f"iou_{i}": v for i, v in enumerate(per_category_iou)})
            
            return metrics

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=1)
        return pred_ids, labels

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg[dataset]['learning_rate'],
        num_train_epochs=cfg[dataset]['epochs'],
        per_device_train_batch_size=cfg[dataset]['batch_size'],
        per_device_eval_batch_size=1,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        log_level='debug',
        logging_steps=1,
        logging_dir=logdir,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.add_callback(ProgressCallback)
    trainer.add_callback(TensorBoardCallback)
    trainer.train()

def fine_tune(args):
    names = ['background', 'auntJemimaLitePancakeSyrup', 'bagClip', 'brush1', 'brush3', 'bumblebeeSardines', 'careOneEyeDrops', 'clearAmericanGoldenPeach', 'clearAmericanSparklingRaspberry', 'concordiaFortifiedWine', 'condimentCup', 'cottonWaxCube', 'crystalLightFruitPunch', 'crystalLightGrapeLemonade', 'crystalLightLiquidBlueRaspberry', 'crystalLightPeachTeaMix', 'dawnDishwashingSoapLemon', 'delMonteAsparagusSpears', 'delMonteDicedTomatoes', 'dvdFelicity', 'dvdSamantha', 'essentiaPurified', 'gameNCAAFootball11', 'ghirardelliBSChocoChips', 'good2growAppleJuice', 'greatValueCanChickPeas', 'greatValueCanChiliBeans', 'greatValueCanPintoBeans', 'greatValueHeavyDutyScrub', 'jelloChocolate', 'knorrChicken', 'knorrNoodles', 'knorrRiceCheddarBroccoli', 'kodiakCakesCCOatmealCup', 'koolaidCherry', 'koolaidOrange', 'laysWavyChips', 'lemonade', 'lipTonBlackTea', 'maizenaVanilla', 'mezzettaRoastedBellPeppers', 'ortegaDicedGreenChiles', 'ortegaTacoMix', 'paperMateMiniBallpoint', 'pinkLemonade', 'pirateBootyPuffs', 'poweradeMountainBerry', 'republicOfTeaBlackberrySage', 'saladVegetableGoya', 'sazonGoya', 'scentedWaxCube', 'schickShavingBlade', 'smuckerStrawberryPreserve', 'ticTac', 'tidePodsColdwaterPacs', 'uncleBenRiceCreamyCheese', 'uncleBenRiceJasmine', 'uncleBenRiceOriginal', 'votivesUnscented', 'walmartFork', 'walmartSpoon', 'wegmansDicedTomatoes', 'wegmansVegetableCrackers', 'whiteCandle']
    colors = [[0,0,0], [247, 114, 123], [247, 115, 107], [247, 117, 85], [247, 119, 50], [234, 128, 50], [224, 135, 50], [214, 140, 50], [206, 144, 50], [199, 147, 50], [192, 150, 50], [185, 153, 50], [179, 155, 50], [172, 157, 49], [166, 159, 49], [159, 162, 49], [151, 164, 49], [143, 166, 49], [133, 168, 49], [122, 170, 49], [108, 173, 49], [88, 176, 49], [56, 179, 49], [50, 178, 81], [50, 177, 102], [51, 177, 116], [52, 176, 126], [52, 175, 135], [52, 175, 142], [53, 174, 148], [53, 174, 154], [53, 173, 159], [54, 173, 164], [54, 172, 169], [54, 172, 174], [55, 171, 179], [55, 171, 184], [55, 170, 189], [56, 169, 195], [56, 168, 201], [57, 167, 208], [58, 166, 217], [59, 165, 228], [60, 162, 241], [90, 158, 244], [115, 154, 244], [134, 149, 244], [150, 145, 244], [164, 140, 244], [177, 135, 244], [189, 129, 244], [202, 123, 244], [214, 115, 244], [227, 106, 244], [241, 94, 244], [245, 94, 234], [245, 97, 221], [245, 100, 210], [245, 102, 200], [246, 104, 190], [246, 106, 181], [246, 108, 171], [246, 109, 161], [247, 111, 150]]

    cfg = load_config()

    # data
    train_ds, val_ds = load_dataset(args, cfg)

    visualize_dataset(train_ds, 10, 'Train')
    visualize_dataset(val_ds, 10, 'Val')

    pdb.set_trace()
    # model
    variant = args['model'].split('_')[1]
    name = f'apple/deeplabv3-mobilevit-{variant}'
    model = MobileViTForSemanticSegmentation.from_pretrained(name, ignore_mismatched_sizes=True)
    model.config.num_labels = cfg[args['dataset']]['num_classes']
    config = model.config
    model.segmentation_head.classifier = MobileViTConvLayer(
        config,
        in_channels=256,
        out_channels=config.num_labels,
        kernel_size=1,
        use_normalization=False,
        use_activation=False,
        bias=True,
    )

    processor = MobileViTImageProcessor.from_pretrained(name)

    # train
    train(model, processor, args['dataset'], train_ds, val_ds, cfg)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Huggingface MobileViT torch model."
    )
    parser.add_argument('-d', '--dataset', default="zebra", type=str, required=False, help='Name of the dataset')
    parser.add_argument('-m', '--model', default="mobilevit_small", type=str, required=False, choices=['mobilevit_xx-small', 'mobilevit_x-small', 'mobilevit_small'], help='Flavors of MobileViT models')
    parser.add_argument('-r', '--resolution', default=512, type=int, required=False, help='Image resolution of the model.')
    parser.add_argument('-t', '--task', default='segment', type=str, required=False, choices=['classify', 'segment'], help='Downstream task')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    fine_tune(args)

