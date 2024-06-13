from peaceful_pie.unity_comms import UnityComms
import pandas as pd
import sys
sys.path.append('/Users/rufaelmarew/Documents/tau/finger_pose_estimation')
from util import read_manus, read_emg, parse_arg
import time
from data import EMGDataset, make_dataset
from config import cfg
import os 
import torch
from dataclasses import dataclass

@dataclass
class Angles:
    predAngles: list
    labelAngles: list


def update_hands(unity_comms, predAngles, labelAngles):
    angles = Angles(predAngles, labelAngles)
    unity_comms.UpdateHands(angles=angles)

def update_hand(unity_comms, angles):
    unity_comms.UpdateHand(angles=angles)

def reset_hand(unity_comms):
    angles = [0 for i in range(19)]
    update_hand(unity_comms, angles)

def get_hand(args):

    label_path = "/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/label_2023-10-02_15-24-12_YH_lab_R.csv"
    dataset = read_manus(label_path)
    dataset.drop(columns=['Thumb_CMC_Flex'], inplace=True)
    # drop first 20 seconds
    dataset = dataset[3000:]

    return dataset

def infered_hand(cfg):
    pred = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'pred_cache.pth'))
    label = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'label_cache.pth'))
    
    pred_list = pred.tolist()
    label_list = label.tolist()
    
    return pred_list, label_list

# read infered data from pth file and update the hand model
def update_both_hands(pred, label, unity_comms):
    # threaded function to update both hands
    for i in range(0, len(pred)):
        update_hands(unity_comms, pred[i][:-1], label[i][:-1])
        print(pred[i][:-1], label[i][:-1])
        time.sleep(1)

# read from csv file and update hand model
def read_pred(args, unity_comms):
    dataset = get_hand(args)
    print(dataset.columns)
    time.sleep(2)
    for i in range(0, len(dataset)):
        angles = dataset.iloc[i].tolist()
        print(angles)
        update_hand(unity_comms, angles)
        time.sleep(0.0001)

def main(cfg):
    unity_comms = UnityComms(cfg.VISUALIZE.PORT)
    # reset_hand(unity_comms)

    pred, label = infered_hand(cfg)
    print(len(pred), len(label))
    # read_pred(cfg, unity_comms)
    update_both_hands(pred, label, unity_comms) 
    # reset_hand(unity_comms)
    # read_pred(cfg, unity_comms)


if __name__ == "__main__":

    args = parse_arg(disc="visualisation script for the model")

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)


    # only for 
    # if cfg.DEBUG:
    #     cfg.SOLVER.LOG_DIR = "../debug"
    #     # set the config attribute of args to 

    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.MODEL.NAME)
    main(cfg)