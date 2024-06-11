from typing import Any
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
from threading import Thread

from hpe.config import cfg
from hpe.utils.data import read_manus, read_leap, build_leap_columns,read_emg_v1
from hpe.utils.misc import setup_logger
from hpe.data import build_dataloaders
from hpe.models import build_model

import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

@dataclass
class params:
    angles: list
    jointNames: list
    handName: str


['Thumb_CMC_Spread', 'Thumb_CMC_Flex', 'Thumb_PIP_Flex', 'Thumb_DIP_Flex',
    'Index_MCP_Spread', 'Index_MCP_Flex', 'Index_PIP_Flex', 'Index_DIP_Flex', 
    'Middle_MCP_Spread', 'Middle_MCP_Flex', 'Middle_PIP_Flex', 'Middle_DIP_Flex', 
    'Ring_MCP_Spread', 'Ring_MCP_Flex', 'Ring_PIP_Flex', 'Ring_DIP_Flex',
    'Pinky_MCP_Spread', 'Pinky_MCP_Flex', 'Pinky_PIP_Flex','Pinky_DIP_Flex']

class HandBase:
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    joints = ["TMC", "MCP", "PIP", "DIP"]
    def __init__(self):
        pass

    def update(self, angles: list, unity_comms: UnityComms):
        self.params.angles = angles
        unity_comms.UpdateHand(angles=self.params)
    
    def make_joint_names(self):
        pass

    def reset(self, unity_comms: UnityComms):
        self.params.angles = [0 for i in range(len(self.num_joints))]
        unity_comms.UpdateHand(angles=self.params)

    def run_from_csv(self, cfg, sleep_time=1):
        pass

    def run_saved_data(self, cfg, sleep_time=1):
        pass

    def run_online(self, cfg, model, model_path, sleep_time=1):
        pass

class HandManus(HandBase):

    def __init__(self, cfg, hand_name="Prediction"):
        super().__init__()
        self.joints = ["CMC", "MCP", "PIP", "DIP"]
        self.rotations = ["Spread", "Flex"]
        self.joint_names = self.make_joint_names()
        self.params = params(angles=[0 for i in range(len(self.joint_names))], jointNames=self.joint_names, handName=hand_name)

    def make_joint_names(self):
        joint_names = []
        for i in self.fingers:
            for j in self.joints:
                if (j == "MCP" and i =="Thumb") or (j == "CMC" and i !="Thumb"):
                    continue
                # if j == "CMC" or j == "MCP":
                #     joint_names.append(f"{i}_{j}_Spread")
                joint_names.append(f"{i}_{j}_Flex")
        return joint_names

class HandLeap(HandBase):
    def __init__(self, cfg, hand_name="Prediction"):
        super().__init__()
        self.joint_names = build_leap_columns()
        self.params = params(angles=[0 for i in range(12)], jointNames=self.joint_names, handName=hand_name)
        self.unity_comms = UnityComms(cfg.VIS.PORT)

    def convert_to_manus(self, keypoints: Any):
        #  convert anlges to list of lists with x,y,z coordinates
        new_keypoints = []
        for i in range(0, len(keypoints), 3):
            new_keypoints.append([keypoints[i], keypoints[i+1], keypoints[i+2]])
        return new_keypoints

    def reset(self):
        pass

    def update(self, keypoints: Any, unity_comms: UnityComms):
        #  convert to manus angles
        self.params.angles = keypoints
        unity_comms.UpdateLeapHands(angles=self.params)
    
    def run_from_csv(self, csv_path, sleep_time=0.001):

        dataset = read_leap(csv_path, positions=False, rotations=True, visualisation=True)
        dataset.drop(columns=["time_leap","timestamp"], inplace=True, errors="ignore")
        self.joint_names = dataset.columns.tolist()

        print(self.joint_names)
        print(len(self.joint_names))
        self.params.jointNames = self.joint_names
        print("started visualisation with {} data points".format(len(dataset)))
        print("press enter to exit")

        for i in range(0, len(dataset)):
            angles = dataset.iloc[i].tolist()
            # self.params.angles = angles
            self.update(angles, self.unity_comms)
            time.sleep(sleep_time)

    def run_from_loader(self, cfg, sleep_time=0.001, dataloader=None):
        if dataloader is None:
            dataloader = build_dataloaders(cfg, save=False, shuffle=False, visualize=True)
            dataloader = dataloader['test']
            dc = dataloader.dataset.dataset
        print("started visualisation with {} data points".format(len(dataloader)* dataloader.batch_size))
        self.joint_names = list(dc.label_columns)
        self.params.jointNames = self.joint_names
        data_iter = iter(dataloader)
        for i in range(0, len(dataloader)):
            data,_,_,_,_, leap_data, gesture = next(data_iter)
            for j in range(0, len(leap_data)):
                print(dc.gesture_mapping_class[gesture[0][j].item()])
                # print(leap_data.shape)
                angles = leap_data[j,0, :].tolist()
                self.update(angles, self.unity_comms)
                #  exit when q is pressed
                if input() == "q":
                    return
                time.sleep(sleep_time)

class Hands(HandBase):
    def __init__(self, cfg):
        
        self.unity_comms = UnityComms(cfg.VIS.PORT)

        self.handPrediction = HandLeap(cfg, hand_name="Prediction")
        self.handLabel = HandLeap(cfg, hand_name="Label")

        # load model from checkpoint
        self.model = build_model(cfg)
        self.model.load_pretrained(cfg.SOLVER.PRETRAINED_PATH+'best_model.pth')
        self.model.eval()

        # load dataset
        dataset = torch.load(cfg.SOLVER.PRETRAINED_PATH+'test_set.pth', map_location=torch.device('cpu'))
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            shuffle=False
        )

        self.joint_columns = dataset.label_columns.tolist()
        self.joint_columns_xt = []

        #  setup logger
        self.logger = setup_logger(cfg=cfg, log_type="visualize", setup_console=False)

    def update(self, keypoints: Any):

        self.handLabel.params.jointNames = self.joint_columns_xt
        self.handPrediction.params.jointNames = self.joint_columns_xt

        assert len(keypoints[0]) == len(self.joint_columns_xt), "label and joint names have different size"
        assert len(keypoints[1]) == len(self.joint_columns_xt), "Prediction and joint names have different sizes"

        self.handPrediction.update(keypoints[0], self.unity_comms)
        self.handLabel.update(keypoints[1], self.unity_comms)
        

    def inference(self, data_t, data_f):
        with torch.no_grad():
            output= self.model(data_t, data_f, return_proj=False)
        return output

    def run_from_pretrained(self, sleep_time=1):
        
        self.logger.info("started visualisation with {} data points".format(len(self.data_loader)))

        data_iter = iter(self.data_loader)
        for i in range(0, len(self.data_loader)):
            data_t,_,data_f,_,_, label, gesture = next(data_iter)
            output = self.inference(data_t, data_f)
            for j in range(0, len(label)):

                name_xt, pred_xt = self.get_full_hand(output.tolist()[j], self.joint_columns)
                _, label_xt = self.get_full_hand(label[j,-1, :].tolist(), self.joint_columns)
                self.logger.info([*label_xt, *pred_xt, self.data_loader.dataset.gesture_mapping_class[gesture[0][j].item()]])
                
                self.joint_columns_xt = name_xt
                self.update((label_xt, pred_xt))
                time.sleep(sleep_time)
                # #  Uncomment to exit when q is pressed
                # if input() == "q":
                #     return

    
    @staticmethod           
    def get_full_hand(data, cols):
        names_to_append = []
        values_to_append = []

        #  append dip joints 
        for i, v in zip(data, cols):
            if "_PIP_Flex" in v:
                if 'Thumb' in v:
                    continue
                else:
                    names_to_append.append(v.replace("_PIP_Flex", "_DIP_Flex"))
                    values_to_append.append(i*(2/3))
            if "Thumb_MCP_Flex" in v:
                names_to_append.append("Thumb_DIP_Flex")
                values_to_append.append(i*(0.5))

            # update to accomodate the initial value
            if "Thumb_TMC_Adb" in v:
                idx = data.index(i)
                data[idx] = data[idx] - 40

        abd_start_pos = {
            "Thumb": -40,
            "Index": -10,
            "Middle": 0,
            "Ring": 10,
            "Pinky": 20
        }
        
        #  append tmc joints
        for i in ['Index', 'Middle', 'Ring', 'Pinky']:
            names_to_append.append(f"{i}_TMC_Flex")
            values_to_append.append(0)
            names_to_append.append(f"{i}_TMC_Adb")
            values_to_append.append(abd_start_pos[i])
        
        #  concatinate and return data and columns
        return cols+names_to_append, data+values_to_append
    

HAND_MODES = {
    "manus": HandManus,
    "leap": HandLeap,
}

def make_hands(mode):
    return HAND_MODES[mode]()

def main(cfg):
    hands = Hands(cfg)
    hands.run_from_pretrained(cfg, sleep_time=1.5)

if __name__ == "__main__":

    cfg.merge_from_file("config.yaml")

    main(cfg)