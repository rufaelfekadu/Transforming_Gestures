from typing import Any
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
from threading import Thread
import time
import torch
import os
import sys
sys.path.append('/Users/rufaelmarew/Documents/tau/finger_pose_estimation')
from hpe.config import cfg
from hpe.utils.data import read_manus, read_leap, build_leap_columns,read_emg_v1
from hpe.utils.misc import setup_logger
from hpe.data import build_dataloaders
from hpe.models import build_model
import numpy as np

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
    def __init__(self):
        self.fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.joints = ["CMC", "MCP", "PIP", "DIP"]
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
        # angles = self.convert_to_manus([0 for i in range(len(self.joint_names))])
        self.params = params(angles=[0 for i in range(12)], jointNames=self.joint_names, handName=hand_name)
        self.unity_comms = UnityComms(cfg.VISUALIZE.PORT)

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

        # label_dir = cfg.VISUALIZE.LABEL_PATH
        # file_name = [i for i in os.listdir(label_dir) if i.endswith(".csv")][0]
        # label_path = os.path.join(label_dir, file_name)

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
                print(dc.gesture_names_mapping_class[gesture[0][j].item()])
                # print(leap_data.shape)
                angles = leap_data[j,0, :].tolist()
                self.update(angles, self.unity_comms)
                #  exit when q is pressed
                if input() == "q":
                    return
                time.sleep(sleep_time)


    
    
class HandEMG(HandBase):
    def __init__(self):
        pass

    def update(self, keypoints: Any):
        pass

    def reset(self):
        pass   

class Hands:
    def __init__(self, cfg):
        
        self.unity_comms = UnityComms(cfg.VISUALIZE.PORT)

        self.handPrediction = HandLeap(cfg, hand_name="Prediction")
        self.handLabel = HandLeap(cfg, hand_name="Label")

        self.fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        # load model from checkpoint
        self.model = build_model(cfg)
        self.model.load_pretrained(cfg.SOLVER.PRETRAINED_PATH)
        self.model.eval()

        # load dataloader
        self.data_loader = build_dataloaders(cfg, save=False)['test']

        #  setup logger
        self.logger = setup_logger(cfg=cfg, log_type="visualize")

    def update(self, keypoints: Any):
        print(len(keypoints[0]), len(keypoints[1]))

        self.handLabel.params.jointNames = self.joint_columns
        self.handPrediction.params.jointNames = self.joint_columns

        self.handPrediction.update(keypoints[0], self.unity_comms)
        self.handLabel.update(keypoints[1], self.unity_comms)
        

    def inference(self, data):
        with torch.no_grad():
            output= self.model(data, return_proj=False)
        return output

    def run_saved_data(self, cfg, sleep_time=1):

        self.logger.log(f"reading from {cfg.SOLVER.LOG_DIR}")
        pred = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'pred_cache.pth')).tolist()
        label = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'label_cache.pth')).tolist()
        
        print("started visualisation with {} data points".format(len(pred)))
        print("press enter to exit")
        for i in range(0, len(pred)):
            self.update((pred[i], label[i]))
            #  exit when enter is pressed
            if input() == "q":
                break
            time.sleep(sleep_time)

    def run_from_dataloader(self, cfg, sleep_time=1):
        
        self.logger.log("started visualisation with {} data points".format(len(self.data_loader)))
        # self.params.jointNames = self.dataloader.dataset.dataset.label_columns
        data_iter = iter(self.dataloader)
        for i in range(0, len(self.dataloader)):
            data,_,_,_,_, label, gesture = next(data_iter)
            output = self.inference(data)
            for j in range(0, len(label)):
                self.logger.log(self.dataloader.dataset.dataset.gesture_names_mapping_class[gesture[0][j].item()])
                #  get full hand angles
                name_xt, pred_xt = self.get_full_hand(output.tolist()[j], self.joint_columns)
                _, label_xt = self.get_full_hand(label[j,-1, :].tolist(), self.joint_columns)
                self.params.jointNames = name_xt
                self.update((label_xt[j].tolist(), pred_xt.tolist()[0]))
                #  exit when enter is pressed
                if input() == "q":
                    return
                time.sleep(sleep_time)

    def run_from_pretrained(self, cfg, sleep_time=1):
        #  forward pass on pretrained model
        dataloader = self.model.train_dataloader()
        # Assuming 'dataloader' is your existing DataLoader
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        num_workers = dataloader.num_workers
        # Create a new DataLoader with shuffle set to False
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # forward pass on pretrained model
        self.joint_columns = dataloader.dataset.dataset.label_columns
        mapping = dataloader.dataset.dataset.gesture_names_mapping_class
        data_iter = iter(dataloader)
        with torch.no_grad():
            for i in range(0, len(dataloader)):
                data, label, gesture = next(data_iter)
                output, loss = self.model(data)
                for j in range(0, len(label)):
                    print(mapping[gesture[0][j].item()])
                    
                    name_xt, pred_xt = self.get_full_hand(output.tolist()[j],self.joint_columns)
                    _, label_xt = self.get_full_hand(label[j,-1, :].tolist(), self.joint_columns)

                    # concatinate the  lists
                    angles = (pred_xt, label_xt)
                    self.joint_columns= name_xt

                    self.update(angles)
                    #  exit when enter is pressed
                    # if input() == "q":
                    #     return
                    time.sleep(sleep_time)
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

        start_pos = {
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
            values_to_append.append(start_pos[i])
        
        #  concatinate and return data and columns
        return cols+names_to_append, data+values_to_append
    
    #  TODO: implement this  
    def run_online(self, cfg, model, model_path, sleep_time=1):
        model.load_pretrained(model_path)
        model.eval()
        pass

HAND_MODES = {
    "manus": HandManus,
    "leap": HandLeap,
}

def make_hands(mode):
    return HAND_MODES[mode]()

def main(cfg):
    hands = Hands(cfg)
    # hands.run_from_loader(cfg, sleep_time=0.01)
    hands.run_from_pretrained(cfg, sleep_time=1.5)
    # hands.run_from_df(cfg, sleep_time=0.01)
    # hands.run_from_csv(cfg) 
    # hands.read_csv(cfg, sleep_time=0.01, data_loader=data_loader)

if __name__ == "__main__":

    # cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.MODEL.NAME)
    # cfg.DATA.EXP_SETUP = 'exp0'
    # cfg.DATA.PATH = './dataset/emgleap/003/S1'
    # cfg.DATA.SEGMENT_LENGTH = 100
    # cfg.DATA.STRIDE = 10
    # cfg.DEBUG = False
    # cfg.VISUALIZE.LABEL_PATH = './dataset/emgleap/003/S1/P1/'
    cfg.merge_from_file("config.yaml")
    # dataloaders = build_dataloader(cfg, save=False)

    main(cfg)