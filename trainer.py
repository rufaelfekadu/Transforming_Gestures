from typing import Any, Optional, Sequence, Union
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lightning.pytorch import callbacks
from torch.optim.optimizer import Optimizer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode

from hpe.model.backbone import Spice
from hpe.model.classifier import SimpleCNN, make_classifier
from hpe.model.linear import make_linear_classifiers
from hpe.model.clustering import make_clustering
from hpe.data.emgdataset import build_dataloaders
from hpe.loss.neuroloss import NeuroLoss
from hpe.loss.contrastive import NTXentLoss_poly
from hpe.utils.misc import plot_emg, plot_sample, plot_scatter_with_pca, plot_confussion


class MLP(nn.Module):
    def __init__(self, infeatures=128, outfeatures=16):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(infeatures, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outfeatures))
    def forward(self, x):
        return self.mlp_head(x)
        

class PlotSample(callbacks.Callback):
    def __init__(self, cfg, dataloaders, device):
        super().__init__()
        self.cfg = cfg
    
    def on_validation_epoch_end(self, trainer, pl_module):
        pass

def compute_topk(outputs, labels, k):
    _, topk_predictions = outputs.topk(k, dim=1)
    correct = topk_predictions.eq(labels.view(-1, 1).expand_as(topk_predictions))
    return correct.sum().item()

class EmgNet(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        #  get cfg from kwargs

        cfg = kwargs['cfg']
        self.cfg = cfg
        self.stage = cfg.STAGE

        # if self.stage == 'classify':
        #     dataloaders = build_dataloader_classification(cfg)
        # else:
        
        # cfg.DATA.PATH = '/Users/rufaelmarew/Documents/tau/finger_pose_estimation/dataset/emgleap'
        dataloaders = build_dataloaders(cfg)

        self.pretrain_loader = dataloaders['pretrain']
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']

        self.num_classes = dataloaders['train'].dataset.dataset.num_classes
        self.gestures = dataloaders['train'].dataset.dataset.gesture_mapping_class
        # self.gestures.remove('rest')

        self.model = Spice(input_size=16, proj_dim=128, d_model=cfg.MODEL.TRANSFORMER.D_MODEL, seq_length=cfg.DATA.SEGMENT_LENGTH, output_size=len(cfg.DATA.LABEL_COLUMNS))
        self.mlp_head = nn.Linear(cfg.MODEL.TRANSFORMER.D_MODEL*2, len(cfg.DATA.LABEL_COLUMNS))
        
        self.mlp_classifier = MLP(infeatures=len(cfg.DATA.LABEL_COLUMNS), outfeatures=self.num_classes)
        self.classifier = make_classifier(cfg)
        self.linear_classifiers = make_linear_classifiers(cfg)
        self.clustering_methods = make_clustering(cfg)

        # loss functions
        self.loss_fn = NeuroLoss(metric=cfg.SOLVER.METRIC, keypoints=cfg.DATA.LABEL_COLUMNS)
        self.criterion = torch.nn.L1Loss()
        self.classfication_criterion = torch.nn.CrossEntropyLoss()

        self.plot_output = 10
        self.train_step_output = []
        self.validation_step_output = []
        self.validation_step_target = []
        self.test_step_output = []
        self.test_step_target = []

        # freeze encoder_f
        # for param in self.model.encoder_f.parameters():
        #     param.requires_grad = False

        self.save_hyperparameters()

    def forward(self, x_t, x_f, target=None):
        h_t, h_f, z_t, z_f = self.model(x_t, x_f)
        #  concatinating the features
        h = torch.cat((h_t, h_f), dim=1)
        feat = self.mlp_head(h)
        # feat = h_t

        if target is not None:
            loss = self.loss_fn(feat, target)
            return feat, loss
        return feat, None
    
    def classification_step(self, batch, batch_idx, stage='train'):
        
        if self.cfg.MODEL.CLASSIFIER.NAME in ['mlp', 'cnn']:
            _, _, _, _, inputs, _, gestures = batch
        else:
            inputs, _, _, _, _, _, gestures = batch
            

        inputs = inputs.to(self.device)
        gestures = gestures.to(self.device)

        logits = self.classifier(inputs)
        loss = self.classfication_criterion(logits, gestures[:,0])
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == gestures[:,0]).item() / len(gestures)
        #  compute top k accuracy
        top_k_acc = []
        for k in [1, 3, 5]:
            top_k_acc.append(compute_topk(logits, gestures[:,0], k)/len(gestures))

        if 'test' in stage:
            self.log_dict({f'classification using {self.cfg.MODEL.CLASSIFIER.NAME} _{stage}_top_{k}_acc':v for k,v  in zip([1, 3, 5], top_k_acc)})
        else:
            self.log_dict({f'classification_{stage}_loss': loss, f'classification_{stage}_acc': acc})
        if 'test_0' in stage:
            self.test_step_output.append(pred.view(inputs.shape[0],-1).detach().cpu())
            self.test_step_target.append(gestures[:,0].detach().cpu())

        return loss
    
    def pretraining_step(self, batch, batch_idx):
        data_t, aug_t, data_f, aug_f, _, _ = batch
        data_t = data_t.to(self.device)
        aug_t = aug_t.to(self.device)
        data_f = data_f.to(self.device)
        aug_f = aug_f.to(self.device)

        h_t, h_f, z_t, z_f = self.model(data_t, data_f)
        h_t_aug, h_f_aug, z_t_aug, z_f_aug = self.model(aug_t, aug_f)

        nt_xent_criterion = NTXentLoss_poly(batch_size=self.cfg.SOLVER.BATCH_SIZE, temperature=self.cfg.SOLVER.TEMPERATURE, device=self.device, use_cosine_similarity=True)

        l_t = nt_xent_criterion(h_t, h_f_aug)
        l_f = nt_xent_criterion(h_f, h_t_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)

        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.2
        loss = lam*(l_t + l_f) + l_TF

        self.log_dict({'pretrain_loss': loss, 'pretrain_loss_t': l_t, 'pretrain_loss_f': l_f, 'pretrain_loss_TF': l_TF, 'pretrain_loss_c': loss_c} )

        return loss

    def finetune_step(self, batch, batch_idx, stage='train'):
        data_t, _, data_f, _, _, label, _ = batch
        data_t = data_t.to(self.device)
        data_f = data_f.to(self.device)
        label = label.to(self.device)

        # nt_xent_criterion = NTXentLoss_poly(batch_size=self.cfg.SOLVER.BATCH_SIZE, temperature=self.cfg.SOLVER.TEMPERATURE, device=self.device, use_cosine_similarity=True)

        h_t, h_f, z_t, z_f = self.model(data_t, data_f)
        
        # l_TF = nt_xent_criterion(z_t, z_f)
        l_TF = 0

        #concat the features
        h = torch.cat((h_t, h_f), dim=1)
        pred = self.mlp_head(h)
        loss_c = self.loss_fn(pred, label)

        #  total loss
        lam = 0.2
        l_T = loss_c[1] + lam*l_TF

        if self.current_epoch % self.plot_output == 0 and stage == 'val':
            self.validation_step_output.append(pred.detach().cpu())
            self.validation_step_target.append(label.detach().cpu())

        elif stage == 'test_0':
            self.test_step_output.append(loss_c[0].detach().cpu())
        
        loss_dict = {i: v for i, v in zip(self.loss_fn.keypoints, loss_c[0])}
        if stage == 'val':
            self.log_dict({f'{stage}_loss_c': loss_c[1], f'{stage}_loss_TF':l_TF, **loss_dict})
        else:
            self.log_dict({f'{stage}_loss_c': loss_c[1], f'{stage}_loss_TF':l_TF})

        return l_T
    
    def training_step(self, batch, batch_idx):

        if self.stage == 'pretrain':
            return self.pretraining_step(batch, batch_idx)
        elif self.stage == 'classify':
            return self.classification_step(batch, batch_idx, stage='train')
        elif self.stage == 'hpe':
            return self.finetune_step(batch, batch_idx, stage='train')
        else:
            raise NotImplementedError(f'Stage {self.stage} not implemented')

    def validation_step(self, batch, batch_idx):
        if self.stage == 'pretrain':
            # skip validation
            return
        elif self.stage == 'classify':
            return self.classification_step(batch, batch_idx, stage='val')
        else:
            return self.finetune_step(batch, batch_idx, stage='val')
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.stage == 'pretrain':
            # skip test
            return
        elif self.stage == 'classify':
            return self.classification_step(batch, batch_idx, stage=f'test_{dataloader_idx}')
        else:
            return self.finetune_step(batch, batch_idx, stage=f'test_{dataloader_idx}')
    
    def plot_bar(self):

        #  plot the scalar values of the logger as bar chart
        fig, ax = plt.subplots(1, 1, figsize=(20,5))
        #  stack the values
        out = torch.stack(self.test_step_output, dim=0).mean(dim=0)
        print(self.cfg.DATA.EXP_SETUP+','+','.join(map(str, out.tolist())), file=open('outputs/mean_values.csv', 'a'))

        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, c in enumerate(fingers):
            idx = [j for j in range(len(self.loss_fn.keypoints)) if c in self.loss_fn.keypoints[j].lower()]
            ax.bar(idx, out[idx], label=c)
            #  set xticks to be the cfg.label_columns
        ax.set_xticks(range(len(out)))
        ax.set_xticklabels(self.cfg.DATA.LABEL_COLUMNS, rotation=45)
        ax.legend()
        plt.tight_layout()
        self.logger.experiment.add_figure('final results', fig, self.current_epoch)
    
    
    def on_test_end(self) -> None:

        if self.stage == 'hpe':
            self.plot_bar()
            self.plot_sample_test(self.test_loader)
            self.linear_evaluation(self.train_dataloader(), self.test_dataloader())
            self.test_step_output = []


        elif self.stage == 'classify':
            # plot the confusion matrix with the values
            # y_pred = torch.argmax(torch.cat(self.test_step_output, dim=0), dim=1)
            y_pred = torch.cat(self.test_step_output, dim=0)
            y_true = torch.cat(self.test_step_target, dim=0)

            # disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=torch.unique(y_true), display_labels=self.gestures[torch.unique(y_true).tolist()])
            # disp.plot(xticks_rotation="vertical", cmap='Blues')
            fig = plot_confussion(y_true, y_pred)
            #  save the figure
            fig.savefig(f'outputs/confusion/{self.cfg.DATA.EXP_SETUP}_{self.cfg.MODEL.CLASSIFIER.NAME}.png')
            self.logger.experiment.add_figure('confusion matrix for classification', fig, self.current_epoch)
            acc = accuracy_score(y_true, y_pred)
            print(self.cfg.DATA.EXP_SETUP+','+self.cfg.MODEL.CLASSIFIER.NAME +','+str(acc), file=open('outputs/linear_results.csv', 'a'))
            self.test_step_output = []
            self.test_step_target = []

        else:
            return
    
        
    def get_features(self, loader):
        features, labels = [], []
        self.model.eval(), self.mlp_head.eval()
        for batch in tqdm(loader):
            data_t, _, data_f, _, _, _, label = batch
            data_t = data_t.to(self.device)
            data_f = data_f.to(self.device)

            feat, _ = self.forward(data_t, data_f)
            features.append(feat.detach().cpu())
            labels.append(label[:,0])

        return torch.cat(features, dim=0), torch.cat(labels, dim=0)
    

    def get_features_rms(self, loader):
        features, labels = [], []
        for batch in tqdm(loader):
            _, _, _, _, inputs, _, label = batch
            features.append(inputs.view(inputs.shape[0], -1).detach().cpu())
            labels.append(label[:,0])
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)
    
    def score_linear(self, model, model_name, test_features, test_labels, idx=0):

        acc = model.score(test_features, test_labels)
        if model_name == 'mlp':
            # compute the top k accuracy
            for k in [1, 3, 5]:
                top_k_acc=compute_topk(model.predict_proba(test_features), test_labels, k)/len(test_labels)
                self.logger.experiment.add_scalar(f'classification_test_{idx}_top_{k}_acc using {model_name}', top_k_acc, self.current_epoch)

        self.logger.experiment.add_scalar(f'classification acc after hpe and {model_name} on {idx}', acc, self.current_epoch)
        print(self.cfg.DATA.EXP_SETUP+','+model_name+','+str(acc), file=open('outputs/linear_results.csv', 'a'))
        
        # plot confussion matrix
        y_pred = model.predict(test_features)
        # disp = ConfusionMatrixDisplay.from_predictions(test_labels, y_pred, labels=torch.unique(test_labels), display_labels=self.gestures[torch.unique(test_labels).tolist()])
        # disp.plot(xticks_rotation="vertical", cmap='Blues')
        fig = plot_confussion(test_labels, y_pred)
        #  save the figure
        fig.savefig(f'outputs/confusion/{self.cfg.DATA.EXP_SETUP}_hpe_{model_name}.png')
        self.logger.experiment.add_figure(f'confusion matrix after hpe and {model_name} on {idx}', fig, self.current_epoch)
        
        return acc
    

    def score_cluster(self, clust, clust_name, test_features, test_labels, idx=0):
            
        test_clusters = clust.predict(test_features)

        preds = np.zeros_like(test_labels)
        for i in range(self.num_classes):
            mask = (test_clusters == i)
            preds[mask] = mode(test_labels[mask], keepdims=True)[0]
        
        acc = accuracy_score(test_labels, preds)
        fig = plot_confussion(test_labels, preds)
        fig.savefig(f'outputs/confusion/{self.cfg.DATA.EXP_SETUP}_hpe_{clust_name}.png')
        self.logger.experiment.add_figure(f'confusion matrix after hpe and {clust_name} on {idx}', fig, self.current_epoch)
        self.logger.experiment.add_scalar(f'classification acc after hpe and {clust_name} on {idx}', acc, self.current_epoch)
        
        print(self.cfg.DATA.EXP_SETUP+','+clust_name+','+str(acc), file=open('outputs/linear_results.csv', 'a'))

    def linear_evaluation(self, train_loader, test_loader):

        #  get features
        train_features, train_labels = self.get_features(train_loader)

        # # plot the scatter plot
        # fig, _ = plot_scatter_with_pca(train_features, train_labels, [gesture for gesture in self.gestures if gesture != 'rest'])
        # fig.savefig(f'outputs/scatter/scatter_plot_hpe_{self.cfg.DATA.EXP_SETUP}.png')
        # self.logger.experiment.add_figure('scatter plot after hpe', fig, self.current_epoch)
        # self.logger.experiment.add_figure('legend', fig_legend, self.current_epoch)
        # save plot
       

        for model_name, model in self.linear_classifiers.items():
            print(f'Evaluating the model using {model_name}')
            model.fit(train_features, train_labels)
            for i, v in enumerate(test_loader):
                test_features, test_labels = self.get_features(v)
                # plot the scatter plot
                fig, _ = plot_scatter_with_pca(test_features, test_labels, [gesture for gesture in self.gestures if gesture != 'rest'])
                fig.savefig(f'outputs/scatter_2/scatter_plot_hpe_{self.cfg.DATA.EXP_SETUP}.png')
                acc = self.score_linear(model, model_name, test_features, test_labels, i)
        
        # clustering evaluation
        for clust_name, clust in self.clustering_methods.items():
            print(f'Evaluating the model using {clust_name}')
            clust.fit(train_features)
            for i, v in enumerate(test_loader):
                test_features, test_labels = self.get_features(v)
                acc = self.score_cluster(clust, clust_name, test_features, test_labels, i)

    
    def plot_sample_test(self, test_loader):
        #  get the first batch
        batch = next(iter(test_loader))
        data_t, _, data_f, _, _, label, gesture = batch

        # get the features
        data_t = data_t.to(self.device)
        data_f = data_f.to(self.device)

        # plot the emg data
        start = np.random.randint(0, len(data_t)-10)
        fig_emg = plot_emg(data_t[start:start+2,:,:].view(-1, len(self.loss_fn.keypoints)).detach().cpu())
        self.logger.experiment.add_figure('test emg data', fig_emg, self.current_epoch)
        
        feat, _ = self.forward(data_t, data_f)
        feat = feat.detach().cpu().view(-1, len(self.loss_fn.keypoints))
        label = label[:,-1,:].view(-1, len(self.loss_fn.keypoints))

        fig = plot_sample(feat, label, self.loss_fn.keypoints)
        self.logger.experiment.add_figure('test sample', fig, self.current_epoch)

        # plot the scatter plot
        feat_rms, label_rms = self.get_features_rms(test_loader)
        fig, _ = plot_scatter_with_pca(feat_rms, label_rms, [gesture for gesture in self.gestures if gesture != 'rest'])
        self.logger.experiment.add_figure('scatter plot RMS', fig, self.current_epoch)
        # self.logger.experiment.add_figure('legend', fig_legend, self.current_epoch)
        # save plot
        fig.savefig(f'outputs/scatter_2/scatter_plot_rms_{self.cfg.DATA.EXP_SETUP}.png')

    def on_validation_epoch_end(self) -> None:
        # plot a sample output
        from matplotlib import pyplot as plt
        if self.stage == 'classify' or self.stage == 'pretrain': 
            return
        
        if len(self.validation_step_output)!=0 and len(self.validation_step_output[0].shape) < 3 and self.current_epoch % self.plot_output == 0:
            pred = torch.concatenate(self.validation_step_output, dim=0).view(-1, self.validation_step_output[0].shape[-1])
            target = torch.concatenate(self.validation_step_target, dim=0)[:,-1,:].view(-1, self.validation_step_target[0].shape[-1])
            
            fig = plot_sample(pred, target, self.loss_fn.keypoints)
            self.logger.experiment.add_figure('validation sample', fig, self.current_epoch)

        self.validation_step_target = []
        self.validation_step_output = []
    
    
    def configure_optimizers(self):
        if self.stage== 'pretrain':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.LR)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                    'monitor': 'pretrain_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.stage == 'classify':
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.cfg.SOLVER.LR)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                    'monitor': 'classification_val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            #  remove the projection head from the model
            # self.model.projector_f = nn.Identity()
            # self.model.projector_t = nn.Identity()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.LR)
            optimizer.add_param_group({'params': self.mlp_head.parameters()})
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                    'monitor': 'val_loss_c',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }


    def train_dataloader(self):
        if self.stage == 'pretrain':
            return self.pretrain_loader
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return [self.test_loader]
    

    
class EmgNetPretrain(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        
        cfg = kwargs['cfg']
        self.cfg = cfg

        #  setup dataloaders
        self.stage = cfg.STAGE
        dataloaders = build_dataloaders(cfg)
        self.pretrain_loader = dataloaders['pretrain']
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        self.test_2_loader = dataloaders['test_2']

        #  setup model
        self.backbone_t = build_backbone(cfg).to(self.device)
        self.backbone_f = build_backbone(cfg).to(self.device)
        # self.backbone_t = make_test(cfg).to(self.device)
        # self.backbone_f = make_test(cfg).to(self.device)
        infeat_t =  (self.backbone_f.d_model)
        infeat_f = (self.backbone_t.d_model)
        self.mlp = MLP(infeatures=infeat_t+infeat_f, outfeatures=len(self.cfg.DATA.LABEL_COLUMNS)).to(self.device)

        # self.backbone_f.mlp_head = nn.Identity()
        # self.backbone_t.mlp_head = nn.Identity()
            

        self.projector_t = nn.Sequential(
            nn.Linear(infeat_t, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)

        self.projector_f = nn.Sequential(
            nn.Linear(infeat_f, 256),
            nn.ReLU(),
            nn.Linear(256, 128)

        ).to(self.device)

        # setup loss
        self.loss_fn = make_loss(cfg)

        self.plot_output = 10
        self.validation_step_output = []
        self.validation_step_target = []

        self.save_hyperparameters()

    def forward(self, x_t, x_f):

        x, o_t = self.backbone_t(x_t)
        h_t = x.view(x.size(0), -1)

        z_t = self.projector_t(h_t)

        f, o_f = self.backbone_f(x_f)
        h_f = f.view(f.size(0), -1)

        z_f = self.projector_f(h_f)
        
        return h_t, o_t, z_t, h_f, o_f, z_f
    
    def pretrain_step(self, batch, batch_idx):

        data, aug1, data_f, aug1_f, _, _ = batch
        data = data.to(self.device)
        aug1 = aug1.to(self.device)
        data_f = data_f.to(self.device)
        aug1_f = aug1_f.to(self.device)

        h_t, o_t, z_t, h_f, o_f, z_f = self.forward(data, data_f)
        h_t_aug, _ , z_t_aug, h_f_aug, _, z_f_aug = self.forward(aug1, aug1_f)

        nt_xent_criterion = NTXentLoss_poly(batch_size=self.cfg.SOLVER.BATCH_SIZE, temperature=self.cfg.SOLVER.TEMPERATURE, device=self.device, use_cosine_similarity=True)
        
        l_t = nt_xent_criterion(h_t, h_f_aug)
        l_f = nt_xent_criterion(h_f, h_t_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)

        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.2
        loss = lam*(l_t + l_f) + loss_c

        self.log_dict({'pretrain_loss': loss, 'pretrain_loss_t': l_t, 'pretrain_loss_f': l_f, 'pretrain_loss_TF': l_TF, 'pretrain_loss_c': loss_c} )
        return loss
    
    def finetune_step(self, batch, batch_idx, stage='train'):
        data, _, data_f, _, label, _ = batch
        data = data.to(self.device)
        data_f = data_f.to(self.device)
        label = label.to(self.device)

        t, o_t = self.backbone_t(data)
        h_t = t.view(t.size(0), -1)
        f, o_f = self.backbone_f(data_f)
        h_f = f.view(f.size(0), -1)

        #  compute the simCLR loss
        nt_xent_criterion = NTXentLoss_poly(batch_size=self.cfg.SOLVER.BATCH_SIZE, temperature=self.cfg.SOLVER.TEMPERATURE, device=self.device, use_cosine_similarity=True)
        l_TF = nt_xent_criterion(h_t, h_f)

        #compute the time and freq loss
        # l_t = self.loss_fn(o_t, label)
        # l_f = self.loss_fn(o_f, label)

        # concatinate the features
        h = torch.cat((h_t, h_f), dim=1)
        pred = self.mlp(h)
        loss_c = self.loss_fn(pred, label)

        #  total loss
        l_T = l_TF  + loss_c[1]

        if self.current_epoch % self.plot_output == 0 and stage == 'val':
            self.validation_step_output.append(pred.detach().cpu())
            self.validation_step_target.append(label.detach().cpu())

        loss_dict = {i: v for i, v in zip(self.loss_fn.keypoints, loss_c[0])}
        if stage == 'val':
            self.log_dict({f'{stage}_loss_c': loss_c[1], f'{stage}_loss_TF':l_TF, **loss_dict})
        else:
            self.log_dict({f'{stage}_loss_c': loss_c[1], f'{stage}_loss_TF':l_TF})
        return loss_c[1]

    
    def training_step(self, batch, batch_idx):
        if self.stage == 'pretrain':
            return self.pretrain_step(batch, batch_idx)
        
        else:
            return self.finetune_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        if self.stage == 'pretrain':
            # skip validation
            return
        else:
            return self.finetune_step(batch, batch_idx, stage='val')
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        if self.stage == 'pretrain':
            # skip test
            return
        else:
            return self.finetune_step(batch, batch_idx, stage=f'test_{dataloader_idx}')
    
    def on_validation_epoch_end(self) -> None:
        # plot a sample output
        from matplotlib import pyplot as plt 
        
        if len(self.validation_step_output)!=0 and len(self.validation_step_output[0].shape) < 3 and self.current_epoch % self.plot_output == 0:
            pred = torch.concatenate(self.validation_step_output, dim=0).view(-1, self.validation_step_output[0].shape[-1])
            target = torch.concatenate(self.validation_step_target, dim=0)[:,0,:].view(-1, self.validation_step_target[0].shape[-1])
            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

            fig, ax = plt.subplots(fingers.__len__(), 1, figsize=(20,10))
            #  randomly sample the start and end of seq
            start = torch.randint(0, pred.shape[0]-100, (1,)).item()
            end = start + 100
            # t = min(100, pred.shape[0])
            #  smaple chunk of the data with label

            # idx_x = torch.randperm(pred.shape[0])[:t]
            #  compute average for each finger
            for i, c in enumerate(fingers):
                idx = [j for j in range(len(self.loss_fn.keypoints)) if c in self.loss_fn.keypoints[j].lower()]
                ax[i].plot(pred[start:end,:][:,idx].mean(dim=1))
                ax[i].plot(target[start:end,:][:,idx].mean(dim=1))
                ax[i].set_title(c)
                #  show legend only for the first plot
                if i == 0:
                    ax[i].legend(['pred', 'target'])
            self.logger.experiment.add_figure('validation sample', fig, self.current_epoch)
            del pred, target

        self.validation_step_target = []
        self.validation_step_output = []
    
    def configure_optimizers(self):
        if self.stage== 'pretrain':
            optimizer = torch.optim.Adam(self.backbone_t.parameters(), lr=self.cfg.SOLVER.LR)
            optimizer.add_param_group({'params': self.projector_t.parameters()})
            optimizer.add_param_group({'params': self.backbone_f.parameters()})
            optimizer.add_param_group({'params': self.projector_f.parameters()})
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                    'monitor': 'pretrain_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            optimizer = torch.optim.Adam(self.backbone_t.parameters(), lr=self.cfg.SOLVER.LR)
            optimizer.add_param_group({'params': self.mlp.parameters()})
            optimizer.add_param_group({'params': self.backbone_f.parameters()})
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                    'monitor': 'val_loss_c',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }


    def train_dataloader(self):
        if self.stage == 'pretrain':
            return self.pretrain_loader
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return [self.test_loader, self.test_2_loader]