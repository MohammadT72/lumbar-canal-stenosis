
# ********* some of the functions in this block of code are sourced from the Monai library. *********
# ---------------------------------------------------------------------------------------------------
# ---- Copyright (c) MONAI Consortium
# ---- Licensed under the Apache License, Version 2.0 (the "License");
# ---- you may not use this file except in compliance with the License.
# ---- You may obtain a copy of the License at
# ----     http://www.apache.org/licenses/LICENSE-2.0
# ---- Unless required by applicable law or agreed to in writing, software
# ---- distributed under the License is distributed on an "AS IS" BASIS,
# ---- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ---- See the License for the specific language governing permissions and
# ---- limitations under the License.

from __future__ import annotations
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchmetrics as tm
import scikitplot as skplt
import torch
import pandas as pd






class metrics_calculator:
    def __init__(self, metrics:list,num_class):
        self.metrics = metrics
        self.num_class=num_class
        self.auc_per_class=tm.AUROC(task='multiclass', num_classes=self.num_class, average='none')
        self.auc_macro=tm.AUROC(task='multiclass', num_classes=self.num_class, average='macro')
        self.auc_weighted=tm.AUROC(task='multiclass', num_classes=self.num_class, average='weighted')
        self.y_prob=None
        self.y=None
    def aggregate(self, y_pred:torch.tensor, y:torch.tensor):
        ''' a function based on metric calculation fuction of MONAI for metric
        calculation per class and per bacth
        y_pred: predictions, it must raw output of the model
        y: True labels, it must not be one-hot encoded
        metrics: a list of names for available metrics
    '''
        self.y_prob=y_pred
        self.y=y
        y_pred_one_hot = F.one_hot(y_pred.argmax(dim=-1),num_classes=self.num_class).to(torch.int64)
        y_one_hot = F.one_hot(y.to(torch.int64),num_classes=self.num_class)
        weights = torch.bincount(y.to(torch.int64))/len(y)
        # calculate the confusion matrix
        cm = get_confusion_matrix(y_pred_one_hot,y_one_hot)
        # total confusion matrix
        total_cm = cm.sum(dim=[0,1])
        metric_dict = {}
        # for each metric, per class and per batch
        for metric_name in self.metrics:
            if metric_name.lower()=='auc':
                metric_dict[metric_name]={}
                # if not isinstance(y_pred,type(torch.Tensor())): # handle this
                # y_pred=y_pred.as_tensor()
                # y=y.as_tensor()
                per_class=self.auc_per_class(y_pred.float(),y.int())
                metric_dict[metric_name]['macro-avg']=self.auc_macro(y_pred.float(),y.int()).item()
                metric_dict[metric_name]['weighted-avg']=self.auc_weighted(y_pred.float(),y.int()).item()
                for idx in range(self.num_class):
                    metric_dict[metric_name][f'class-{idx}']=per_class[idx].item()
            else:
                # per class
                per_class_values = compute_confusion_matrix_metric(metric_name=metric_name, confusion_matrix=cm.sum(dim=0))
                # micro average
                micro_avg = compute_confusion_matrix_metric(metric_name=metric_name, confusion_matrix=cm.sum(dim=[0,1]))
                # macro average
                macro_avg = torch.sum(per_class_values)/len(per_class_values)
                # weighted
                weighted_avg = torch.sum(torch.mul(per_class_values,weights))
                metric_dict[metric_name]={}
                for idx in range(len(per_class_values)):
                    metric_dict[metric_name][f'class-{idx}'] = per_class_values[idx].item()
                for idx in range(len(weights)):
                    metric_dict[metric_name][f'class-{idx}-weight'] = weights[idx].item()
                metric_dict[metric_name]['micro-avg']=micro_avg.item()
                metric_dict[metric_name]['macro-avg']=macro_avg.item()
                metric_dict[metric_name]['weighted-avg']=weighted_avg.item()
                metric_dict[metric_name]['TP']=total_cm[0].item()
                metric_dict[metric_name]['FP']=total_cm[1].item()
                metric_dict[metric_name]['TN']=total_cm[2].item()
                metric_dict[metric_name]['FN']=total_cm[3].item()
        return metric_dict
    def plot(self,labels:dict,path='/content',name='cm',cmap='Blues',figsize=(15,15)):
        y_pred=self.y_prob.argmax(dim=-1)
        skplt.metrics.plot_confusion_matrix(self.y, y_pred,normalize=False,cmap=cmap,figsize=figsize)
        plt.yticks(ticks = list(labels.keys()) ,labels = list(labels.values()), rotation = 'horizontal')
        plt.xticks(ticks = list(labels.keys()) ,labels = list(labels.values()), rotation = 'vertical')
        plt.savefig(fname=Path(path).joinpath(f'{name}_Confusion_matrix.png'),dpi=300)
        skplt.metrics.plot_confusion_matrix(self.y, y_pred,normalize=True,cmap=cmap,figsize=figsize)
        plt.yticks(ticks = list(labels.keys()) ,labels = list(labels.values()), rotation = 'horizontal')
        plt.xticks(ticks = list(labels.keys()) ,labels = list(labels.values()), rotation = 'vertical')
        plt.savefig(fname=Path(path).joinpath(f'{name}_Normalized_confusion_matrix.png'),dpi=300)
        skplt.metrics.plot_roc(self.y, self.y_prob,figsize=figsize, cmap='viridis')
        plt.savefig(fname=Path(path).joinpath(f'{name}_roc_curve.png'),dpi=300)
        plt.show()
  

class k_fold_plotter:
    def __init__(self,folder_path, save=True,show=False,prefix='plot', k_fold=True):
        self.df=None
        self.folder_path=folder_path
        self.save=save
        self.show=show
        self.prefix=prefix
        self.k_fold=k_fold
    def load_logs(self,log_id):
        self.df=pd.read_csv(Path(self.folder_path).joinpath('runs','logs',f'{log_id}.csv'),index_col=0).reset_index(drop=True)
        self.df=self.clean_logs()
    def clean_logs(self):
        count = np.isinf(self.df.iloc[:,1:]).values.sum()
        if count>0:
            self.df=self.df.replace([np.inf, -np.inf], np.nan)
        df=self.df.iloc[:,1:].astype(dtype=np.float32)
        df=df.fillna(method='ffill')
        df=df.fillna(method='bfill')
        max_epoch_fold_index=df.groupby(by='fold_num').std()['epoch_num'].idxmax()
        max_epoch=len(df[df['fold_num']==max_epoch_fold_index])
        temp_df=pd.DataFrame.from_dict(dict.fromkeys(df.columns,[]))
        for fold in df['fold_num'].unique():
          fold_df=df[df['fold_num']==fold]
          epochs_num=len(fold_df)
          diff=max_epoch-epochs_num
          last_row=fold_df.iloc[-1,:]
          fold_df=pd.concat([fold_df]+[last_row.to_frame().T]*diff, ignore_index=True)
          fold_df['epoch_num']=list(range(max_epoch))
          temp_df=pd.concat([temp_df,fold_df],ignore_index=True)
        return temp_df
    def plot_metrics(self,search_id):
        list_metrics=pd.unique([x.split('_')[0] for x in self.df.columns[4:]])
        # calculate mean and standard deviation for each epoch
        min_epoch_fold_index=self.df.groupby(by='fold_num').std()['epoch_num'].idxmin()
        min_epoch=len(self.df[self.df['fold_num']==min_epoch_fold_index])
        means = self.df.groupby(by='epoch_num').mean().iloc[:min_epoch,:]
        stds = self.df.groupby(by='epoch_num').std().iloc[:min_epoch,:]
        x=list(range(min_epoch))
        selected={}
        for metric in list_metrics:
            selected[metric]={}
            list_avg=['_micro-avg','_macro-avg','_weighted-avg']
            if metric.lower()=='auc':
                list_avg=['_macro-avg','_weighted-avg']
            for avg in list_avg:
                metric_avg=metric+avg
                val_mean = means[metric_avg]
                val_std = stds[metric_avg]
                selected[metric][metric_avg]={'mean':val_mean,'std':val_std}

        # plot the training and validation losses with mean and standard deviation
        for key in selected.keys():
            selected_metric=selected[key]
            if key.lower()!='auc':
                micro_mean=selected_metric[key+'_micro-avg']['mean']
                micro_std=selected_metric[key+'_micro-avg']['std']
            macro_mean=selected_metric[key+'_macro-avg']['mean']
            macro_std=selected_metric[key+'_macro-avg']['std']
            weighted_mean=selected_metric[key+'_weighted-avg']['mean']
            weighted_std=selected_metric[key+'_weighted-avg']['std']
            # plt.figure(figsize=(10,5))
            if key.lower()!='auc': 
                plt.plot(x,micro_mean, label=f'Micro average = {micro_mean.max():.2f}')
                plt.fill_between(range(min_epoch), micro_mean- micro_std, micro_mean + micro_std, alpha=0.1)
            plt.plot(x,macro_mean, label=f'Macro average = {macro_mean.max():.2f}')
            plt.fill_between(range(min_epoch), macro_mean- macro_std, macro_mean + macro_std, alpha=0.1)
            plt.plot(x,weighted_mean, label=f'weighted average = {weighted_mean.max():.2f}')
            plt.fill_between(range(min_epoch), weighted_mean- weighted_std, weighted_mean + weighted_std, alpha=0.1)
            # plt.xticks(ticks=x, labels=x)
            plt.ylim(0,1)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            if self.k_fold:
              plt.title(f'Validation {key}\nfor 10-fold cross-validation')
            else:
              plt.title(f'Validation {key}\n')
            plt.legend()
            if self.save:
                target_path = Path(self.folder_path).joinpath('runs','plots',search_id)
                target_path.mkdir(parents=True,exist_ok=True)
                plt.savefig(str(target_path.joinpath(key + '.png')), dpi=300)
            if self.show:
                plt.show()
            plt.clf()
            plt.close()
    def plot_loss(self,search_id, best_epoch):
        # calculate mean and standard deviation for each epoch
        min_epoch_fold_index=self.df.groupby(by='fold_num').std()['epoch_num'].idxmin()
        min_epoch=len(self.df[self.df['fold_num']==min_epoch_fold_index])
        means = self.df.groupby(by='epoch_num').mean().iloc[:min_epoch,:]
        stds = self.df.groupby(by='epoch_num').std().iloc[:min_epoch,:]
        x=list(range(min_epoch))
        mean_train_loss=means['train_loss']
        std_train_loss=stds['train_loss']
        mean_val_loss=means['val_loss']
        std_val_loss=stds['val_loss']
        # plot the training and validation losses with mean and standard deviation
        plt.plot(x,mean_train_loss, label='Train loss')
        plt.fill_between(range(min_epoch), mean_train_loss- std_train_loss, mean_train_loss + std_train_loss, alpha=0.1)
        plt.plot(x,mean_val_loss, label='Validation loss')
        plt.fill_between(range(min_epoch), mean_val_loss- std_val_loss, mean_val_loss + std_val_loss, alpha=0.1)
        # plt.xticks(ticks=x, labels=x)
        upper_limit=mean_val_loss.max()
        # if (not isinstance(mean_val_loss.max(),float)) or (not isinstance(mean_val_loss.max(),type(torch.float32))):
        #     upper_limit=torch.from_numpy(upper_limit)
        plt.ylim(0,upper_limit)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        if self.k_fold:
            plt.title(f'Train and validation loss\nfor 10-fold cross-validation')
        else:
            plt.title(f'Train and validation loss\n')
        plt.axvline(x=best_epoch,linestyle='dotted')
        plt.annotate('Early stopping check point', xy=(int(best_epoch/2),
        upper_limit-upper_limit*0.2))
        plt.legend()
        if self.save:
            target_path = Path(self.folder_path).joinpath('runs','plots',search_id)
            target_path.mkdir(parents=True,exist_ok=True)
            plt.savefig(str(target_path.joinpath('learning_curve.png')), dpi=300)
        if self.show:
            plt.show()
        plt.clf()
        plt.close()
    def plot_class_metrics(self,classes_names:dict,search_id, num_class=5):
        list_metrics=pd.unique([x.split('_')[0] for x in self.df.columns[4:]])
        # calculate mean and standard deviation for each epoch
        min_epoch_fold_index=self.df.groupby(by='fold_num').std()['epoch_num'].idxmin()
        min_epoch=len(self.df[self.df['fold_num']==min_epoch_fold_index])
        means = self.df.groupby(by='epoch_num').mean().iloc[:min_epoch,:]
        stds = self.df.groupby(by='epoch_num').std().iloc[:min_epoch,:]
        x=list(range(min_epoch))
        selected={}
        for metric in list_metrics:
            selected[metric]={}
            for class_idx in range(num_class):
                metric_class_name=metric+f'_class-{class_idx}'
                val_mean = means[metric_class_name]
                val_std = stds[metric_class_name]
                selected[metric][metric_class_name]={'mean':val_mean,'std':val_std}

        # plot the validation metrics with mean and standard deviation
        for key in selected.keys():
            selected_metric=selected[key]
            # plt.figure(figsize=(8,5))
            for idx in range(num_class):    
                class_mean=selected_metric[key+f'_class-{idx}']['mean']
                class_std=selected_metric[key+f'_class-{idx}']['std']
                plt.plot(x,class_mean, label=f'Class {classes_names[idx]} = {class_mean.max():.2f}')
                plt.fill_between(range(min_epoch), class_mean- class_std, class_mean + class_std, alpha=0.1)
            # plt.xticks(ticks=x, labels=x)
            plt.ylim(0,1)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            if self.k_fold:
              plt.title(f'Validation {key} per class\nfor 10-fold cross-validation')
            else:
              plt.title(f'Validation {key} per class\n')
            plt.legend()
            if self.save:
                target_path = Path(self.folder_path).joinpath('runs','plots',search_id)
                target_path.mkdir(parents=True,exist_ok=True)
                plt.savefig(str(target_path.joinpath(f'class_{key}.png')), dpi=300)
            if self.show:
                plt.show()
            plt.clf()
            plt.close()


def get_confusion_matrix(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute confusion matrix. A tensor with the shape [BC4] will be returned. Where, the third dimension
    represents the number of true positive, false positive, true negative and false negative values for
    each channel of each sample within the input batch. Where, B equals to the batch size and C equals to
    the number of classes that need to be computed.

    Args:
        y_pred: input data to compute. It must be one-hot format and first dim is batch.
            The values should be binarized.
        y: ground truth to compute the metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to include metric computation on the first channel of
            the predicted output. Defaults to True.

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    # get confusion matrix related metric
    batch_size, n_class = y_pred.shape[:2]
    # convert to [BNS], where S is the number of pixels for one sample.
    # As for classification tasks, S equals to 1.
    y_pred = y_pred.reshape(batch_size, n_class, -1)
    y = y.reshape(batch_size, n_class, -1)
    tp = ((y_pred + y) == 2).float()
    tn = ((y_pred + y) == 0).float()

    tp = tp.sum(dim=[2])
    tn = tn.sum(dim=[2])
    p = y.sum(dim=[2])
    n = y.shape[-1] - p

    fn = p - tp
    fp = n - tn

    return torch.stack([tp, fp, tn, fn], dim=-1)


def compute_confusion_matrix_metric(metric_name: str, confusion_matrix: torch.Tensor) -> torch.Tensor:
    """
    This function is used to compute confusion matrix related metric.

    Args:
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
            ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
            ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
            ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
            ``"informedness"``, ``"markedness"``]
            Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
            and you can also input those names instead.
        confusion_matrix: Please see the doc string of the function ``get_confusion_matrix`` for more details.

    Raises:
        ValueError: when the size of the last dimension of confusion_matrix is not 4.
        NotImplementedError: when specify a not implemented metric_name.

    """

    metric = check_confusion_matrix_metric_name(metric_name)

    input_dim = confusion_matrix.ndimension()
    if input_dim == 1:
        confusion_matrix = confusion_matrix.unsqueeze(dim=0)
    if confusion_matrix.shape[-1] != 4:
        raise ValueError("the size of the last dimension of confusion_matrix should be 4.")

    tp = confusion_matrix[..., 0]
    fp = confusion_matrix[..., 1]
    tn = confusion_matrix[..., 2]
    fn = confusion_matrix[..., 3]
    p = tp + fn
    n = fp + tn

    tp=torch.where(tp==0.0,1.0,tp)
    tn=torch.where(tn==0.0,1.0,tn)
    fp=torch.where(fp==0.0,1.0,fp)
    fn=torch.where(fn==0.0,1.0,fn)
    # calculate metric
    numerator: torch.Tensor
    denominator: torch.Tensor | float
    nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)
    if metric == "tpr":
        numerator, denominator = tp, p
    elif metric == "tnr":
        numerator, denominator = tn, n
    elif metric == "ppv":
        numerator, denominator = tp, (tp + fp)
    elif metric == "npv":
        numerator, denominator = tn, (tn + fn)
    elif metric == "fnr":
        numerator, denominator = fn, p
    elif metric == "fpr":
        numerator, denominator = fp, n
    elif metric == "fdr":
        numerator, denominator = fp, (fp + tp)
    elif metric == "for":
        numerator, denominator = fn, (fn + tn)
    elif metric == "pt":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
        denominator = tpr + tnr - 1.0
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tn), (p + n)
    elif metric == "ba":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator, denominator = (tpr + tnr), 2.0
    elif metric == "f1":
        numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        numerator = torch.sqrt(ppv * tpr)
        denominator = 1.0
    elif metric == "bm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = tpr + tnr - 1.0
        denominator = 1.0
    elif metric == "mk":
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
        numerator = ppv + npv - 1.0
        denominator = 1.0
    else:
        raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator


def check_confusion_matrix_metric_name(metric_name: str) -> str:
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    if metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    if metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    if metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    if metric_name in ["false_omission_rate", "for"]:
        return "for"
    if metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    if metric_name in ["accuracy", "acc"]:
        return "acc"
    if metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    if metric_name in ["f1_score", "f1"]:
        return "f1"
    if metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    if metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    if metric_name in ["informedness", "bookmaker_informedness", "bm", "youden_index", "youden"]:
        return "bm"
    if metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    raise NotImplementedError("the metric is not implemented.")