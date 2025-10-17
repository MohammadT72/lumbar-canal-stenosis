from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
from print_color import print
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingLR,ReduceLROnPlateau

class trainer:
    def __init__(self,train_data_loader, valid_data_loader,save_models,
                model,optimizer,loss_function, max_epochs,with_scheduler,
                class_col_name,image_key,
                metric_calculator,amp=True, val_interval=1,device='cpu',show_interval=10,
                l1_regulariztion_lambda=0.0,l2_regulariztion_lambda=0.0,
                early_stop_callback=True,patience=2,early_stop_verbose=True,
                early_stop_delta=0, save_path='/content',class_num=5,
                grid_search_ID=0,fold_num=0):
        """
        Initializes the trainer class with various parameters.
        train_data_loader: training data loader
        valid_data_loader: validation data loader
        save_models: flag to save models during training loop (True or False)
        model: model object
        optimizer: optimizer object
        loss_function: loss function object
        max_epochs: maximum number of epochs
        metric_calculator: metric calculator object
        amp: flag for automatic mixed precision (True or False)
        val_interval: validation interval for training loop
        device: device for training (cpu or cuda)
        early_stop_callback: flag for early stop callback (True or False)
        patience: patience value for early stop callback
        early_stop_verbose: flag for verbose output of early stop callback (True or False)
        early_stop_delta: delta value for early stop callback
        save_path: path to save models during training loop
        class_num: number of classes in the dataset
        grid_search_ID: ID of the grid search run
        fold_num: fold number for cross-validation
        """
        self.train_loader = train_data_loader
        self.val_loader = valid_data_loader
        self.model = model
        self.optimizer = optimizer
        self.with_scheduler=with_scheduler
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.class_num= class_num
        self.class_col_name=class_col_name
        self.image_key=image_key
        self.metric_calculator = metric_calculator
        self.train_num_batch = len(train_data_loader)
        self.early_stop_callback=early_stop_callback
        self.l1_regulariztion_lambda=l1_regulariztion_lambda
        self.l2_regulariztion_lambda=l2_regulariztion_lambda
        self.val_num_batch = len(valid_data_loader)
        self.val_interval = val_interval
        self.show_interval=show_interval
        self.device=device
        self.save_models=save_models
        self.root_dir = '/content'
        try:
          self.enabled = True if 'cuda' in device.type else False
        except:
          self.enabled = False
        if not amp:
            self.enabled = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
        self.patience = patience
        self.verbose = early_stop_verbose
        self.delta = early_stop_delta
        self.grid_search_ID = grid_search_ID
        self.fold_num = fold_num
        self.main_dir = Path(save_path).joinpath('runs')
        self.log_dir = Path(self.main_dir).joinpath('logs')
        self.plot_dir = Path(self.main_dir).joinpath('plots')
        self.model_dir = Path(self.main_dir).joinpath('models')

        if self.with_scheduler:
          self.scheduler=MultiStepLR(self.optimizer,[5,15,25],0.1)
          self.scheduler=ReduceLROnPlateau(self.optimizer, mode='min', 
          factor=0.1, patience=5, threshold=0.01,verbose=True)
          # self.scheduler=CosineAnnealingLR(self.optimizer,
          # T_max=self.max_epochs,eta_min=float(self.optimizer.defaults['lr'])*10e-3, verbose=True)
        self.best_mcc = -1
        self.best_mcc_epoch = -1
        self.logs = {}
        self.val_loss_min = np.Inf
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch=0

    def reset_trainer(self):
        # --------------- RESET TRAINER --------------- #
        self.best_mcc = -1
        self.best_mcc_epoch = -1
        self.logs = {}
        self.val_loss_min = np.Inf
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def make_dirs(self):
        # --------------- CREATE DIRECTORIES --------------- #
        # Create directories for saving logs, plots and models
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def early_stop_check(self, input_value, loss, model, epoch):
        # --------------- EARLY STOPPING --------------- #
        score=input_value
        if loss:
          score=-input_value
        if self.best_score is None:
          self.best_score=score
          self.best_epoch=epoch
          if self.save_models:
              self.save_checkpoint(model, best=True)
        elif (score + self.delta) < self.best_score:
              self.counter += 1
              print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
              # Stop training if validation loss does not improve for patience number of epochs
              if self.counter >= self.patience:
                  if self.save_models:
                      self.save_checkpoint(model, best=False)
                  self.early_stop = True
        else:
            self.best_score=score
            self.best_epoch=epoch
            # Save model if validation loss is minimum so far
            if self.save_models:
                self.save_checkpoint(model)
            self.counter = 0
    def save_checkpoint(self, model, best=True):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased.  Saving model ...')
        # Save model state dictionary to file
        if best:
          model_path=self.model_dir.joinpath(f'{self.grid_search_ID}_{self.fold_num}_best.pt')
        else:
          model_path=self.model_dir.joinpath(f'{self.grid_search_ID}_{self.fold_num}_final.pt')
        torch.save(model.state_dict(), model_path)


    def add_l1_l2_regularization(self,loss):
        # L1 regularization
        l1_norm = torch.norm(torch.cat([x.view(-1) for x in self.model.parameters()]), 1)
        print(l1_norm)
        # L2 regularization
        l2_norm = torch.norm(torch.cat([x.view(-1) for x in self.model.parameters()]), 2)
        print(l2_norm)
        return loss + (self.l1_regulariztion_lambda*l1_norm) + (self.l2_regulariztion_lambda*l2_norm)

    def train(self,epoch):
        # --------------- TRAIN --------------- #
        # Print epoch number
        print("-" * 10)
        print(f"epoch {epoch + 1}/{self.max_epochs}")
        print(f"Grid search ID:{self.grid_search_ID}, Fold number: {self.fold_num+1}")
        if self.l1_regulariztion_lambda>0:
            print("loss with l1_regulariztion")
        if self.l2_regulariztion_lambda>0:
            print("loss with l2_regulariztion")
        # Set the model to training mode
        self.model.train()
        epoch_loss = 0
        step = 0
        # Create an empty dictionary to store logs for this epoch
        self.logs[epoch]={}
        # Iterate over batches in training data loader
        for batch_data in self.train_loader:
            step += 1
            inputs, labels = batch_data[self.image_key].to(self.device), batch_data[self.class_col_name].to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Use mixed precision for faster training
            with torch.cuda.amp.autocast(enabled=self.enabled):
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels.to(torch.long))
                if (self.l1_regulariztion_lambda>0) or (self.l1_regulariztion_lambda>0):
                    loss = self.add_l1_l2_regularization(loss)
            # Scale the loss to bring gradients into the FP16 dynamic range
            self.scaler.scale(loss).backward()
            # Unscale gradients to FP32 for optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()
            if step % self.show_interval == 0:
                print(f"{step}/{self.train_num_batch}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        # Save average training loss in logs dictionary for this epoch
        self.logs[epoch]['train_loss'] = epoch_loss
        # Print average training loss for this epoch
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        return epoch_loss

    def validation(self, epoch):
        # --------------- EVALUATION --------------- #
        # Evaluate the model every self.val_interval epochs
        if (epoch + 1) % self.val_interval == 0:
            # Set the model to evaluation mode
            self.model.eval()
            # Disable gradient calculation for inference
            with torch.no_grad():
                val_epoch_loss = 0
                val_step = 0
                y_pred = torch.tensor([], dtype=torch.float32, device=self.device)
                y = torch.tensor([], dtype=torch.long, device=self.device)
                for val_data in self.val_loader:
                    val_step += 1
                    val_images, val_labels = (
                        val_data[self.image_key].to(self.device),
                        val_data[self.class_col_name].to(self.device),
                    )
                    # Use mixed precision for faster inference
                    with torch.cuda.amp.autocast():
                        val_outputs = self.model(val_images)
                    # Calculate the loss between predicted and actual labels
                    val_loss = self.loss_function(val_outputs, val_labels.to(torch.long))
                    if (self.l1_regulariztion_lambda>0) or (self.l1_regulariztion_lambda>0):
                        val_loss = self.add_l1_l2_regularization(val_loss)
                    val_epoch_loss += val_loss.item()
                    # Concatenate predicted and actual labels for metric calculation
                    y_pred = torch.cat([y_pred, val_outputs], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                # Calculate average validation loss per epoch
                val_epoch_loss /= val_step
                # Save validation loss in logs dictionary
                self.logs[epoch]['val_loss'] = val_epoch_loss
                # Print average validation loss per epoch and tag it as success in green color on yellow background
                print(f"epoch {epoch + 1} average val loss: {val_epoch_loss:.4f}",tag='success', tag_color='green', color='yellow')
                # Calculate metrics such as accuracy and F1 score on predicted and actual labels
                self.metric_calculation(epoch,y_pred,y)
                return val_epoch_loss
        
    def metric_calculation(self,epoch,y_pred,y):
        # --------------- METRIC CALCULATION --------------- #
        # Calculate confusion matrix and classification report
        # y_pred must be the raw output of the model
        # y must not be one hot-encoded
        metric_results = self.metric_calculator.aggregate(y_pred.detach().cpu(),y.detach().cpu())
        # Save metric results in logs dictionary
        self.logs[epoch]['metrics'] = metric_results
        # Extract AUC, F1 score and MCC from metric results
        auc = self.logs[epoch]['metrics']['AUC']['weighted-avg']
        f1_score = self.logs[epoch]['metrics']['f1 score']['weighted-avg']
        mcc = self.logs[epoch]['metrics']['matthews correlation coefficient']['weighted-avg']
        # Update best MCC and best MCC epoch if current MCC is better than previous best MCC
        if mcc > self.best_mcc:
            self.best_mcc = mcc
            self.best_mcc_epoch = epoch + 1
        # Print current epoch, F1 score, AUC, MCC, best MCC and best MCC epoch
        print(
            f"current epoch: {epoch + 1}"
            f" current F1 score: {f1_score:.4f}"
            f" current AUC: {auc:.4f}"
            f" current MCC: {mcc:.4f}"
            f" Best MCC: {self.best_mcc:.4f}"
            f" at epoch: {self.best_mcc_epoch}")

    def start(self, return_model=False,return_epoch=False):
        # ///////////////////////////////////////////////

        # ********** Training **********
        # Create an empty dictionary to store logs
        self.logs={}
        # Create directories for saving logs and models
        self.make_dirs()
        # Iterate over epochs
        for epoch in range(self.max_epochs):
            # Train the model for one epoch
            train_loss = self.train(epoch)
            # Evaluate the model on validation set and calculate validation loss
            val_loss = self.validation(epoch)
            if self.with_scheduler:
              self.scheduler.step(val_loss)
            # Check if early stopping is enabled and stop training if validation loss does not improve
            if self.early_stop_callback:
                self.early_stop_check(input_value=val_loss,
                loss=True, model=self.model,epoch=epoch)
            if self.early_stop:
                print("Early stopping")
                break
        # ********** Saving logs **********
        # Save logs dictionary as a CSV file
        print('Saving log ...')
        self.save_logs()
        if return_model:
          return self.model
        if return_epoch:
          return self.best_epoch

    
    def serialize_logs(self):
        # Create an empty dictionary to store the serialized logs
        serialized_logs={}
        # Add the grid search ID, fold number, epoch number, train loss, validation loss, and metrics to the dictionary
        serialized_logs['grid_search_ID']=[self.grid_search_ID]*len(self.logs.keys())
        serialized_logs['fold_num']=[self.fold_num]*len(self.logs.keys())
        serialized_logs['epoch_num']=list(self.logs.keys())
        serialized_logs['train_loss']=[]
        serialized_logs['val_loss']=[]

        # Iterate over the keys of the logs and add the train loss, validation loss, and metrics to the dictionary
        for epoch in list(self.logs.keys()):
            serialized_logs['train_loss'].append(self.logs[epoch]['train_loss'])
            serialized_logs['val_loss'].append(self.logs[epoch]['val_loss'])
            metrics = self.logs[epoch]['metrics']
            for metric in list(metrics.keys()):
                for item in list(metrics[metric].keys()):
                    name=f'{metric}_{item}'
                    if name in list(serialized_logs.keys()):
                        value=metrics[metric][item]
                        serialized_logs[name].append(value)
                    else:
                        serialized_logs[name]=[]
                        value=metrics[metric][item]
                        serialized_logs[name].append(value)
        # Return the serialized logs
        return serialized_logs

    def save_logs(self):
        # Serialize the logs
        serialized_logs = self.serialize_logs()
        # Create a Pandas DataFrame from the serialized logs
        df_logs = pd.DataFrame.from_dict(serialized_logs)
        # Save the DataFrame to a CSV file with the grid search ID as the filename
        log_dir = self.log_dir.joinpath(f"{self.grid_search_ID}.csv")
        if os.path.exists(log_dir):
            main_df=pd.read_csv(log_dir, index_col=0)
            main_df=pd.concat([main_df,df_logs])
            main_df.reset_index(drop=True).to_csv(log_dir)
        else:
            df_logs.to_csv(log_dir)
