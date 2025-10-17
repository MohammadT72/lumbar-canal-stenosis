# main.py

import os
import shutil
import zipfile
from glob import glob
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from ultralytics import YOLO, NAS
import argparse

def KFold(main_df, n_splits=10, col_name='names'):
    """
    Creates stratified group k-fold splits for the dataset.
    """
    data = main_df.reset_index(drop=True)
    # First split to separate out a test set
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_val_idx, test_idx) in enumerate(cv.split(data, data['class_idx'], data[col_name])):
        data['test'] = False
        data.loc[test_idx, 'test'] = True
        break  # Only take the first split to define test set
    # Further split the train_val data into train and validation sets
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=26)
    train_val = data[data['test'] == False].reset_index()
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_val, train_val['class_idx'], train_val[col_name])):
        data[f'fold_{fold+1}'] = None
        old_index_train = train_val['index'].iloc[train_idx]
        old_index_val = train_val['index'].iloc[val_idx]
        data.loc[old_index_train, f'fold_{fold+1}'] = 'train'
        data.loc[old_index_val, f'fold_{fold+1}'] = 'val'
    return data

def check_data_leakage(main_df, n_splits=5, col_name='names'):
    """
    Checks for data leakage between train, validation, and test sets.
    """
    for idx in range(n_splits):
        # Check if any name appears in both train and val sets
        assert len(set(main_df[main_df[f'fold_{idx+1}'] == 'train'][col_name]).intersection(
            set(main_df[main_df[f'fold_{idx+1}'] == 'val'][col_name]))) == 0, \
            'Some names are in both the train and val sets!'
        # Check if any name appears in both train and test sets
        assert len(set(main_df[main_df[f'fold_{idx+1}'] == 'train'][col_name]).intersection(
            set(main_df[main_df['test'] == True][col_name]))) == 0, \
            'Some names are in both the train and test sets!'
        # Check if any name appears in both val and test sets
        assert len(set(main_df[main_df[f'fold_{idx+1}'] == 'val'][col_name]).intersection(
            set(main_df[main_df['test'] == True][col_name]))) == 0, \
            'Some names are in both the val and test sets!'
    print('No data leakage')

class KFoldTrainer:
    def __init__(self, main_df, k, dataset_dir,
                 save_dir='/content/drive/MyDrive/results/lumbar_yolo',
                 save_name='runs_f', classes_dict={0: 'ROI'},
                 model_name='yolov8n.pt', task='detect',
                 epochs=110, image_size=640, batch_size=32,
                 lr_init=0.001, lr_final=1e-7, optimizer='Adam',
                 device=0, early_stop=0):
        """
        Initialize the KFoldTrainer with dataset and training parameters.
        """
        self.df = main_df
        self.k = k
        self.main_dir = dataset_dir
        self.model_name = model_name
        self.save_dir = save_dir
        self.save_name = save_name
        self.task = task
        self.config_dir = str(Path(self.main_dir).joinpath('config.yml'))
        self.epochs = epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.classes_dict = classes_dict
        self.optimizer = optimizer
        self.device = device
        self.early_stop = early_stop
        self.model = None

    def data_selection(self, fold_num):
        """
        Generate train.txt, val.txt, test.txt files based on the current fold.
        """
        # Prepare training data
        train = self.df[self.df[f'fold_{fold_num}'] == 'train']['image_path'].tolist()
        train_file = Path(self.main_dir).joinpath('train.txt')
        with open(train_file, 'w') as f:
            for item in train:
                f.write(item + '\n')
        # Prepare validation data
        val = self.df[self.df[f'fold_{fold_num}'] == 'val']['image_path'].tolist()
        val_file = Path(self.main_dir).joinpath('val.txt')
        with open(val_file, 'w') as f:
            for item in val:
                f.write(item + '\n')
        # Prepare test data
        test = self.df[self.df['test'] == True]['image_path'].tolist()
        test_file = Path(self.main_dir).joinpath('test.txt')
        with open(test_file, 'w') as f:
            for item in test:
                f.write(item + '\n')

    def final_data_selection(self):
        """
        Generate train.txt, val.txt, test.txt files for the final training.
        """
        # Prepare training data (all data except test set)
        train = self.df[self.df['test'] == False]['image_path'].tolist()
        train_file = Path(self.main_dir).joinpath('train.txt')
        with open(train_file, 'w') as f:
            for item in train:
                f.write(item + '\n')
        # Prepare validation and test data (both are test set in this case)
        val = self.df[self.df['test'] == True]['image_path'].tolist()
        val_file = Path(self.main_dir).joinpath('val.txt')
        with open(val_file, 'w') as f:
            for item in val:
                f.write(item + '\n')
        test = val  # Same as val
        test_file = Path(self.main_dir).joinpath('test.txt')
        with open(test_file, 'w') as f:
            for item in test:
                f.write(item + '\n')

    def make_config(self):
        """
        Create a configuration file for training.
        """
        data = dict(
            path=self.main_dir,
            train=str(Path(self.main_dir).joinpath('train.txt')),
            val=str(Path(self.main_dir).joinpath('val.txt')),
            test=str(Path(self.main_dir).joinpath('test.txt')),
            names=self.classes_dict
        )
        config_file = Path(self.main_dir).joinpath('config.yml')
        with open(config_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def train(self, name):
        """
        Train the model using the specified configuration.
        """
        if 'nas' in name.lower():
            self.model = NAS(model=self.model_name, task=self.task)
        else:
            self.model = YOLO(model=self.model_name, task=self.task)
        self.model.train(data=self.config_dir,
                         epochs=self.epochs,
                         imgsz=self.image_size,
                         batch=self.batch_size,
                         lr0=self.lr_init,
                         lrf=self.lr_final,
                         optimizer=self.optimizer,
                         device=self.device,
                         patience=self.early_stop,
                         workers=15,
                         cache=True,
                         name=name)

    def check_runs(self):
        """
        Unzip previous runs if exists.
        """
        target_zip = Path(self.save_dir).joinpath(f'{self.save_name}.zip')
        if os.path.exists(target_zip):
            with zipfile.ZipFile(target_zip, 'r') as zip_ref:
                zip_ref.extractall('/content')

    def make_archive(self, source, destination):
        """
        Archive the runs directory to save results.
        """
        base = os.path.basename(destination)
        name, format = os.path.splitext(base)
        format = format.lstrip('.')
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move(f'{name}.{format}', destination)

    def val(self):
        """
        Perform validation.
        """
        self.model.val()

    def reset_fold(self):
        """
        Reset files and model for the next fold.
        """
        # Remove train.txt, val.txt, test.txt files
        train_file = Path(self.main_dir).joinpath('train.txt')
        if os.path.exists(train_file):
            os.remove(train_file)
        val_file = Path(self.main_dir).joinpath('val.txt')
        if os.path.exists(val_file):
            os.remove(val_file)
        test_file = Path(self.main_dir).joinpath('test.txt')
        if os.path.exists(test_file):
            os.remove(test_file)
        self.model = None

    def __call__(self, start=1):
        """
        Run training for each fold.
        """
        for fold in range(start, self.k + 1):
            print(f'---------- fold {fold} ----------')
            self.data_selection(fold)
            self.make_config()
            self.check_runs()
            self.train(f'fold_{fold}')
            self.val()
            self.make_archive(os.path.join(os.getcwd(), 'runs'),
                              str(Path(self.save_dir).joinpath(f'{self.save_name}.zip')))
            self.reset_fold()

    def final_test(self):
        """
        Run final training on all data except test set.
        """
        print('---------- final training ----------')
        self.final_data_selection()
        self.make_config()
        self.train('final_training')
        self.val()
        self.make_archive(os.path.join(os.getcwd(), 'runs'),
                          str(Path(self.save_dir).joinpath(f'{self.save_name}.zip')))
        self.reset_fold()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Training with YOLO')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory where the dataset is located')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/results/lumbar_yolo',
                        help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='yolov8n.pt',
                        help='Pre-trained YOLO model')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=110,
                        help='Number of training epochs')
    parser.add_argument('--image_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr_init', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lr_final', type=float, default=1e-7,
                        help='Final learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use')
    parser.add_argument('--device', type=int, default=0,
                        help='Device ID (e.g., 0 for GPU 0)')
    parser.add_argument('--early_stop', type=int, default=0,
                        help='Early stopping patience')
    parser.add_argument('--classes_dict', type=str, default='',
                        help='Path to YAML or JSON file with class names')
    parser.add_argument('--start_fold', type=int, default=1,
                        help='Starting fold number')
    parser.add_argument('--final_training', action='store_true',
                        help='Run final training on all data except test set')
    args = parser.parse_args()

    # Load classes_dict from file if provided
    if args.classes_dict:
        # Assume it's a YAML or JSON file
        with open(args.classes_dict, 'r') as f:
            classes_dict = yaml.safe_load(f)
    else:
        classes_dict = {0: 'ROI'}

    # Define base directory for the dataset
    base_dir = args.dataset_dir

    # Get list of image files
    images = glob(str(Path(base_dir).joinpath('images', '*.png')))

    # Corresponding label files
    labels = [im.replace('images', 'labels').replace('.png', '.txt') for im in images]

    # Extract names from label paths
    names = [Path(lb).stem.split('_')[0] for lb in labels]

    # Extract class indices from label files
    classes = []
    for lb in labels:
        with open(lb, 'r') as f:
            content = f.read()
            if content:
                classes.append(int(content[0]))
            else:
                classes.append(0)  # Default class if label file is empty

    # Create a DataFrame to hold image paths, label paths, names, and class indices
    main_df = pd.DataFrame({
        'image_path': images,
        'label_path': labels,
        'names': names,
        'class_idx': classes
    })

    # Generate the new DataFrame with K-Fold splits
    new_df = KFold(main_df, n_splits=args.k_folds)

    # Check for data leakage in the new DataFrame
    check_data_leakage(new_df, n_splits=args.k_folds)

    # Initialize the KFoldTrainer with parsed arguments
    trainer = KFoldTrainer(
        main_df=new_df,
        k=args.k_folds,
        dataset_dir=base_dir,
        save_dir=args.save_dir,
        classes_dict=classes_dict,
        model_name=args.model_name,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        optimizer=args.optimizer,
        device=args.device,
        early_stop=args.early_stop
    )

    # Run k-fold cross-validation training
    trainer(start=args.start_fold)

    # Optionally, run final training on all data
    if args.final_training:
        trainer.final_test()

if __name__ == '__main__':
    main()
