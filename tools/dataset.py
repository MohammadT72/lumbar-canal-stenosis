# ------------------------------------------------------------------------------
# Dataset Preparation Class
# ------------------------------------------------------------------------------

from pathlib import Path
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class make_dataset:
    """
    A class to create and manage datasets for training machine learning models,
    including handling labels, creating dataframes, performing stratified group
    k-fold cross-validation splits, and plotting data distributions.

    Attributes:
        folder_path (str): Path to the dataset folder.
        folder_base (bool): If True, data is organized in subfolders per class.
        n_splits (int): Number of splits for k-fold cross-validation.
        col_name (str): Column name for grouping, e.g., patient IDs.
        k_fold (bool): Whether to perform k-fold cross-validation.
        list_classes_names (list or str): List of class names to include or 'all'.
        class_col_name (str): Column name for class labels.
        target_col_name (str): Column name for target labels after mapping.
        target_map (dict): Mapping from original classes to target classes.
    """

    def __init__(
        self,
        folder_path,
        folder_base=False,
        n_splits=10,
        col_name='int_ID',
        k_fold=True,
        list_classes_names='all',
        class_col_name=None,
        target_col_name=None,
        target_map=None
    ):
        """
        Initialize the make_dataset class with dataset parameters.

        Args:
            folder_path (str): Path to the dataset folder.
            folder_base (bool): If True, data is organized in subfolders per class.
            n_splits (int): Number of splits for k-fold cross-validation.
            col_name (str): Column name for grouping, e.g., patient IDs.
            k_fold (bool): Whether to perform k-fold cross-validation.
            list_classes_names (list or str): List of class names to include or 'all'.
            class_col_name (str): Column name for class labels.
            target_col_name (str): Column name for target labels after mapping.
            target_map (dict): Mapping from original classes to target classes.
        """
        # Initialize lists to store data
        self.list_images = []
        self.list_labels = []
        self.list_classes = []
        self.list_x = []
        self.list_y = []
        self.list_width = []
        self.list_height = []
        self.names = []

        # Set dataset parameters
        self.folder_path = folder_path
        self.list_classes_names = list_classes_names
        self.class_col_name = class_col_name
        self.target_col_name = target_col_name
        self.target_map = target_map
        self.k_fold = k_fold
        self.n_splits = n_splits
        self.col_name = col_name
        self.folder_base = folder_base

        # Initialize dataframes and labels
        self.labels = None
        self.k_fold_df = None
        self.main_df = None

    # --------------------------------------------------------------------------
    # Method to Extract Labels and Images
    # --------------------------------------------------------------------------

    def extract_labels(self):
        """
        Extract labels and image paths from the dataset.

        If folder_base is True, assumes data is organized in subfolders per class.
        Otherwise, expects images and labels in 'images' and 'labels' directories.
        """
        print('Extracting labels ...')
        if self.folder_base:
            # For folder-based datasets where each class has its own folder
            main_dir = Path(self.folder_path)
            # Get list of class names (subfolders in the main directory)
            classes_name = [
                file for file in os.listdir(self.folder_path)
                if not os.path.isfile(os.path.join(self.folder_path, file))
            ]
            print(classes_name)
            # Map class indices to class names
            self.labels = dict(zip(list(range(len(classes_name))), classes_name))
            # Loop through each class folder
            for idx, name in enumerate(classes_name):
                class_dir = main_dir.joinpath(name)
                list_images_names = os.listdir(class_dir)
                # Store full paths to images
                self.list_images.extend([str(class_dir.joinpath(x)) for x in list_images_names])
                # Store class indices
                self.list_classes.extend([idx] * len(list_images_names))
                # Extract names (assuming names are before an underscore)
                self.names.extend([x.split('_')[0] for x in list_images_names])
            # Filter classes if specified
            if self.list_classes_names != 'all':
                self.labels = self.select_items_in_dict(self.labels, self.list_classes_names)
        else:
            # For datasets with 'images' and 'labels' directories
            self.list_images = glob(str(Path(self.folder_path).joinpath('images', '*')))
            # Loop through each image
            for image_path in tqdm(self.list_images):
                # Construct corresponding label path
                label_path = image_path.split('.png')[0].replace('images', 'labels') + '.txt'
                self.list_labels.append(label_path)
                # Read label file and extract class and bounding box information
                with open(label_path, 'r') as text:
                    class_idx, x, y, width, height = text.readline().strip().split(' ')
                    self.list_classes.append(int(class_idx))
                    self.list_x.append(float(x))
                    self.list_y.append(float(y))
                    self.list_width.append(float(width))
                    self.list_height.append(float(height))

    # --------------------------------------------------------------------------
    # Method to Create DataFrame from Extracted Data
    # --------------------------------------------------------------------------

    def make_dataframe(self):
        """
        Create a Pandas DataFrame from the extracted labels and images.

        Adds grouping based on names or indices if folder_base is True.
        """
        if self.folder_base:
            # Create DataFrame with images and class indices
            self.main_df = pd.DataFrame.from_dict({
                'image': self.list_images,
                'class_idx': self.list_classes,
                'names': self.names,
                self.col_name: None
            })
            # Assign unique IDs based on 'names' for grouping
            for idx, name in enumerate(np.unique(self.main_df['names'])):
                indices = self.main_df[self.main_df['names'] == name].index
                self.main_df.loc[indices, self.col_name] = idx
        else:
            # Create DataFrame with images, labels, and bounding box info
            self.main_df = pd.DataFrame.from_dict({
                'image': self.list_images,
                'label': self.list_labels,
                'class_idx': self.list_classes,
                'x': self.list_x,
                'y': self.list_y,
                'width': self.list_width,
                'height': self.list_height
            })

    # --------------------------------------------------------------------------
    # Method to Load Additional Information
    # --------------------------------------------------------------------------

    def load_info(self):
        """
        Load additional information from 'info.csv' in the dataset folder.

        The 'info.csv' file is expected to contain supplementary data.
        """
        self.k_fold_df = pd.read_csv(
            Path(self.folder_path).joinpath('info.csv'),
            index_col=0
        )

    # --------------------------------------------------------------------------
    # Method to Perform Stratified Group K-Fold Split
    # --------------------------------------------------------------------------

    def StratifiedGroupKFold(self):
        """
        Perform a stratified group k-fold split on the dataset.

        Splits the data into training, validation, and testing sets while
        preserving the class distribution and ensuring that groups are not
        split across different sets.
        """
        data = self.main_df.reset_index(drop=True)
        # First split to separate out the test set
        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for fold, (train_val_idx, test_idx) in enumerate(
            cv.split(data, data['class_idx'], data[self.col_name])
        ):
            data['test'] = False
            data.loc[test_idx, 'test'] = True
            break  # Only need to perform this split once
        # Second split to create training and validation sets
        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=26)
        train_val = data[data['test'] == False].reset_index()
        for fold, (train_idx, val_idx) in enumerate(
            cv.split(train_val, train_val['class_idx'], train_val[self.col_name])
        ):
            data[f'fold_{fold + 1}'] = None
            old_index_train = train_val['index'].iloc[train_idx]
            old_index_val = train_val['index'].iloc[val_idx]
            data.loc[old_index_train, f'fold_{fold + 1}'] = 'train'
            data.loc[old_index_val, f'fold_{fold + 1}'] = 'val'
        self.main_df = data

    # --------------------------------------------------------------------------
    # Method to Check for Data Leakage
    # --------------------------------------------------------------------------

    def check_data_leakage(self):
        """
        Check for data leakage between training, validation, and test sets.

        Ensures that the same groups (e.g., patients) are not present in both
        training and validation or test sets across all folds.
        """
        k = self.n_splits
        for idx in range(k):
            # Check overlap between training and validation groups
            train_groups = set(self.main_df[self.main_df[f'fold_{idx + 1}'] == 'train'][self.col_name])
            val_groups = set(self.main_df[self.main_df[f'fold_{idx + 1}'] == 'val'][self.col_name])
            assert len(train_groups.intersection(val_groups)) == 0, \
                'Some groups are in both the train and validation sets!'
            # Check overlap between training and test groups
            test_groups = set(self.main_df[self.main_df['test'] == True][self.col_name])
            assert len(train_groups.intersection(test_groups)) == 0, \
                'Some groups are in both the train and test sets!'
            # Check overlap between validation and test groups
            assert len(val_groups.intersection(test_groups)) == 0, \
                'Some groups are in both the validation and test sets!'
        print('No data leakage detected.')

    # --------------------------------------------------------------------------
    # Method to Load Class Labels
    # --------------------------------------------------------------------------

    def load_labels(self):
        """
        Load class labels from 'classes.txt' in the dataset folder.

        The 'classes.txt' file should contain class labels, one per line.
        """
        self.labels = pd.read_csv(
            Path(self.folder_path).joinpath('classes.txt'),
            header=None, delimiter='\t'
        ).to_dict()[0]
        # Filter classes if specified
        if self.list_classes_names != 'all':
            self.labels = self.select_items_in_dict(self.labels, self.list_classes_names)

    # --------------------------------------------------------------------------
    # Method to Generate New Labels Based on a Mapping
    # --------------------------------------------------------------------------

    def generate_new_labels(self, mapping, class_col_name, target_name):
        """
        Generate new labels based on a provided mapping.

        Args:
            mapping (dict): Mapping from new class indices to original class indices.
            class_col_name (str): Column name of the original class labels.
            target_name (str): Column name for the new target labels.

        This method creates a new column in the main dataframe with the new labels.
        """
        y = self.main_df[class_col_name].tolist()
        new_y = []
        for class_idx in y:
            for key, value in mapping.items():
                if class_idx in value:
                    new_y.append(key)
        self.main_df[target_name] = new_y

    # --------------------------------------------------------------------------
    # Callable Method to Execute the Dataset Preparation
    # --------------------------------------------------------------------------

    def __call__(self):
        """
        Execute the dataset preparation steps.

        Returns:
            pd.DataFrame: The prepared main dataframe with all necessary information.
        """
        # Extract labels and images
        self.extract_labels()
        # Create dataframe from extracted data
        self.make_dataframe()
        # Load class labels if not folder-based
        if not self.folder_base:
            self.load_labels()
        # Generate new labels if mapping is provided
        if self.class_col_name and self.target_col_name and self.target_map:
            self.generate_new_labels(self.target_map, self.class_col_name, self.target_col_name)
        # Perform k-fold splitting if enabled
        if self.k_fold:
            if not self.folder_base:
                # Load additional info and merge with main dataframe
                self.load_info()
                self.main_df = self.main_df.sort_values(by=['image'])
                self.k_fold_df = self.k_fold_df.sort_values(by=['image'])
                self.main_df = pd.concat(
                    [self.k_fold_df.drop(['image', 'label'], axis=1), self.main_df],
                    axis=1
                )
            # Perform stratified group k-fold split
            self.StratifiedGroupKFold()
            # Check for data leakage
            self.check_data_leakage()
        return self.main_df

    # --------------------------------------------------------------------------
    # Method to Plot K-Fold Splits
    # --------------------------------------------------------------------------

    def plot_k_fold(self):
        """
        Plot the k-fold cross-validation splits.

        Visualizes how the data is split into training and validation sets across
        different folds, along with the class and group distributions.
        """
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm
        fig, ax = plt.subplots(figsize=(12, 6))
        data = self.main_df
        # Exclude test data for plotting
        X = data[data['test'] == False].reset_index(drop=True)
        y = X.class_idx.to_numpy().astype(np.int8)
        group = X[self.col_name].to_numpy().astype(np.int8)
        k = self.n_splits

        # Generate the training/testing visualizations for each CV split
        for idx in range(k):
            # Identify training and validation indices
            tr = X[X[f'fold_{idx + 1}'] == 'train'].index
            tt = X[X[f'fold_{idx + 1}'] == 'val'].index
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1  # Validation set
            indices[tr] = 0  # Training set

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [idx + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=20,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )

        # Plot the data classes and groups at the end
        ax.scatter(
            range(len(X)),
            [k + 0.5] * len(X),
            c=group,
            marker="_",
            lw=20,
            cmap=cmap_data
        )

        ax.scatter(
            range(len(X)),
            [k + 1.5] * len(X),
            c=y,
            marker="_",
            lw=20,
            cmap=cmap_data
        )

        # Formatting
        yticklabels = list(range(k)) + ["group", "class"]
        ax.set(
            yticks=np.arange(k + 2) + 0.5,
            yticklabels=yticklabels,
            xlabel="Sample index",
            ylabel="CV iteration",
            ylim=[k + 2.2, -0.2],
            xlim=[0, len(X)],
        )
        ax.set_title(f"{type(StratifiedGroupKFold).__name__}", fontsize=15)
        ax.legend(
            [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
            ["Validation set", "Training set"],
            loc=(1.02, 0.8),
        )
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(right=0.7)
        plt.show()

    # --------------------------------------------------------------------------
    # Method to Plot Class Distribution Across Folds
    # --------------------------------------------------------------------------

    def plot_class_dist(self):
        """
        Plot the class distribution for each fold.

        Visualizes the number of samples per class in the training, validation,
        and test sets for each fold.
        """
        k = self.n_splits
        fig, axes = plt.subplots(k, 1, figsize=(7, 35), constrained_layout=True)

        for idx, ax in enumerate(axes):
            width = 0.25
            multiplier = 0
            # Get class counts for each dataset split
            train_test = {
                'train': self.main_df[self.main_df[f'fold_{idx + 1}'] == 'train']['class_idx']
                          .value_counts().sort_index().tolist(),
                'validation': self.main_df[self.main_df[f'fold_{idx + 1}'] == 'val']['class_idx']
                              .value_counts().sort_index().tolist(),
                'test': self.main_df[self.main_df['test'] == True]['class_idx']
                        .value_counts().sort_index().tolist()
            }
            x = np.arange(len(self.labels))
            # Plot bars for each dataset split
            for dataset, values in train_test.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, values, width, label=dataset)
                ax.bar_label(rects, padding=3)
                multiplier += 1

            # Set labels and titles
            ax.set_ylabel('Number of Samples')
            ax.set_title(f'Class Distribution - Fold {idx + 1}')
            ax.set_xticks(x + width, self.labels.values())
            ax.legend(loc='upper left')

        plt.show()

    # --------------------------------------------------------------------------
    # Helper Method to Select Items in a Dictionary
    # --------------------------------------------------------------------------

    def select_items_in_dict(self, dictionary, list_classes_names):
        """
        Select items from a dictionary based on a list of keys.

        Args:
            dictionary (dict): The original dictionary.
            list_classes_names (list): List of keys to select.

        Returns:
            dict: A new dictionary containing only the selected items.
        """
        new_dict = {}
        for key, item in dictionary.items():
            if key in list_classes_names:
                new_dict[key] = item
        return new_dict
