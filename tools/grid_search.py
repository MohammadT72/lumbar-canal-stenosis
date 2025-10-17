# Standard library imports
import os
import itertools
from pathlib import Path

# Third-party imports
import yaml
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
import pynvml

# Local imports
from print_color import print  # Custom print function with color support
from utils import *           
from metrics import *          
from trainer import trainer    # Custom trainer class
from dataset import make_dataset  # Function to create the dataset


# ------------------------------------------------------------------------------
# Custom Dataset Class
# ------------------------------------------------------------------------------

class CustomDataset(Dataset):
    """
    A custom Dataset class for loading and transforming data.

    Attributes:
        data_list (list): List of data samples.
        transforms (callable): Transformations to apply to each sample.
    """
    def __init__(self, data, transform):
        """
        Initialize the dataset with data and transformations.

        Args:
            data (list): List of data samples.
            transform (callable): Transformations to apply to each sample.
        """
        self.data_list = data
        self.transforms = transform

    def __len__(self):
        """
        Return the total number of samples.

        Returns:
            int: Number of samples.
        """
        return len(self.data_list)

    def __getitem__(self, index):
        """
        Retrieve a sample and apply transformations.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Any: Transformed data sample.
        """
        # Make a copy of the data to avoid modifying the original
        data = self.transforms(self.data_list[index].copy())
        return data

# ------------------------------------------------------------------------------
# Grid Searcher Class
# ------------------------------------------------------------------------------

class grid_searcher:
    """
    A class for performing grid search on a PyTorch model.

    Args:
        folder_path (str): Path to the dataset folder. Default is './datasets/Final_dataset'.
        folder_base (bool): Indicates if the folder is a base folder. Default is False.
        max_epochs (int): Maximum number of epochs for training. Default is 50.
        num_class (int): Number of classes in the dataset. Default is 5.
        list_classes_names (list or str): List of class names to include or 'all'. Default is 'all'.
        col_name (str): Column name for the identifier. Default is 'int_ID'.
        class_col_name (str): Column name for the class labels. Default is 'class_idx'.
        image_key (str): Key for the image data. Default is 'image'.
        target_col_name (str, optional): Column name for target labels if different from class_col_name.
        target_map (dict, optional): Mapping for target labels.
        train_cache_rate (float): Fraction of training data to cache. Default is 0.5.
        val_cache_rate (float): Fraction of validation data to cache. Default is 0.5.
        k (int): Number of folds for k-fold cross-validation. Default is 10.
        with_scheduler (bool): Whether to use a learning rate scheduler. Default is True.
        val_interval (int): Interval for validation during training. Default is 1.
        save_path (str): Path to save results. Default is './results'.
        metrics (list or str): Metrics to evaluate. Default is 'all'.
        save_models (bool): Whether to save the trained models. Default is False.
        weighted_loss (bool): Whether to use weighted loss. Default is False.
        gpu (bool): Whether to use GPU acceleration. Default is True.
        amp (bool): Whether to use Automatic Mixed Precision (AMP). Default is True.
        num_workers (int): Number of workers for data loading. Default is 0.
        check (bool): Flag for checking data consistency. Default is True.
        transfer_learning (bool): Whether to use transfer learning. Default is True.
        over_sampling (bool): Whether to apply over-sampling to balance classes. Default is False.
        early_stop_callback (bool): Whether to use early stopping. Default is True.
        early_stop_patience (int): Patience epochs for early stopping. Default is 20.
        early_stop_delta (float): Minimum change to qualify as an improvement. Default is 0.
        early_stop_verbose (bool): Verbosity for early stopping. Default is True.
        data_fraction (float): Fraction of data to use. Default is 1.0.
    """

    def __init__(
        self,
        folder_path='./datasets/Final_dataset',
        folder_base=False,
        max_epochs=50,
        num_class=5,
        list_classes_names='all',
        col_name='int_ID',
        class_col_name='class_idx',
        image_key='image',
        target_col_name=None,
        target_map=None,
        train_cache_rate=0.5,
        val_cache_rate=0.5,
        k=10,
        with_scheduler=True,
        val_interval=1,
        save_path='./results',
        metrics='all',
        save_models=False,
        weighted_loss=False,
        gpu=True,
        amp=True,
        num_workers=0,
        check=True,
        transfer_learning=True,
        over_sampling=False,
        early_stop_callback=True,
        early_stop_patience=20,
        early_stop_delta=0,
        early_stop_verbose=True,
        data_fraction=1.0
    ):
        """
        Initialize a new instance of the grid_searcher class.
        """
        # Set up paths and configurations
        self.config_path = Path(folder_path).joinpath('grid_search_config.yaml')
        self.folder_path = folder_path
        self.save_path = save_path if save_path is not None else folder_path
        self.k = k  # Number of folds for cross-validation
        self.save_models = save_models

        # Create dataset loader
        self.dataset_loader = make_dataset(
            folder_path=self.folder_path,
            folder_base=folder_base,
            n_splits=k,
            class_col_name=class_col_name,
            target_col_name=target_col_name,
            target_map=target_map,
            col_name=col_name,
            k_fold=True,
            list_classes_names=list_classes_names
        )

        # Data fraction to use
        self.frac = data_fraction
        self.class_col_name = class_col_name
        if target_col_name is not None and target_map is not None:
            self.class_col_name = target_col_name

        # Data keys and caching rates
        self.image_key = image_key
        self.train_cache_rate = train_cache_rate
        self.val_cache_rate = val_cache_rate

        # Training configurations
        self.trainer = trainer  # Custom trainer
        self.k_fold_ploter = k_fold_plotter(self.save_path)  # Assumed to be defined elsewhere
        self.metrics = metrics
        self.val_interval = val_interval
        self.max_epochs = max_epochs
        self.with_scheduler = with_scheduler
        self.num_workers = num_workers
        self.check = check
        self.transfer_learning = transfer_learning
        self.over_sampling = over_sampling
        self.weighted_loss = weighted_loss
        self.list_classes_names = list_classes_names

        # Set number of classes
        self.num_class = num_class
        if isinstance(list_classes_names, list):
            self.num_class = len(self.list_classes_names)
        if target_col_name is not None and target_map is not None:
            self.num_class = len(target_map)

        # Early stopping configurations
        self.early_stop_callback = early_stop_callback
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.early_stop_verbose = early_stop_verbose

        # AMP and GPU configurations
        self.amp = amp
        self.gpu = gpu
        self.device = 'cpu'
        if gpu:
            # Check if GPU is available and use it if it is
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up directories
        self.grid_search_dir = Path(self.save_path).joinpath('runs')
        self.grid_search_dir.mkdir(parents=True, exist_ok=True)

        # Initialize variables
        self.data = None
        self.labels_dict = None
        self.config = None
        self.freeze_base_first = False
        self.train_data_list = None
        self.val_data_list = None
        self.test_data_list = None
        self.train_loader = None
        self.val_loader = None
        self.val_transforms = None
        self.train_transforms = None
        self.loss_name = None
        self.loss_function = None
        self.batch_size = 0
        self.regularization_l1 = 0.0
        self.regularization_l2 = 0.0
        self.learning_rate = None
        self.input_size = None
        self.optimizer = None
        self.optimizer_name = None
        self.metric_calculator = None
        self.grid_search_id = None
        self.grid_search_df = None
        self.grid_search_list = None
        self.classifier_key = None
        self.classifier_in_features = None
        self.model = None
        self.start_fold = None

    # --------------------------------------------------------------------------
    # Method to Read Configuration
    # --------------------------------------------------------------------------

    def read_config(self):
        """
        Read the grid search configuration from a YAML file.
        """
        # Check if config file exists
        if not os.path.exists(self.config_path):
            raise ValueError("Config file does not exist: {}".format(self.config_path))

        # Load YAML configuration
        with open(self.config_path, 'r') as stream:
            self.config = yaml.safe_load(stream)

    # --------------------------------------------------------------------------
    # Method to Select Data Lists for Training, Validation, and Testing
    # --------------------------------------------------------------------------

    def select_data_list(self, fold, frac=0.1):
        """
        Selects a subset of the data for training, validation, and testing.

        Args:
            fold (int): The fold number to use for training and validation.
            frac (float): The fraction of the data to use (e.g., 0.1 for 10%).
        """
        # Ensure that data is loaded
        if self.data is None:
            raise ValueError("Data is not loaded. Please load data before selecting data lists.")

        # Filter data based on class names if specified
        if self.list_classes_names != 'all':
            self.data = self.data[self.data[self.class_col_name].isin(self.list_classes_names)]
            self.data = self.data.reset_index(drop=True)

        # Apply over-sampling if enabled
        if self.over_sampling:
            self.data = self.data.reset_index()
            X = self.data[['index', 'int_ID']]
            y = self.data[self.class_col_name]

            # Perform random over-sampling to balance classes
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            self.data = self.data.drop(['index'], axis=1).loc[X_resampled['index'].tolist()]
            self.data = self.data.reset_index(drop=True)

        # Select training data
        train_data_list = self.data[
            self.data[f'fold_{fold+1}'] == 'train'
        ].drop(['age', 'gender'], axis=1, errors='ignore').to_dict('records')
        self.train_data_list = train_data_list[:int(len(train_data_list) * frac)]

        # Select validation data
        val_data_list = self.data[
            self.data[f'fold_{fold+1}'] == 'val'
        ].drop(['age', 'gender'], axis=1, errors='ignore').to_dict('records')
        self.val_data_list = val_data_list[:int(len(val_data_list) * frac)]

        # Select test data
        test_data_list = self.data[
            self.data['test'] == True
        ].drop(
            ['age', 'gender'] + [f'fold_{x+1}' for x in range(self.k)],
            axis=1,
            errors='ignore'
        ).to_dict('records')
        self.test_data_list = test_data_list[:int(len(test_data_list) * frac)]

        # Print the number of records in each dataset
        print(f'Unique labels: {self.data[self.class_col_name].unique()}')
        print(f'Number of records in the train data list: {len(self.train_data_list)}')
        print(f'Number of records in the validation data list: {len(self.val_data_list)}')
        print(f'Number of records in the test data list: {len(self.test_data_list)}')

    # --------------------------------------------------------------------------
    # Method to Detect Starting Fold Number
    # --------------------------------------------------------------------------

    def detect_fold_number(self):
        """
        Detect the starting fold number for resuming training if applicable.
        """
        # Paths to log files
        log_csv = str(Path(self.save_path).joinpath('runs', 'logs', f'{self.grid_search_id}.csv'))
        log_csv_freeze = str(Path(self.save_path).joinpath('runs', 'logs', f'{self.grid_search_id}_freeze.csv'))

        # Check if log file exists and set starting fold number
        if os.path.exists(log_csv):
            df = pd.read_csv(log_csv, index_col=0)
            fold = int(df['fold_num'].max())
            self.start_fold = fold if fold > 0 else 0
        elif os.path.exists(log_csv_freeze):
            df = pd.read_csv(log_csv_freeze, index_col=0)
            fold = int(df['fold_num'].max())
            self.start_fold = fold if fold > 0 else 0
        else:
            self.start_fold = 0

    # --------------------------------------------------------------------------
    # Method to Update Grid Search Results
    # --------------------------------------------------------------------------

    def update_grid_search(self):
        """
        Updates the grid search results with the current grid search ID,
        marking the grid search as completed.
        """
        # Load the grid search results
        grid_search_csv_dir = self.grid_search_dir.joinpath('grid_search.csv')
        grid_search_df = pd.read_csv(grid_search_csv_dir, index_col=0)

        # Find the index of the current grid search ID
        index = grid_search_df[
            grid_search_df['grid_search_id'] == self.grid_search_id
        ].index[0]

        # Update the 'done' column to True
        grid_search_df.loc[index, 'done'] = True

        # Save the updated grid search results
        grid_search_df.to_csv(self.grid_search_dir.joinpath('grid_search.csv'))

    # --------------------------------------------------------------------------
    # Method to Generate Grid Search Combinations
    # --------------------------------------------------------------------------

    def generate_grid_search_combinations(self):
        """
        Generate all possible combinations of hyperparameters for grid search
        and save them to a CSV file.

        This method reads the hyperparameter configurations from `self.config`,
        creates all possible combinations using `itertools.product`, assigns
        unique IDs to each combination, and saves the combinations to a CSV
        file for tracking.

        Attributes Updated:
            self.grid_search_df (pd.DataFrame): DataFrame containing all grid
                search combinations with unique IDs and a 'done' flag.
        """
        # List to store hyperparameter values
        list_items = []
        # Iterate over each hyperparameter in the config
        for key in list(self.config.keys()):
            if isinstance(self.config[key], dict):
                # If the value is a dictionary, get its keys (possible options)
                list_items.append(list(self.config[key].keys()))
            elif isinstance(self.config[key], list):
                # If the value is a list, use it directly
                list_items.append(self.config[key])
        # Generate all combinations using itertools.product
        combinations = list(itertools.product(*list_items))
        # Create a DataFrame from the combinations
        df_combinations = pd.DataFrame(combinations, columns=self.config.keys())
        # Assign unique IDs to each grid search configuration
        df_combinations['grid_search_id'] = [f'LS-{i}' for i in range(len(df_combinations))]
        # Initialize the 'done' column to track completed searches
        df_combinations['done'] = [False] * len(df_combinations)
        # Save the DataFrame to an attribute
        self.grid_search_df = df_combinations
        # Save the combinations to a CSV file
        df_combinations.to_csv(self.grid_search_dir.joinpath('grid_search.csv'))

    # --------------------------------------------------------------------------
    # Method to Select Grid Search Configurations
    # --------------------------------------------------------------------------

    def select_search_list(self):
        """
        Select the grid search IDs that need to be processed.

        This method checks the existing grid search CSV file for configurations
        that haven't been processed yet (where 'done' is False). If all
        configurations are done, it sets `self.grid_search_list` to 'done'.
        If the CSV file doesn't exist, it generates a new set of combinations.

        Attributes Updated:
            self.grid_search_df (pd.DataFrame): Updated DataFrame from CSV.
            self.grid_search_list (list or str): List of grid search IDs to process
                or 'done' if all are completed.
        """
        grid_search_csv_dir = self.grid_search_dir.joinpath('grid_search.csv')

        # Check if the grid search results file exists
        if os.path.exists(grid_search_csv_dir):
            # Load the grid search results
            grid_search_df = pd.read_csv(grid_search_csv_dir, index_col=0)
            self.grid_search_df = grid_search_df

            # Select the grid search IDs that haven't been completed yet
            selected = grid_search_df[grid_search_df['done'] == False]
            if len(selected) > 0:
                self.grid_search_list = selected['grid_search_id'].tolist()
            else:
                # If all configurations are done
                self.grid_search_list = 'done'
        else:
            # Generate a new set of grid search combinations if the file doesn't exist
            self.generate_grid_search_combinations()
            # Select all grid search IDs to run
            self.grid_search_list = self.grid_search_df['grid_search_id'].tolist()

    # --------------------------------------------------------------------------
    # Method to Prepare Settings for a Specific Grid Search Configuration
    # --------------------------------------------------------------------------

    def prepare_settings(self):
        """
        Prepare the settings and initialize components for the current grid search ID.

        This method extracts the hyperparameters for the current `self.grid_search_id`
        from `self.grid_search_df`, initializes the model, loss function, optimizer,
        data transformations, and metric calculator based on these settings.

        Attributes Updated:
            self.model: Initialized model with replaced classifier.
            self.loss_function: Loss function instance.
            self.optimizer: Optimizer instance.
            self.metric_calculator: Metric calculator instance.
            self.batch_size (int): Batch size for training.
            self.learning_rate (float): Learning rate for optimizer.
            self.input_size (tuple): Input size expected by the model.
            self.regularization_l1 (float): L1 regularization coefficient.
            self.regularization_l2 (float): L2 regularization coefficient.
            self.freeze_base_first (bool): Whether to freeze base model layers initially.
        """
        # Get the settings for the current grid search ID
        settings = self.grid_search_df[self.grid_search_df['grid_search_id'] == self.grid_search_id]

        # Extract hyperparameters from settings
        model_name = settings['models'].iloc[0]
        classifier_size = settings['classifier_size'].iloc[0]
        activation_type = settings['activation_type'].iloc[0]
        augmentation = settings['augmentations'].iloc[0]
        self.regularization_l1 = settings['regularization_l1'].iloc[0]
        self.regularization_l2 = settings['regularization_l2'].iloc[0]
        self.freeze_base_first = settings['freeze_base_first'].iloc[0]
        hidden_params = settings['hidden_params'].iloc[0]
        drop_out = settings['drop_out'].iloc[0]
        self.input_size = tuple(self.config['models'][model_name]['input_size'])
        self.loss_name = self.config['losses'][settings['losses'].iloc[0]]['name']
        self.optimizer_name = self.config['optimizer'][settings['optimizer'].iloc[0]]['name']
        self.batch_size = settings['batch_size'].iloc[0]
        self.learning_rate = settings['learning_rate'].iloc[0]
        weights_name = self.config['models'][model_name]['weights']

        # Print the settings
        print(f'***** Search ID: {self.grid_search_id} *****')
        print(f'''Settings:
        --augmentations: {augmentation}
        --model: {model_name}
        --classifier_size: {classifier_size}
        --activation_type: {activation_type}
        --hidden_params: {hidden_params}
        --drop_out: {drop_out}
        --batch size: {self.batch_size}
        --loss_name: {self.loss_name}
        --optimizer: {self.optimizer_name}
        --learning rate: {self.learning_rate}
        --weights: {weights_name}''')

        # Get classifier configuration
        self.classifier_key = self.config['models'][model_name]['classifier']['key']
        self.classifier_in_features = self.config['models'][model_name]['classifier']['in_features']

        # Initialize the model
        self.model = self.model_selection(model_name=model_name, weights_name=weights_name)

        # Replace the classifier layer with custom settings
        self.replace_classifier(model_name, classifier_size, activation_type, hidden_params, drop_out)
        self.model = self.model.to(self.device)

        # Initialize the loss function
        self.loss_function = self.get_loss_function(self.loss_name)()
        if self.weighted_loss:
            weights = self.calculate_class_weights()
            self.loss_function = self.get_loss_function(self.loss_name)(class_weights=weights)

        # Initialize the optimizer
        self.optimizer = self.get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate)

        # Prepare data transformations
        self.make_transforms(model_name, augmentation)

        # Initialize metric calculator
        self.metric_calculator = self.make_metric_calculator()

    # --------------------------------------------------------------------------
    # Method to Reset Settings for a Fold
    # --------------------------------------------------------------------------

    def reset_fold(self):
        """
        Reset the training settings for a new fold during cross-validation.

        This method re-initializes the model, loss function, optimizer, and data
        transformations for the current grid search ID and resets data loaders
        and cached GPU memory.

        Attributes Reset:
            self.model: Re-initialized model.
            self.loss_function: Re-initialized loss function.
            self.optimizer: Re-initialized optimizer.
            self.metric_calculator: Re-initialized metric calculator.
            Data loaders and lists are reset to None.
        """
        # Re-prepare settings (similar to prepare_settings)
        settings = self.grid_search_df[self.grid_search_df['grid_search_id'] == self.grid_search_id]
        model_name = settings['models'].iloc[0]
        classifier_size = settings['classifier_size'].iloc[0]
        activation_type = settings['activation_type'].iloc[0]
        augmentation = settings['augmentations'].iloc[0]
        self.regularization_l1 = settings['regularization_l1'].iloc[0]
        self.regularization_l2 = settings['regularization_l2'].iloc[0]
        self.freeze_base_first = settings['freeze_base_first'].iloc[0]
        hidden_params = settings['hidden_params'].iloc[0]
        drop_out = settings['drop_out'].iloc[0]
        self.input_size = tuple(self.config['models'][model_name]['input_size'])
        self.loss_name = self.config['losses'][settings['losses'].iloc[0]]['name']
        self.optimizer_name = self.config['optimizer'][settings['optimizer'].iloc[0]]['name']
        self.batch_size = settings['batch_size'].iloc[0]
        self.learning_rate = settings['learning_rate'].iloc[0]
        weights_name = self.config['models'][model_name]['weights']
        self.classifier_key = self.config['models'][model_name]['classifier']['key']
        self.classifier_in_features = self.config['models'][model_name]['classifier']['in_features']

        # Re-initialize the model and other components
        self.model = self.model_selection(model_name=model_name, weights_name=weights_name)
        self.replace_classifier(model_name, classifier_size, activation_type, hidden_params, drop_out)
        self.model = self.model.to(self.device)
        self.loss_function = self.get_loss_function(self.loss_name)()
        if self.weighted_loss:
            weights = self.calculate_class_weights()
            self.loss_function = self.get_loss_function(self.loss_name)(class_weights=weights)
        self.optimizer = self.get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate)
        self.make_transforms(model_name, augmentation)
        self.metric_calculator = self.make_metric_calculator()

        # Reset data lists and data loaders
        self.reset_loaders()

        # Free up GPU memory
        self.free_memory()

    # --------------------------------------------------------------------------
    # Method to Reset Data Loaders
    # --------------------------------------------------------------------------

    def reset_loaders(self):
        """
        Reset the data lists and data loaders to None.

        This is useful when starting a new fold or grid search to ensure that
        data from previous runs doesn't interfere with the new run.

        Attributes Reset:
            self.train_data_list, self.val_data_list, self.test_data_list
            self.train_loader, self.val_loader
        """
        self.train_data_list = None
        self.val_data_list = None
        self.test_data_list = None
        self.train_loader = None
        self.val_loader = None

    # --------------------------------------------------------------------------
    # Method to Reset Grid Search Variables
    # --------------------------------------------------------------------------

    def reset_grid_search(self):
        """
        Reset grid search variables to prepare for a new grid search run.

        This method resets all the variables and components that are specific
        to a grid search configuration.

        Attributes Reset:
            All attributes related to data, model, loss function, optimizer,
            metric calculator, and grid search ID.
        """
        self.train_data_list = None
        self.val_data_list = None
        self.test_data_list = None
        self.train_loader = None
        self.val_loader = None
        self.val_transforms = None
        self.train_transforms = None
        self.loss_function = None
        self.batch_size = None
        self.optimizer = None
        self.metric_calculator = None
        self.grid_search_id = None
        self.classifier_key = None
        self.classifier_in_features = None
        self.model = None

    # --------------------------------------------------------------------------
    # Method to Free GPU Memory
    # --------------------------------------------------------------------------

    def free_memory(self):
        """
        Free up the GPU memory by clearing the cache.

        This is useful after model training or evaluation to ensure that
        memory is available for subsequent operations.
        """
        if self.gpu:
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # Method to Check Data Consistency and Preprocessing
    # --------------------------------------------------------------------------

    def check_data(self):
        """
        Check the input data, transformations, and labels for datasets.

        This method prints out the preprocessing transformations, input sizes,
        value ranges, and sample labels for both training and validation data.
        It is useful for verifying that data loading and preprocessing are
        working as intended.
        """
        if self.check:
            print('Checking the input ...')
            # Check training data
            print('--- train data')
            print('------ preprocessing')
            print(self.train_transforms)
            sample = next(iter(self.train_loader))
            input_size = sample[self.image_key].shape
            minv = sample[self.image_key].min()
            maxv = sample[self.image_key].max()
            labels = sample[self.class_col_name]
            print(f'input size: {input_size}')
            print(f'input minv: {minv}, input maxv: {maxv}')
            print(f'Sample of labels: {labels[:5]}')
            # Check validation data
            print('--- validation data')
            print('------ preprocessing')
            print(self.val_transforms)
            sample = next(iter(self.val_loader))
            input_size = sample[self.image_key].shape
            minv = sample[self.image_key].min()
            maxv = sample[self.image_key].max()
            labels = sample[self.class_col_name]
            print(f'input size: {input_size}')
            print(f'input minv: {minv}, input maxv: {maxv}')
            print(f'Sample of labels: {labels[:5]}')

    # --------------------------------------------------------------------------
    # Method to Check Model Parameters
    # --------------------------------------------------------------------------

    def check_model(self):
        """
        Check the model parameters, devices, and gradients.

        This method prints out counts and lists of parameters that require
        gradients and the devices they are located on. It helps to ensure that
        all parameters are correctly set up for training.
        """
        names = []
        devices_list = []
        params_grad = []
        for name, param in self.model.named_parameters():
            names.append(name)
            devices_list.append(param.device)
            params_grad.append(param.requires_grad)
        params_df = pd.DataFrame.from_dict({
            'Names': names,
            'Requires_grad': params_grad,
            'devices': devices_list
        })
        print(f'Number of devices: {params_df.devices.value_counts()}')
        print(f'Number of Requires_grad:\n{params_df.Requires_grad.value_counts()}')
        print(f'List of parameters requiring gradients:\n{params_df[params_df.Requires_grad == True]}')

    # --------------------------------------------------------------------------
    # Method to Estimate VRAM Usage
    # --------------------------------------------------------------------------

    def estimate_vram_usage(self):
        """
        Estimate the VRAM usage of the model with the current settings.

        This method moves the model and a sample input to the GPU, performs a
        forward and backward pass, and calculates the VRAM usage. It returns
        True if the total VRAM used exceeds the total available VRAM.

        Returns:
            bool: True if VRAM usage exceeds total VRAM, False otherwise.
        """
        if self.gpu:
            # Initialize NVML to get GPU properties
            index = torch.cuda.current_device()
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            props = torch.cuda.get_device_properties(self.device)
            total_memory = props.total_memory / 1024 ** 3  # Convert bytes to GB
            print("Estimating VRAM usage ...")
            print(f'Total VRAM: {total_memory:.2f} GB')

            # Initial memory usage
            init_used = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            init_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            init_used_nvml = init_info.used / 1024 ** 3

            # Move model to device
            self.model.to(self.device)
            # Create a random input sample with the specified batch size and input size
            input_sample = torch.randn((self.batch_size, 3) + self.input_size)
            input_sample = input_sample.to(self.device)

            # Forward pass
            output = self.model(input_sample)
            # Compute loss (mean of outputs for testing)
            loss = output.mean()
            # Backward pass
            loss.backward()

            # Final memory usage
            final_used = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            final_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            final_used_nvml = final_info.used / 1024 ** 3

            # Total used VRAM
            total_used = final_used + final_used_nvml - init_used - init_used_nvml
            self.free_memory()
            print(f"Total used VRAM: {total_used:.2f} GB")
            # Return True if used VRAM exceeds total available VRAM
            return total_used > total_memory

    # --------------------------------------------------------------------------
    # Method to Reduce Batch Size
    # --------------------------------------------------------------------------

    def reduce_batch_size(self):
        """
        Reduce the batch size by half.

        This method is useful when VRAM usage is too high, and we need to
        reduce the batch size to fit the model and data into memory.
        """
        new_batch_size = int(self.batch_size / 2)
        print(f'Reducing batch size from {self.batch_size} to {new_batch_size}')
        self.batch_size = new_batch_size

    # --------------------------------------------------------------------------
    # Method to Prepare Settings for Freezing Model
    # --------------------------------------------------------------------------

    def prepare_freeze_settings(self):
        """
        Prepare settings for training with the base model frozen.

        This method adjusts the batch size and learning rate for initial
        training with the base layers frozen to speed up training and allow
        for larger batch sizes.

        Attributes Updated:
            self.batch_size (int): Set to 512 for frozen training.
            self.early_stop_patience (int): Set to 5 epochs.
            self.learning_rate (float): Increased by a factor of 100.
            self.optimizer: Re-initialized with the new learning rate.
        """
        self.batch_size = 512  # Set a large batch size since the model is frozen
        self.early_stop_patience = 5  # Set early stopping patience
        print(f"Freeze settings - batch size: {self.batch_size}")
        # Increase learning rate for faster convergence on frozen layers
        self.learning_rate = self.learning_rate * 100
        print(f"Freeze settings - learning rate: {self.learning_rate}")
        # Re-initialize optimizer with new learning rate
        self.optimizer = self.get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate)

    # --------------------------------------------------------------------------
    # Method to Prepare Settings for Unfreezing Model
    # --------------------------------------------------------------------------

    def prepare_unfreeze_settings(self):
        """
        Prepare settings for training after unfreezing the model.

        This method restores the original batch size and learning rate, and
        unfreezes all layers of the model for fine-tuning.

        Attributes Updated:
            self.batch_size (int): Restored to original value.
            self.learning_rate (float): Restored to original value.
            self.loss_function: Re-initialized if weighted loss is used.
            self.optimizer: Re-initialized with the original learning rate.
        """
        # Get original settings
        settings = self.grid_search_df[self.grid_search_df['grid_search_id'] == self.grid_search_id]
        self.early_stop_patience = 20  # Reset early stopping patience
        self.batch_size = settings['batch_size'].iloc[0]
        print(f"Unfreeze settings - batch size: {self.batch_size}")
        self.learning_rate = settings['learning_rate'].iloc[0]
        print(f"Unfreeze settings - learning rate: {self.learning_rate}")
        # Re-initialize loss function
        self.loss_function = self.get_loss_function(self.loss_name)()
        if self.weighted_loss:
            weights = self.calculate_class_weights()
            self.loss_function = self.get_loss_function(self.loss_name)(class_weights=weights)
        # Re-initialize optimizer with original learning rate
        self.optimizer = self.get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.learning_rate)
        # Unfreeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = True

    # --------------------------------------------------------------------------
    # Method to test training
    # --------------------------------------------------------------------------
    def test_training(self, grid_search_id=None):
        """
        Perform a test training run using a random grid search configuration and a random fold.

        Args:
            grid_search_id (str, optional): Specific grid search ID to use. If None, a random one is selected.

        This method performs a single training run to test the training pipeline.
        It selects a random grid search configuration and a random fold, sets up the data loaders,
        checks the data and model, estimates VRAM usage, and then starts the training using the trainer class.
        After training, it attempts to plot the results.
        """
        # Read config file if not already loaded
        if self.config is None:
            self.read_config()

        # Set the grid search CSV directory
        grid_search_csv_dir = self.grid_search_dir.joinpath('grid_search.csv')

        # Check if the grid search CSV exists, if not generate it
        if not os.path.exists(grid_search_csv_dir):
            self.generate_grid_search_combinations()

        # Select the search list (grid search IDs to process)
        self.select_search_list()

        # Load data if not already loaded
        if self.data is None:
            self.data = self.dataset_loader()
            self.labels_dict = self.dataset_loader.labels

        # Select a random grid search ID if none provided
        if grid_search_id is None:
            idx = int(np.random.randint(0, len(self.grid_search_list), 1)[0])
            self.grid_search_id = self.grid_search_list[idx]
        else:
            self.grid_search_id = grid_search_id

        # Reset any previous fold settings
        self.reset_fold()

        # Select a random fold
        fold = int(np.random.randint(0, self.k, 1)[0])

        # Select data for the fold
        self.select_data_list(fold, frac=self.frac)

        # Create data loaders
        self.make_dataloader(self.batch_size)

        # Check data and model
        self.check_data()
        self.check_model()

        # Estimate VRAM usage
        vram_overloaded = self.estimate_vram_usage()

        # Initialize the trainer
        trainer = self.trainer(
            train_data_loader=self.train_loader,
            valid_data_loader=self.val_loader,
            save_models=self.save_models,
            model=self.model,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            max_epochs=self.max_epochs,
            with_scheduler=self.with_scheduler,
            metric_calculator=self.metric_calculator,
            class_col_name=self.class_col_name,
            image_key=self.image_key,
            amp=self.amp,
            val_interval=self.val_interval,
            device=self.device,
            show_interval=10,
            early_stop_callback=self.early_stop_callback,
            patience=self.early_stop_patience,
            early_stop_verbose=self.early_stop_verbose,
            early_stop_delta=self.early_stop_delta,
            save_path=self.save_path,
            class_num=self.num_class,
            grid_search_ID=self.grid_search_id + '_test',
            fold_num=fold
        )

        # Start training and get the best epoch
        best_epoch = trainer.start(return_epoch=True)

        # Attempt to plot results
        try:
            self.k_fold_ploter.load_logs(self.grid_search_id + '_test')
            self.k_fold_ploter.plot_loss(self.grid_search_id + '_test', best_epoch)
            self.k_fold_ploter.plot_metrics(self.grid_search_id + '_test')
            self.k_fold_ploter.plot_class_metrics(
                self.labels_dict,
                str(self.grid_search_id) + '_test',
                num_classes=self.num_class
            )
        except Exception as e:
            # Ignore plotting errors
            pass

    # --------------------------------------------------------------------------
    # Method to Perform K-Fold Cross-Validation
    # --------------------------------------------------------------------------

    def k_fold(self, start_fold=None):
        """
        Perform k-fold cross-validation training.

        Args:
            start_fold (int, optional): Fold number to start from. If None, starts from self.start_fold.

        This method iterates over the number of folds specified by self.k,
        and performs training on each fold. It handles freezing and unfreezing of
        the base model if specified, and updates the grid search results after completion.
        """
        # Set starting fold if provided
        if start_fold is not None:
            self.start_fold = start_fold

        # Loop over each fold
        for fold in range(self.start_fold, self.k):
            # Print current fold
            print(f'********* fold {fold + 1} *********')

            # Reset previous fold settings
            self.reset_fold()

            # Select data for current fold
            self.select_data_list(fold, frac=self.frac)

            # Handle freezing of base model if specified
            if self.freeze_base_first:
                # Prepare settings for freezing
                self.prepare_freeze_settings()

                # Create data loaders
                self.make_dataloader(self.batch_size)

                # Check data and model
                self.check_data()
                self.check_model()

                # Estimate VRAM usage
                vram_overloaded = self.estimate_vram_usage()

                # Initialize trainer for frozen model
                trainer = self.trainer(
                    train_data_loader=self.train_loader,
                    valid_data_loader=self.val_loader,
                    save_models=self.save_models,
                    model=self.model,
                    optimizer=self.optimizer,
                    loss_function=self.loss_function,
                    max_epochs=self.max_epochs,
                    with_scheduler=self.with_scheduler,
                    metric_calculator=self.metric_calculator,
                    class_col_name=self.class_col_name,
                    image_key=self.image_key,
                    amp=self.amp,
                    val_interval=self.val_interval,
                    device=self.device,
                    show_interval=10,
                    early_stop_callback=self.early_stop_callback,
                    patience=self.early_stop_patience,
                    early_stop_verbose=self.early_stop_verbose,
                    early_stop_delta=self.early_stop_delta,
                    save_path=self.save_path,
                    class_num=self.num_class,
                    grid_search_ID=self.grid_search_id + '_freeze',
                    fold_num=fold
                )

                # Free GPU memory
                self.free_memory()

                # Start training and get the updated model
                self.model = trainer.start(return_model=True)

                # Prepare settings for unfreezing
                self.prepare_unfreeze_settings()

                # Recreate data loaders (force reload)
                self.make_dataloader(self.batch_size, force_load=True)

                # Check data and model
                self.check_data()
                self.check_model()

                # Estimate VRAM usage
                vram_overloaded = self.estimate_vram_usage()

                # Initialize trainer for unfrozen model
                trainer = self.trainer(
                    train_data_loader=self.train_loader,
                    valid_data_loader=self.val_loader,
                    save_models=self.save_models,
                    model=self.model,
                    optimizer=self.optimizer,
                    loss_function=self.loss_function,
                    max_epochs=self.max_epochs,
                    with_scheduler=self.with_scheduler,
                    class_col_name=self.class_col_name,
                    image_key=self.image_key,
                    metric_calculator=self.metric_calculator,
                    amp=self.amp,
                    val_interval=self.val_interval,
                    device=self.device,
                    show_interval=10,
                    early_stop_callback=self.early_stop_callback,
                    patience=self.early_stop_patience,
                    early_stop_verbose=self.early_stop_verbose,
                    early_stop_delta=self.early_stop_delta,
                    save_path=self.save_path,
                    class_num=self.num_class,
                    l1_regulariztion_lambda=self.regularization_l1,
                    l2_regulariztion_lambda=self.regularization_l2,
                    grid_search_ID=self.grid_search_id + '_unfreeze',
                    fold_num=fold
                )

                # Start training
                trainer.start()
            else:
                # No freezing, proceed directly
                self.make_dataloader(self.batch_size)
                self.check_data()
                self.check_model()
                vram_overloaded = self.estimate_vram_usage()

                # Initialize trainer
                trainer = self.trainer(
                    train_data_loader=self.train_loader,
                    valid_data_loader=self.val_loader,
                    save_models=self.save_models,
                    model=self.model,
                    optimizer=self.optimizer,
                    loss_function=self.loss_function,
                    max_epochs=self.max_epochs,
                    with_scheduler=self.with_scheduler,
                    class_col_name=self.class_col_name,
                    image_key=self.image_key,
                    metric_calculator=self.metric_calculator,
                    amp=self.amp,
                    val_interval=self.val_interval,
                    device=self.device,
                    show_interval=10,
                    early_stop_callback=self.early_stop_callback,
                    patience=self.early_stop_patience,
                    early_stop_verbose=self.early_stop_verbose,
                    early_stop_delta=self.early_stop_delta,
                    save_path=self.save_path,
                    class_num=self.num_class,
                    grid_search_ID=self.grid_search_id,
                    fold_num=fold
                )

                # Start training
                trainer.start()

        # After all folds, update grid search results
        self.update_grid_search()

        # Attempt to plot results
        try:
            if self.freeze_base_first:
                # Load logs for unfreeze phase
                self.k_fold_ploter.load_logs(self.grid_search_id + '_unfreeze')
                self.k_fold_ploter.plot_loss(self.grid_search_id + '_unfreeze')
                self.k_fold_ploter.plot_metrics(self.grid_search_id + '_unfreeze')
                self.k_fold_ploter.plot_class_metrics(
                    self.labels_dict,
                    str(self.grid_search_id) + '_unfreeze'
                )
            else:
                # Load logs for normal phase
                self.k_fold_ploter.load_logs(self.grid_search_id)
                self.k_fold_ploter.plot_loss(self.grid_search_id)
                self.k_fold_ploter.plot_metrics(self.grid_search_id)
                self.k_fold_ploter.plot_class_metrics(
                    self.labels_dict,
                    self.grid_search_id
                )
        except Exception as e:
            # Ignore plotting errors
            pass

    # --------------------------------------------------------------------------
    # Method to Create Preprocessed Dataset
    # --------------------------------------------------------------------------

    def make_preprocessed_dataset(self):
        """
        Preprocess the dataset and save the preprocessed images.

        This method applies the validation transformations to the dataset,
        saves the transformed images as .npy files, updates the data paths,
        and adjusts the image key to point to the preprocessed images.
        """
        # Convert data to list of records
        data_list = self.data.to_dict('records')

        # Define base and new paths
        base_path = Path(self.folder_path).joinpath('images')
        new_path = Path(self.folder_path).joinpath('preprocessed_images')

        # Extract image names
        names = [data['image'].split(str(base_path))[-1][1:] for data in data_list]

        # Use validation transforms
        transforms = self.val_transforms

        # Create dataset and dataloader
        ds = CustomDataset(data_list, transforms)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        # Ensure new directory exists
        new_path.mkdir(parents=True, exist_ok=True)

        # List to store new image paths
        list_new_images_path = []

        # Iterate over data and save preprocessed images
        for idx, data in enumerate(dl):
            image = data['image'][0].numpy()
            target_path = new_path.joinpath(names[idx].replace('.png', '.npy'))
            list_new_images_path.append(str(target_path))
            np.save(target_path, image)

        # Update data with new image paths
        self.data['preprocessed_image'] = list_new_images_path

        # Update image key
        self.image_key = 'preprocessed_image'

        # Adjust transforms for raw data
        self.make_raw_transforms()

    # --------------------------------------------------------------------------
    # Method to Perform Grid Search
    # --------------------------------------------------------------------------

    def search(self, start_fold=None):
        """
        Perform the grid search over all configurations.

        Args:
            start_fold (int, optional): Fold number to start from.

        This method iterates over all grid search configurations,
        sets up the settings, detects the fold number to start from,
        performs k-fold cross-validation, and resets settings after each run.
        """
        # Read config file if not already loaded
        if self.config is None:
            self.read_config()

        # Set the grid search CSV directory
        grid_search_csv_dir = self.grid_search_dir.joinpath('grid_search.csv')

        # Check if the grid search CSV exists, if not generate it
        if not os.path.exists(grid_search_csv_dir):
            self.generate_grid_search_combinations()

        # Select the search list (grid search IDs to process)
        self.select_search_list()

        # Load data if not already loaded
        if self.data is None:
            self.data = self.dataset_loader()
            self.labels_dict = self.dataset_loader.labels

        # Iterate over each grid search ID
        for idx in range(len(self.grid_search_list)):
            # Set the current grid search ID
            self.grid_search_id = self.grid_search_list[idx]

            # Prepare settings for current grid search configuration
            self.prepare_settings()

            # Detect starting fold number in case of resuming
            self.detect_fold_number()

            # Perform k-fold cross-validation
            self.k_fold(start_fold=start_fold)

            # Reset settings for next grid search configuration
            self.reset_grid_search()

    # --------------------------------------------------------------------------
    # Method to Create Metric Calculator
    # --------------------------------------------------------------------------

    def make_metric_calculator(self):
        """
        Create an instance of metrics_calculator with the selected metrics.

        Returns:
            metrics_calculator: An instance configured with the selected metrics.
        """
        # Define a list of available metrics
        list_metrics = [
            "sensitivity", "specificity", "precision", "negative predictive value",
            "false discovery rate", "threat score", "accuracy", "balanced accuracy",
            "f1 score", "matthews correlation coefficient", "AUC"
        ]

        # If 'all' metrics are requested, select all available metrics
        if self.metrics == 'all':
            selected_metrics = list_metrics
        else:
            selected_metrics = self.metrics  # Assuming self.metrics is a list of metrics

        # Return an instance of metrics_calculator with the selected metrics
        return metrics_calculator(metrics=selected_metrics, num_class=self.num_class)

    # --------------------------------------------------------------------------
    # Method to Create Raw Data Transforms
    # --------------------------------------------------------------------------

    def make_raw_transforms(self):
        """
        Create transformations for raw preprocessed images.

        This method sets up the transformations for training and validation
        when using preprocessed images stored as .npy files.
        """
        self.train_transform = Compose([
            LoadNPY(keys=[self.image_key]),
            ToTensord(
                keys=[self.image_key, self.class_col_name],
                dtype=torch.float32,
                device=self.device
            )
        ])
        self.val_transform = Compose([
            LoadNPY(keys=[self.image_key]),
            ToTensord(
                keys=[self.image_key, self.class_col_name],
                dtype=torch.float32,
                device=self.device
            )
        ])
    # --------------------------------------------------------------------------
    # Method to Create Data Transformations
    # --------------------------------------------------------------------------

    def make_transforms(self, model_name, aug=False):
        """
        Create data transformations for training and validation datasets.

        Args:
            model_name (str): The name of the model to be used.
            aug (bool): Whether to include data augmentations. Default is False.

        This method constructs a list of transformations including loading images,
        preprocessing steps, optional augmentations, and converting data to tensors.
        It sets up `self.train_transforms` and `self.val_transforms` accordingly.
        """
        # Base transformations: load image and repeat channels to have 3 channels
        base = [
            LoadImaged(keys=[self.image_key], ensure_channel_first=True),
            RepeatChanneld(keys=[self.image_key], repeats=3)
        ]

        # Main preprocessing transformations from the configuration file for the model
        main = self.make_list_of_preprocessing(
            self.config['models'][model_name]['preprocessing']
        )

        # Optional augmentations if aug is True
        if aug:
            augmentations = self.make_augmentations(
                self.config['models'][model_name]['preprocessing']
            )

        # End transformations: convert image and class index to tensors
        end = [
            ToTensord(
                keys=[self.image_key, self.class_col_name],
                dtype=torch.float32,
                device=self.device
            )
        ]

        # Compose transformations for training
        if aug:
            self.train_transforms = Compose(base + augmentations + main + end)
        else:
            self.train_transforms = Compose(base + main + end)

        # Compose transformations for validation (no augmentations)
        self.val_transforms = Compose(base + main + end)

    # --------------------------------------------------------------------------
    # Method to Create Data Loaders
    # --------------------------------------------------------------------------

    def make_dataloader(self, batch_size, force_load=False):
        """
        Create data loaders for training and validation datasets.

        Args:
            batch_size (int): The batch size for the data loaders.
            force_load (bool): Whether to force reloading the data loaders even if they exist.

        This method initializes `self.train_loader` and `self.val_loader` using the
        specified batch size and the transformations defined in `self.train_transforms`
        and `self.val_transforms`.
        """
        # Check if data loaders need to be created or reloaded
        if (self.train_loader is None and self.val_loader is None) or force_load:
            # Create custom datasets with the data and transformations
            train_ds = CustomDataset(
                data=self.train_data_list, transform=self.train_transforms
            )
            val_ds = CustomDataset(
                data=self.val_data_list, transform=self.val_transforms
            )

            # Create data loaders with the datasets
            self.train_loader = DataLoader(
                train_ds,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=self.num_workers
            )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=self.num_workers
            )

    # --------------------------------------------------------------------------
    # Method to Replace the Classifier Layer in the Model
    # --------------------------------------------------------------------------

    def replace_classifier(self, model_name, cls_name, activation, hidden_params, drop_out):
        """
        Replace the classifier layer of the model with a custom classifier.

        Args:
            model_name (str): The name of the model being used.
            cls_name (str): The name of the classifier configuration to use.
            activation (str): The activation function to use in the classifier.
            hidden_params (int): The number of hidden units in the classifier.
            drop_out (float): The dropout rate to use in the classifier.

        This method defines custom classifiers and replaces the classifier layer
        of the model accordingly. It also handles freezing the base model layers
        if specified.
        """
        # Dictionary of activation functions
        activations = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh(),
            'Leakyrelu': torch.nn.LeakyReLU(),
        }

        # Dictionary of classifier configurations
        classifiers = {
            'medium': torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1, end_dim=-1),
                torch.nn.Linear(
                    in_features=self.classifier_in_features, out_features=hidden_params
                ),
                activations[activation],
                torch.nn.Dropout(drop_out),
                torch.nn.Linear(
                    in_features=hidden_params, out_features=self.num_class, bias=False
                )
            ),
            'medium_flatten': torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1, end_dim=-1),
                torch.nn.Linear(
                    in_features=self.classifier_in_features, out_features=hidden_params
                ),
                activations[activation],
                torch.nn.BatchNorm1d(hidden_params),
                torch.nn.Dropout(drop_out),
                torch.nn.Linear(
                    in_features=hidden_params, out_features=self.num_class, bias=False
                )
            ),
            'large': torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.classifier_in_features, out_features=hidden_params
                ),
                activations[activation],
                torch.nn.BatchNorm1d(hidden_params),
                torch.nn.Dropout(drop_out),
                torch.nn.Linear(
                    in_features=hidden_params,
                    out_features=int(hidden_params / 2),
                    bias=False
                ),
                activations[activation],
                torch.nn.BatchNorm1d(int(hidden_params / 2)),
                torch.nn.Dropout(drop_out),
                torch.nn.Linear(
                    in_features=int(hidden_params / 2),
                    out_features=self.num_class,
                    bias=False
                )
            ),
        }

        # Freeze base model layers if required
        if self.freeze_base_first:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the classifier layer based on the classifier key
        if self.classifier_key == 'classifier':
            self.model.classifier = classifiers[cls_name]
        elif self.classifier_key == 'medium_flatten':
            self.model.classifier = classifiers['medium_flatten']
        elif self.classifier_key == 'fc':
            self.model.fc = classifiers[cls_name]
        elif self.classifier_key == 'head':
            self.model.head = classifiers[cls_name]
        elif self.classifier_key == 'heads':
            self.model.heads = classifiers[cls_name]

    # --------------------------------------------------------------------------
    # Method to Select and Initialize the Model
    # --------------------------------------------------------------------------

    def model_selection(self, model_name, weights_name):
        """
        Select and initialize the model architecture with optional pre-trained weights.

        Args:
            model_name (str): The name of the model to initialize.
            weights_name (str): The name of the pre-trained weights to use.

        Returns:
            torch.nn.Module: The initialized model.

        This method looks up the model from a predefined dictionary and initializes it.
        If transfer learning is enabled, it loads the specified pre-trained weights.
        """
        # Dictionary of available models
        models = {
            'vgg16': torchvision.models.vgg16,
            'vgg19': torchvision.models.vgg19,
            'densenet121': torchvision.models.densenet121,
            'densenet161': torchvision.models.densenet161,
            'densenet169': torchvision.models.densenet169,
            'densenet201': torchvision.models.densenet201,
            'efficientnet_b0': torchvision.models.efficientnet_b0,
            'efficientnet_b3': torchvision.models.efficientnet_b3,
            'efficientnet_b7': torchvision.models.efficientnet_b7,
            'efficientnet_v2_s': torchvision.models.efficientnet_v2_s,
            'efficientnet_v2_m': torchvision.models.efficientnet_v2_m,
            'mobilenet_v2': torchvision.models.mobilenet_v2,
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152,
            'wide_resnet50_2': torchvision.models.wide_resnet50_2,
            'inception_v3': torchvision.models.inception_v3,
            'vit_b_16': torchvision.models.vit_b_16,
            'maxvit_t': torchvision.models.maxvit_t,
            'swin_t': torchvision.models.swin_t,
            'swin_v2_t': torchvision.models.swin_v2_t,
        }

        # Check if the model name is valid
        if model_name not in models:
            raise ValueError(f'Model {model_name} is not in the implemented list')

        # Initialize the model with or without pre-trained weights
        if self.transfer_learning:
            return models[model_name](weights=weights_name)
        else:
            return models[model_name]()

    # --------------------------------------------------------------------------
    # Method to Create Preprocessing Transformations
    # --------------------------------------------------------------------------

    def make_list_of_preprocessing(self, dict_preprocessing):
        """
        Create a list of preprocessing transformations based on the configuration.

        Args:
            dict_preprocessing (dict): Dictionary containing preprocessing configurations.

        Returns:
            list: List of preprocessing transformations.

        This method reads the preprocessing steps from the configuration dictionary
        and constructs a list of transformations accordingly.
        """
        # List to store preprocessing transforms
        list_preprocess = []

        # Order of preprocessing steps to consider
        final_names = ['resize', 'crop', 'scale', 'normalize']

        # Available preprocessing steps in the configuration
        names = list(dict_preprocessing.keys())

        # Filter final_names to include only available steps
        final_names = [n for n in final_names if n in names]

        # Iterate over the preprocessing steps
        for name in final_names:
            if name == 'resize':
                key = dict_preprocessing[name]['key']
                size = tuple(dict_preprocessing[name]['size'])
                mode = dict_preprocessing[name]['mode']
                list_preprocess.append(
                    Resized(keys=[key], spatial_size=size, mode=mode)
                )
            elif name == 'crop':
                key = dict_preprocessing[name]['key']
                roi = tuple(dict_preprocessing[name]['roi'])
                list_preprocess.append(
                    CenterSpatialCropd(keys=[key], roi_size=roi)
                )
            elif name == 'normalize':
                key = dict_preprocessing[name]['key']
                means = tuple(dict_preprocessing[name]['means'])
                stds = tuple(dict_preprocessing[name]['stds'])
                list_preprocess.append(
                    Normalized(keys=[key], means=means, stds=stds)
                )
            elif name == 'scale':
                key = dict_preprocessing[name]['key']
                minv = dict_preprocessing[name]['minv']
                maxv = dict_preprocessing[name]['maxv']
                list_preprocess.append(
                    ScaleIntensityd(keys=[key], minv=minv, maxv=maxv)
                )

        # Return the list of preprocessing transforms
        return list_preprocess

    # --------------------------------------------------------------------------
    # Method to Create Data Augmentations
    # --------------------------------------------------------------------------

    def make_augmentations(self, dict_preprocessing):
        """
        Create a list of data augmentation transformations.

        Args:
            dict_preprocessing (dict): Dictionary containing preprocessing configurations.

        Returns:
            list: List of augmentation transformations.

        This method defines and returns a list of augmentation transformations
        such as random adaptive histogram equalization, random rotation, and random scaling.
        """
        # List to store augmentation transforms
        list_augs = []

        # Define possible augmentation names
        augmentation_names = ['random_adapthist', 'random_scale', 'random_rotate']

        # Iterate over augmentation names and add corresponding transforms
        for name in augmentation_names:
            if name == 'random_adapthist':
                list_augs.append(
                    RandomAdapthistd(keys=[self.image_key], p=0.2)
                )
            elif name == 'random_rotate':
                list_augs.append(
                    RandomRotate(keys=[self.image_key], p=0.2)
                )
            elif name == 'random_scale':
                list_augs.append(
                    RandomScale(keys=[self.image_key], scale_limit=0.3, p=0.2)
                )

        # Return the list of augmentation transforms
        return list_augs

    # --------------------------------------------------------------------------
    # Method to Get Optimizer Class
    # --------------------------------------------------------------------------

    def get_optimizer(self, opt_name):
        """
        Get the optimizer class based on its name.

        Args:
            opt_name (str): The name of the optimizer ('adam', 'adamw', 'rmsprop', 'sgd').

        Returns:
            torch.optim.Optimizer: The optimizer class corresponding to the name.

        This method maps the optimizer name to its corresponding PyTorch optimizer class.
        """
        # Map optimizer names to classes
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'rmsprop': torch.optim.RMSprop,
            'sgd': torch.optim.SGD
        }

        # Return the optimizer class
        return optimizers.get(opt_name)

    # --------------------------------------------------------------------------
    # Method to Get Loss Function Class
    # --------------------------------------------------------------------------

    def get_loss_function(self, loss_func_name):
        """
        Get the loss function class based on its name.

        Args:
            loss_func_name (str): The name of the loss function ('crossentropy', 'focal_loss').

        Returns:
            Callable: The loss function class corresponding to the name.

        This method maps the loss function name to its corresponding PyTorch loss function class.
        """
        # Map loss function names to classes
        loss_functions = {
            'crossentropy': torch.nn.CrossEntropyLoss,
            'focal_loss': FocalLoss  # Assuming FocalLoss is defined elsewhere
        }

        # Return the loss function class
        return loss_functions.get(loss_func_name)

    # --------------------------------------------------------------------------
    # Method to Calculate Class Weights
    # --------------------------------------------------------------------------

    def calculate_class_weights(self):
        """
        Calculate class weights for handling class imbalance.

        Returns:
            torch.Tensor: A tensor containing the class weights.

        This method computes class weights based on the frequency of each class in the dataset.
        It is useful for addressing class imbalance during training.
        """
        # Get class labels from the data
        y = self.data[self.class_col_name].tolist()

        # If specific class names are selected, filter data accordingly
        if self.list_classes_names != 'all':
            data = self.data[
                self.data[self.class_col_name].isin(self.list_classes_names)
            ]
            data = data.reset_index(drop=True)
            y = data[self.class_col_name].tolist()
            del data  # Clean up

        # Compute class weights using scikit-learn
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )

        # Convert weights to a tensor and move to the appropriate device
        return torch.from_numpy(weights).float().to(self.device)
