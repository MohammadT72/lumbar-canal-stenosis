# main.py

import argparse
import torch
import yaml
from pathlib import Path

# Import the grid_searcher class
# Make sure to adjust the import path based on your project structure
from tools.grid_search import grid_searcher

def main():
    """
    Main function to start the grid search training process.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Start grid search training process.')

    # Add arguments
    parser.add_argument('--folder_path', type=str, default='./datasets/Final_dataset',
                        help='Path to the dataset folder.')
    parser.add_argument('--save_path', type=str, default='./results',
                        help='Path to save training results and models.')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to the grid search configuration YAML file. If not provided, uses default in folder_path.')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs for training.')
    parser.add_argument('--num_class', type=int, default=5,
                        help='Number of classes in the dataset.')
    parser.add_argument('--list_classes_names', type=str, nargs='+', default='all',
                        help='List of class names to include or "all".')
    parser.add_argument('--col_name', type=str, default='int_ID',
                        help='Column name for the identifier.')
    parser.add_argument('--class_col_name', type=str, default='class_idx',
                        help='Column name for the class labels.')
    parser.add_argument('--image_key', type=str, default='image',
                        help='Key for the image data.')
    parser.add_argument('--target_col_name', type=str, default=None,
                        help='Column name for target labels if different from class_col_name.')
    parser.add_argument('--target_map', type=str, default=None,
                        help='Path to a YAML file containing mapping for target labels.')
    parser.add_argument('--train_cache_rate', type=float, default=0.5,
                        help='Fraction of training data to cache.')
    parser.add_argument('--val_cache_rate', type=float, default=0.5,
                        help='Fraction of validation data to cache.')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of folds for k-fold cross-validation.')
    parser.add_argument('--with_scheduler', action='store_true',
                        help='Whether to use a learning rate scheduler.')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Interval for validation during training.')
    parser.add_argument('--save_models', action='store_true',
                        help='Whether to save the trained models.')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Whether to use weighted loss.')
    parser.add_argument('--gpu', action='store_true',
                        help='Whether to use GPU acceleration.')
    parser.add_argument('--amp', action='store_true',
                        help='Whether to use Automatic Mixed Precision (AMP).')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading.')
    parser.add_argument('--no_check', action='store_true',
                        help='Disable checking data consistency.')
    parser.add_argument('--no_transfer_learning', action='store_true',
                        help='Disable transfer learning.')
    parser.add_argument('--over_sampling', action='store_true',
                        help='Whether to apply over-sampling to balance classes.')
    parser.add_argument('--no_early_stop_callback', action='store_true',
                        help='Disable early stopping.')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Patience epochs for early stopping.')
    parser.add_argument('--early_stop_delta', type=float, default=0,
                        help='Minimum change to qualify as an improvement.')
    parser.add_argument('--early_stop_verbose', action='store_true',
                        help='Enable verbosity for early stopping.')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help='Fraction of data to use.')
    parser.add_argument('--folder_base', action='store_true',
                        help='Indicates if the folder is a base folder.')

    # Parse arguments
    args = parser.parse_args()

    # Process list_classes_names argument
    if args.list_classes_names != 'all':
        if isinstance(args.list_classes_names, list):
            list_classes_names = args.list_classes_names
        else:
            list_classes_names = [args.list_classes_names]
    else:
        list_classes_names = 'all'

    # Load target_map if provided
    target_map = None
    if args.target_map is not None:
        with open(args.target_map, 'r') as f:
            target_map = yaml.safe_load(f)

    # Create an instance of the grid_searcher class with provided arguments
    gs = grid_searcher(
        folder_path=args.folder_path,
        folder_base=args.folder_base,
        max_epochs=args.max_epochs,
        num_class=args.num_class,
        list_classes_names=list_classes_names,
        col_name=args.col_name,
        class_col_name=args.class_col_name,
        image_key=args.image_key,
        target_col_name=args.target_col_name,
        target_map=target_map,
        train_cache_rate=args.train_cache_rate,
        val_cache_rate=args.val_cache_rate,
        k=args.k,
        with_scheduler=args.with_scheduler,
        val_interval=args.val_interval,
        save_path=args.save_path,
        metrics='all',
        save_models=args.save_models,
        weighted_loss=args.weighted_loss,
        gpu=args.gpu,
        amp=args.amp,
        num_workers=args.num_workers,
        check=not args.no_check,
        transfer_learning=not args.no_transfer_learning,
        over_sampling=args.over_sampling,
        early_stop_callback=not args.no_early_stop_callback,
        early_stop_patience=args.early_stop_patience,
        early_stop_delta=args.early_stop_delta,
        early_stop_verbose=args.early_stop_verbose,
        data_fraction=args.data_fraction
    )

    # Set the config path if provided
    if args.config_path is not None:
        gs.config_path = args.config_path

    # Read the grid search configuration from the YAML file
    gs.read_config()

    # Start the grid search process
    gs.search()

if __name__ == '__main__':
    main()
