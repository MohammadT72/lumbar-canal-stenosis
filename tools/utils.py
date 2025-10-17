import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as tt
import albumentations as A
from PIL import Image as ImageReader

class FocalLoss(torch.nn.Module):
    """
    Implementation of Focal Loss as described in the paper:
    "Focal Loss for Dense Object Detection" by Lin et al.
    
    Args:
        alpha (float, optional): Weighting factor for the rare class.
        gamma (float, optional): Focusing parameter to down-weight easy examples.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'.
        class_weights (Tensor, optional): A manual rescaling weight given to each class.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        # Compute the probabilities of the targets
        pt = torch.exp(-ce_loss)
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        # Apply alpha factor if provided
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LoadNPY:
    """
    Transformer that loads images stored in .npy files.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
    """
    def __init__(self, keys):
        self.image = keys[0]

    def __call__(self, data):
        # Retrieve the list of image paths
        img_list = data[self.image]
        if not isinstance(img_list, list):
            img_list = [data[self.image]]
        loaded_images = []
        for img_dir in img_list:
            # Load the image from .npy file
            image_array = np.load(img_dir)
            # Convert to torch tensor
            loaded_images.append(torch.from_numpy(image_array))
        # Stack images along a new dimension
        data[self.image] = torch.stack(loaded_images, dim=0)
        return data

class LoadImaged:
    """
    Transformer that loads images from file paths using PIL.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        ensure_channel_first (bool): If True, ensures the channel dimension is first.
    """
    def __init__(self, keys, ensure_channel_first=True):
        self.channel_first = ensure_channel_first
        self.reader = ImageReader
        self.image = keys[0]

    def __call__(self, data):
        # Retrieve the list of image paths
        img_list = data[self.image]
        if not isinstance(img_list, list):
            img_list = [data[self.image]]
        loaded_images = []
        for img_dir in img_list:
            # Open image and convert to numpy array
            image_array = np.array(self.reader.open(img_dir))
            if self.channel_first:
                if image_array.ndim > 2:
                    # Move channel axis to first dimension
                    image_array = np.moveaxis(image_array, -1, 0)
                elif image_array.ndim == 2:
                    # Add channel dimension
                    image_array = np.expand_dims(image_array, 0)
            else:
                if image_array.ndim == 2:
                    # Add channel dimension at the end
                    image_array = np.expand_dims(image_array, -1)
            # Convert to torch tensor
            loaded_images.append(torch.from_numpy(image_array))
        # Stack images along a new dimension
        data[self.image] = torch.stack(loaded_images, dim=0)
        # Record channel ordering
        data['channel'] = 'first' if self.channel_first else 'last'
        return data

class RandomAdapthistd:
    """
    Transformer that applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to images with a probability p.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        p (float): Probability of applying the transform.
    """
    def __init__(self, keys, p=0.5):
        self.p = p
        self.image = keys[0]

    def __call__(self, data):
        img_list = data[self.image]
        adapted_images = []
        for img in img_list:
            if np.random.rand() > self.p:
                # Skip transformation with probability (1-p)
                adapted_images.append(img)
                continue
            # Randomly select clip_limit for CLAHE
            clip_limit = int(np.random.randint(1, 6, 1)[0])
            clahe = A.CLAHE(clip_limit=clip_limit, p=1.0)  # p=1.0 since we're applying it manually
            # Extract the first channel
            if data['channel'] == 'first':
                channel_dim = img.shape[0]
                image_array = img[0, :, :]
            else:
                channel_dim = img.shape[-1]
                image_array = img[:, :, 0]
            if isinstance(image_array, torch.Tensor):
                image_array = image_array.numpy()
            # Convert to uint8
            image_array = image_array.astype(np.uint8)
            # Apply CLAHE
            image_array = clahe(image=image_array)['image']
            # Convert back to tensor
            tensorized = torch.from_numpy(image_array)
            # Repeat across channels
            if data['channel'] == 'first':
                tensorized = tensorized.unsqueeze(0).repeat(channel_dim, 1, 1)
            else:
                tensorized = tensorized.unsqueeze(-1).repeat(1, 1, channel_dim)
            adapted_images.append(tensorized)
        data[self.image] = torch.stack(adapted_images, dim=0)
        return data

class RepeatChanneld:
    """
    Transformer that repeats a single channel image to create a multi-channel image.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        repeats (int): Number of times to repeat the channel.
    """
    def __init__(self, keys, repeats=3):
        self.repeats = repeats
        self.image = keys[0]

    def __call__(self, data):
        img_list = data[self.image]
        repeated_images = []
        for img in img_list:
            if data['channel'] == 'first':
                # Extract the first channel and repeat it
                image_array = [img[0, :, :]] * self.repeats
                image_array = torch.stack(image_array, dim=0)
            else:
                image_array = [img[:, :, 0]] * self.repeats
                image_array = torch.stack(image_array, dim=-1)
            repeated_images.append(image_array)
        data[self.image] = torch.stack(repeated_images, dim=0)
        return data

class Resized:
    """
    Transformer that resizes images to a specified spatial size.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        spatial_size (tuple): Desired output size.
        mode (str): Interpolation mode to use for resizing.
    """
    def __init__(self, keys, spatial_size=(224, 224), mode='bilinear'):
        # Select interpolation mode
        interpolation_modes = {
            'bilinear': tt.InterpolationMode.BILINEAR,
            'bicubic': tt.InterpolationMode.BICUBIC,
            'nearest': tt.InterpolationMode.NEAREST,
            'nearest_exact': tt.InterpolationMode.NEAREST_EXACT
        }
        if mode not in interpolation_modes:
            raise ValueError(f"Unsupported interpolation mode: {mode}")
        interpolation = interpolation_modes[mode]
        self.trans = tt.Resize(spatial_size, interpolation=interpolation, antialias=False)
        self.image = keys[0]

    def __call__(self, data):
        img_list = data[self.image]
        resized_images = []
        for img in img_list:
            # Move channel dimension to first if necessary
            if data['channel'] == 'last':
                img = torch.moveaxis(img, -1, 0)
            # Apply resize transform
            image_array = self.trans(img)
            # Move channel dimension back if necessary
            if data['channel'] == 'last':
                image_array = torch.moveaxis(image_array, 0, -1)
            resized_images.append(image_array)
        data[self.image] = torch.stack(resized_images, dim=0)
        return data

class ScaleIntensityd:
    """
    Transformer that scales the intensity of images to a specified range.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        minv (float): Minimum value after scaling.
        maxv (float): Maximum value after scaling.
    """
    def __init__(self, keys, minv=0.0, maxv=1.0):
        self.minv = minv
        self.maxv = maxv
        self.image = keys[0]

    def __call__(self, data):
        img_list = data[self.image]
        mina = img_list.min()
        maxa = img_list.max()
        if mina == maxa:
            # Avoid division by zero if all values are the same
            data[self.image] = img_list * self.minv if self.minv is not None else img_list
            return data
        # Normalize to [0, 1]
        norm = (img_list - mina) / (maxa - mina)
        if (self.minv is None) or (self.maxv is None):
            data[self.image] = norm
            return data
        # Scale to [minv, maxv]
        data[self.image] = (norm * (self.maxv - self.minv)) + self.minv
        return data

class CenterSpatialCropd:
    """
    Transformer that crops the image at the center to a specified ROI size.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        roi_size (tuple): Desired output size.
    """
    def __init__(self, keys, roi_size=(224, 224)):
        self.trans = tt.CenterCrop(size=roi_size)
        self.image = keys[0]

    def __call__(self, data):
        img_list = data[self.image]
        cropped_images = []
        for img in img_list:
            # Move channel dimension to first if necessary
            if data['channel'] == 'last':
                img = torch.moveaxis(img, -1, 0)
            # Apply center crop
            image_array = self.trans(img)
            # Move channel dimension back if necessary
            if data['channel'] == 'last':
                image_array = torch.moveaxis(image_array, 0, -1)
            cropped_images.append(image_array)
        data[self.image] = torch.stack(cropped_images, dim=0)
        return data

class ToTensord:
    """
    Transformer that converts data to torch tensors.

    Args:
        keys (list): List of keys in the data dict to convert.
        dtype (torch.dtype): Desired data type of the tensor.
        device (str or torch.device): Desired device of the tensor.
    """
    def __init__(self, keys, dtype=torch.float32, device='cpu'):
        self.dtype = dtype
        self.device = device
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            value = data[key]
            if not isinstance(value, torch.Tensor):
                if isinstance(value, int):
                    value = torch.tensor(value)
                else:
                    value = torch.from_numpy(value)
            data[key] = value.to(self.dtype).to(self.device)
        return data

class Normalized:
    """
    Transformer that normalizes images using specified mean and standard deviation.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        means (tuple): Means for each channel.
        stds (tuple): Standard deviations for each channel.
    """
    def __init__(self, keys, means=(0.485,), stds=(0.229,)):
        self.norm = A.Normalize(mean=means, std=stds, max_pixel_value=1.0, p=1.0)
        self.image = keys[0]

    def __call__(self, data):
        # Remove batch dimension if present
        img = torch.squeeze(data[self.image])
        # Move channels to last dimension and convert to numpy
        img = torch.moveaxis(img, 0, 2).numpy()
        # Apply normalization
        normalized = self.norm(image=img)['image']
        # Convert back to tensor
        normalized = torch.from_numpy(normalized.astype(np.float32))
        # Move channels back to first dimension
        normalized = torch.moveaxis(normalized, 2, 0)
        data[self.image] = normalized
        return data

class RandomRotate:
    """
    Transformer that randomly rotates images within a specified angle range.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        angle_limit (int or tuple): Range of angles for rotation.
        p (float): Probability of applying the transform.
    """
    def __init__(self, keys, angle_limit=(-90, 90), p=0.5):
        self.rotate = A.Rotate(
            limit=angle_limit,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=False,
            p=p
        )
        self.image = keys[0]

    def __call__(self, data):
        img_list = data[self.image]
        rotated_images = []
        for img in img_list:
            # Extract the first channel
            if data['channel'] == 'first':
                channel_dim = img.shape[0]
                image_array = img[0, :, :]
            else:
                channel_dim = img.shape[-1]
                image_array = img[:, :, 0]
            if isinstance(image_array, torch.Tensor):
                image_array = image_array.numpy()
            # Convert to uint8
            image_array = image_array.astype(np.uint8)
            # Apply rotation
            image_array = self.rotate(image=image_array)['image']
            # Convert back to tensor
            tensorized = torch.from_numpy(image_array)
            # Repeat across channels
            if data['channel'] == 'first':
                tensorized = tensorized.unsqueeze(0).repeat(channel_dim, 1, 1)
            else:
                tensorized = tensorized.unsqueeze(-1).repeat(1, 1, channel_dim)
            rotated_images.append(tensorized)
        data[self.image] = torch.stack(rotated_images, dim=0)
        return data

class RandomScale:
    """
    Transformer that randomly scales images by cropping and resizing.

    Args:
        keys (list): List of keys in the data dict to apply the transformation.
        scale_limit (float): Fractional limit for scaling (e.g., 0.1 means up to 10% scaling).
        mode (str): Interpolation mode to use for resizing.
        p (float): Probability of applying the transform.
    """
    def __init__(self, keys, scale_limit=0.1, mode='bilinear', p=0.5):
        # Select interpolation mode
        interpolation_modes = {
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'nearest': cv2.INTER_NEAREST,
            'nearest_exact': cv2.INTER_NEAREST_EXACT
        }
        if mode not in interpolation_modes:
            raise ValueError(f"Unsupported interpolation mode: {mode}")
        self.interpolation = interpolation_modes[mode]
        self.image = keys[0]
        self.scale_limit = scale_limit
        self.p = p

    def __call__(self, data):
        if np.random.rand() <= self.p:
            img_list = data[self.image]
            scaled_images = []
            for img in img_list:
                # Extract the first channel
                if data['channel'] == 'first':
                    channel_dim = img.shape[0]
                    image_array = img[0, :, :]
                else:
                    channel_dim = img.shape[-1]
                    image_array = img[:, :, 0]
                if isinstance(image_array, torch.Tensor):
                    image_array = image_array.numpy()
                # Convert to uint8
                image_array = image_array.astype(np.uint8)
                h, w = image_array.shape
                # Randomly select scaling factor within scale_limit
                scale_factor = 1.0 - np.random.uniform(0, self.scale_limit)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                # Compute top-left corner for cropping
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                # Crop image
                image_array = image_array[top:top + new_h, left:left + new_w]
                # Resize back to original size
                image_array = cv2.resize(image_array, (w, h), interpolation=self.interpolation)
                # Convert back to tensor
                tensorized = torch.from_numpy(image_array)
                # Repeat across channels
                if data['channel'] == 'first':
                    tensorized = tensorized.unsqueeze(0).repeat(channel_dim, 1, 1)
                else:
                    tensorized = tensorized.unsqueeze(-1).repeat(1, 1, channel_dim)
                scaled_images.append(tensorized)
            data[self.image] = torch.stack(scaled_images, dim=0)
        return data
