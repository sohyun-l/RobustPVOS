import os
import numpy as np
from PIL import Image
import cv2
import warnings
import shutil
import albumentations as A
import imgaug.augmenters as iaa
import random
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import hashlib
import inspect

if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'object'):
    np.object = object

warnings.filterwarnings('ignore')


def snow(x, severity=1, angle=0, density=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    brightness_coeff = 0.8 + (severity - 1) * 0.03
    snow_point_lower = 0.01 + (severity - 1) * 0.02
    snow_point_upper = 0.03 + (severity - 1) * 0.02
    if density is None:
        density_val = 0.01 + (severity - 1) * 0.02
    else:
        density_val = density
    transform = A.Compose([
        A.RandomSnow(brightness_coeff=brightness_coeff, 
                     snow_point_lower=snow_point_lower, 
                     snow_point_upper=snow_point_upper, p=1)],
    )
    x = transform(image=x)['image']
    aug = iaa.Snowflakes(density=density_val, flake_size=(0.6, 0.8), speed=(0.01, 0.015), angle=angle, random_state=random_state)
    x = aug.augment_image(x)
    return x

def fog(x, severity=1, fog_coef_lower=None, fog_coef_upper=None, alpha_coef=None, random_state=None):
    x = np.array(x)
    MAX_SIZE = 10000
    h, w = x.shape[:2]
    resize_needed = False
    original_size = (w, h)
    if h > MAX_SIZE or w > MAX_SIZE:
        resize_needed = True
        if h > w:
            new_h = MAX_SIZE
            new_w = int(w * MAX_SIZE / h)
        else:
            new_w = MAX_SIZE
            new_h = int(h * MAX_SIZE / w)
        x = cv2.resize(x, (new_w, new_h))
    severity = max(1, min(5, severity))
    if fog_coef_lower is None:
        fog_coef_lower = 0.1 + (severity - 1) * 0.05
    if fog_coef_upper is None:
        fog_coef_upper = 0.15 + (severity - 1) * 0.05
    if alpha_coef is None:
        alpha_coef = 0.05 + (severity - 1) * 0.02
    aug = iaa.Fog(random_state=random_state)
    x = aug.augment_image(x)
    transform = A.Compose(
        [A.RandomFog(fog_coef_lower=fog_coef_lower, fog_coef_upper=fog_coef_upper, alpha_coef=alpha_coef, p=1)],
    )
    x = transform(image=x)
    x = x['image']
    if resize_needed:
        x = cv2.resize(x, original_size)
    return x

def rain(x, severity=1, drop_size=None, speed=None, density=None, nb_iterations=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if drop_size is None:
        drop_size_val = 0.15 + (severity - 1) * 0.05
        drop_size_min = drop_size_val
        drop_size_max = drop_size_val + 0.02
    else:
        drop_size_min, drop_size_max = drop_size
    if speed is None:
        speed_val = 0.02 + (severity - 1) * 0.02
        speed_min = speed_val
        speed_max = speed_val + 0.01
    else:
        speed_min, speed_max = speed
    
    if nb_iterations is None:
        nb_iterations = 1 + (severity - 1) * 1
    else:
        nb_iterations = int(nb_iterations)
    # print(drop_size_min, drop_size_max, speed_min, speed_max)
    

    
    aug = iaa.Rain(drop_size=(drop_size_min, drop_size_max), speed=(speed_min, speed_max), nb_iterations=nb_iterations, random_state=random_state)
    x = aug.augment_image(x)
    return x

def gauss_noise(x, severity=1, var_limit=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if var_limit is None:
        var_min = 50 + (severity - 1) * 25
        var_max = 75 + (severity - 1) * 25
        var = (var_min, var_max)
    else:
        var = var_limit
    transform = A.Compose(
        [A.GaussNoise(var_limit=var, per_channel=True, p=1)],
    )
    x = transform(image=x)
    x = x['image']
    return x

def ISO_noise(x, severity=1, color_shift=None, intensity=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if color_shift is None:
        color_shift_min = 0.1 + (severity - 1) * 0.05
        color_shift_max = 0.15 + (severity - 1) * 0.05
        color_shift = (color_shift_min, color_shift_max)
    if intensity is None:
        intensity_min = 0.85 + (severity - 1) * 0.03
        intensity_max = 0.9 + (severity - 1) * 0.03
        intensity = (intensity_min, intensity_max)
    transform = A.Compose(
        [A.ISONoise(color_shift=color_shift, intensity=intensity, p=1)],
    )
    x = transform(image=x)
    x = x['image']
    return x

def impulse_noise(x, severity=1, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    multiplier_min = 1.1 + (severity - 1) * 0.1  # 1.1 ~ 1.5
    multiplier_max = 1.2 + (severity - 1) * 0.1  # 1.2 ~ 1.6
    multiplier = (multiplier_min, multiplier_max)
    
    transform = A.Compose( 
        [A.MultiplicativeNoise(multiplier=multiplier, p=1)],
    )     
    x = transform(image=x)
    x = x['image'] 
    return x

def resampling_blur(x, severity=1, resize_factor=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if resize_factor is None:
        resize_factor = 1.1 + (severity - 1) * 0.1
    ori_height, ori_width = x.shape[0], x.shape[1]
    new_height, new_width = int(x.shape[0]/resize_factor), int(x.shape[1]/resize_factor)
    img_down = cv2.resize(x, (new_width, new_height))
    x = cv2.resize(img_down, (ori_width, ori_height))
    return x

def motion_blur(x, severity=1, blur_limit=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if blur_limit is None:
        blur_limit = int(5 + (severity - 1) * 3)
        if blur_limit % 2 == 0:
            blur_limit += 1
    transform = A.Compose(
        [A.MotionBlur(blur_limit=blur_limit, p=1)],
    )
    x = transform(image=x)
    x = x['image']
    return x

def zoom_blur(x, severity=1, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    max_factor = 1.1 + (severity - 1) * 0.05  # 1.1 ~ 1.3
    step_factor_min = 0.02 + (severity - 1) * 0.01  # 0.02 ~ 0.06
    step_factor_max = 0.03 + (severity - 1) * 0.01  # 0.03 ~ 0.07
    step_factor = (step_factor_min, step_factor_max)
    
    transform = A.Compose(
        [A.ZoomBlur(max_factor=max_factor, step_factor=step_factor, p=1)],
    )    
    x = transform(image=x)
    x = x['image'] 
    return x

def color_jitter(x, severity=1, brightness=None, contrast=None, saturation=None, hue=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if brightness is None:
        brightness = 0.9 + (severity - 1) * 0.02
    if contrast is None:
        contrast = 0.1 + (severity - 1) * 0.02
    if saturation is None:
        saturation = 0.9 + (severity - 1) * 0.02
    if hue is None:
        hue = 0.9 + (severity - 1) * 0.02
    transform = A.Compose(
        [A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1)],
    )
    x = transform(image=x)
    x = x['image']
    return x

def compression(x, severity=1, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    quality_lower = 30 - (severity - 1) * 5  # 30 ~ 10
    quality_upper = 40 - (severity - 1) * 5  # 40 ~ 20
    
    transform = A.Compose(
        [A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=1)],
    )    
    x = transform(image=x)
    x = x['image']  
    return x

def elastic_transform(x, severity=1, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    alpha = 20 + (severity - 1) * 20      # 20 ~ 100
    sigma = 2 + (severity - 1) * 2        # 2 ~ 10
    alpha_affine = 1 + (severity - 1) * 1 # 1 ~ 5
    
    transform = A.Compose(
        [A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=1)],
    )    
    x = transform(image=x)
    x = x['image'] 
    return x

def frosted_glass_blur(x, severity=1, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    max_displacement = int(1 + (severity - 1) * 0.5)
    max_displacement = max(1, max_displacement)
    
    height, width, _ = x.shape
    augmented_image = np.copy(x)

    for y in range(height):
        for x_coord in range(width):
            displacement_x = np.random.randint(-max_displacement, max_displacement + 1)
            displacement_y = np.random.randint(-max_displacement, max_displacement + 1)

            target_x = max(0, min(width - 1, x_coord + displacement_x))
            target_y = max(0, min(height - 1, y + displacement_y))

            augmented_image[y, x_coord] = x[target_y, target_x]

    return augmented_image

def brightness(x, severity=1, factor=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if factor is None:
        factor = 0.95 - (severity - 1) * 0.01
    augmented_image = np.clip(x * factor, 0, 255).astype(np.uint8)
    return augmented_image

def contrast(x, severity=1, factor=None, random_state=None):
    x = np.array(x)
    severity = max(1, min(5, severity))
    if factor is None:
        factor = 1.1 + (severity - 1) * 0.02
    x = x.astype(np.float32)
    augmented_image = (x - 128) * factor + 128
    augmented_image = np.clip(augmented_image, 0, 255)
    augmented_image = augmented_image.astype(np.uint8)
    return augmented_image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def fourier_sampling(T, M_static, C=4, seed=None):
    """
    T: number of frames
    M_static: static magnitude (ex. 0.5)
    C: number of sine bases
    """
    if T <= 0:
        return np.array([])
    
    if T == 1:
        return np.array([M_static])
    
    rng = np.random.RandomState(seed)
    K = np.arange(2*T)
    Ms = np.zeros((C, T))
    ws = rng.dirichlet([1.0]*C)
    for b in range(C):
        fb = rng.uniform(0.2, 1.5)
        A = rng.uniform(0, 1)
        ob = rng.randint(0, T)
        x = np.sin(2 * fb * np.pi * K[ob:ob+T] / (T-1))
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = M_static - M_static * (A/1.0) + x * (2 * M_static * (A/1.0))
        Ms[b] = x
    M = np.sum(ws[:, None] * Ms, axis=0)
    return M  # shape: [T]


def process_image(args):
    input_path, output_path, corruption_fn, severity, *extra_args = args
    if os.path.exists(output_path):
        return True
    try:
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)
        try:
            sig = inspect.signature(corruption_fn)
            kwargs = {'severity': severity}
            param_names = list(sig.parameters.keys())
            for name, val in zip(param_names[2:], extra_args):
                kwargs[name] = val
            corrupted = corruption_fn(img_np, **kwargs)
        except Exception as e:
            print(f"Error applying corruption to {input_path}: {str(e)}")
            return False
        if not isinstance(corrupted, np.ndarray):
            print(f"Error: corruption function returned invalid type for {input_path}")
            return False
        try:
            output_img = Image.fromarray(np.uint8(corrupted))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_img.save(output_path)
            return True
        except Exception as e:
            print(f"Error saving corrupted image {output_path}: {str(e)}")
            return False
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def process_directory(input_dir, output_dir, corruption_fn, severity, gradual_change=True):
    os.makedirs(output_dir, exist_ok=True)
    image_files = []
    for root, _, files in os.walk(input_dir):
        if 'Annotations' in root:
            continue
        video_id = os.path.relpath(root, input_dir)
        video_seed = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % (2**32)
        set_seed(video_seed)
        rng = np.random.RandomState(video_seed)
        video_random_state = np.random.RandomState(video_seed + 12345)
        image_items = [item for item in files if item.lower().endswith(('.png', '.jpg', '.jpeg')) and not item.startswith('._')]
        image_items.sort()
        num_frames = len(image_items)
        
        if num_frames == 0:
            continue
        
        default_brightness = 0.85
        default_contrast = 0.15
        default_saturation = 0.85
        default_hue = 0.85
        default_fog_coef_lower = 0.15
        default_fog_coef_upper = 0.25
        default_alpha_coef = 0.09
        default_var_min = 100
        default_var_max = 150
        default_color_shift_min = 0.2
        default_color_shift_max = 0.25
        default_intensity_min = 0.85
        default_intensity_max = 0.9
        default_resize_factor = 1.2
        default_blur_limit = 7
        default_angle = 0
        default_density = 0.1
        default_drop_size = 0.25
        default_speed = 0.08
        default_nb_iterations = 1
        brightness_arr = fourier_sampling(num_frames, default_brightness, seed=video_seed+1)
        contrast_arr = fourier_sampling(num_frames, default_contrast, seed=video_seed+2)
        saturation_arr = fourier_sampling(num_frames, default_saturation, seed=video_seed+3)
        hue_arr = fourier_sampling(num_frames, default_hue, seed=video_seed+4)
        fog_coef_lower_arr = fourier_sampling(num_frames, default_fog_coef_lower, seed=video_seed+5)
        fog_coef_upper_arr = fourier_sampling(num_frames, default_fog_coef_upper, seed=video_seed+6)
        alpha_coef_arr = fourier_sampling(num_frames, default_alpha_coef, seed=video_seed+7)
        var_min_arr = fourier_sampling(num_frames, default_var_min, seed=video_seed+8)
        var_max_arr = fourier_sampling(num_frames, default_var_max, seed=video_seed+9)
        color_shift_min_arr = fourier_sampling(num_frames, default_color_shift_min, seed=video_seed+10)
        color_shift_max_arr = fourier_sampling(num_frames, default_color_shift_max, seed=video_seed+11)
        intensity_min_arr = fourier_sampling(num_frames, default_intensity_min, seed=video_seed+12)
        intensity_max_arr = fourier_sampling(num_frames, default_intensity_max, seed=video_seed+13)
        resize_factor_arr = fourier_sampling(num_frames, default_resize_factor, seed=video_seed+14)
        blur_limit_arr = fourier_sampling(num_frames, default_blur_limit, seed=video_seed+15)
        angle_arr = fourier_sampling(num_frames, default_angle, seed=video_seed+16)
        density_arr = fourier_sampling(num_frames, default_density, seed=video_seed+17)
        drop_size_arr = fourier_sampling(num_frames, default_drop_size, seed=video_seed+18)
        speed_arr = fourier_sampling(num_frames, default_speed, seed=video_seed+19)
        nb_iterations_arr = fourier_sampling(num_frames, default_nb_iterations, seed=video_seed+20)
        for idx, item in enumerate(image_items):
            input_path = os.path.join(root, item)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            frame_brightness = max(0.7, min(1.0, brightness_arr[idx]))
            frame_contrast = max(0.0, min(0.3, contrast_arr[idx]))
            frame_saturation = max(0.7, min(1.0, saturation_arr[idx]))
            frame_hue = max(0.0, min(0.3, abs(hue_arr[idx])))
            frame_fog_coef_lower = max(0.0, min(0.3, fog_coef_lower_arr[idx]))
            frame_fog_coef_upper = max(frame_fog_coef_lower + 0.01, min(0.4, fog_coef_upper_arr[idx]))
            frame_alpha_coef = max(0.0, min(0.2, alpha_coef_arr[idx]))
            frame_var_min = max(1, int(abs(var_min_arr[idx])))
            frame_var_max = max(frame_var_min + 1, int(abs(var_max_arr[idx])))
            frame_var_limit = (frame_var_min, frame_var_max)
            frame_color_shift_min = max(0.0, min(0.3, color_shift_min_arr[idx]))
            frame_color_shift_max = max(frame_color_shift_min + 0.01, min(0.4, color_shift_max_arr[idx]))
            frame_color_shift = (frame_color_shift_min, frame_color_shift_max)
            frame_intensity_min = max(0.7, min(1.0, intensity_min_arr[idx]))
            frame_intensity_max = max(frame_intensity_min + 0.01, min(1.0, intensity_max_arr[idx]))
            frame_intensity = (frame_intensity_min, frame_intensity_max)
            frame_resize_factor = max(1.05, abs(resize_factor_arr[idx]))
            frame_blur_limit = max(3, int(abs(blur_limit_arr[idx])))
            if frame_blur_limit % 2 == 0:
                frame_blur_limit += 1
            frame_angle = max(-90, min(90, angle_arr[idx]))
            frame_density = max(0.0, min(0.2, density_arr[idx]))
            frame_drop_size_val = max(0.05, min(0.4, drop_size_arr[idx]))  # 0.1~1.0 -> 0.05~0.4
            frame_drop_size = (frame_drop_size_val, frame_drop_size_val + 0.02)  # 0.05 -> 0.02
            frame_speed_val = max(0.01, min(0.2, speed_arr[idx]))  # 0.01~0.5 -> 0.01~0.2
            frame_speed = (frame_speed_val, frame_speed_val + 0.01)  # 0.02 -> 0.01
            frame_nb_iterations = max(1, int(abs(nb_iterations_arr[idx])))
            if corruption_fn.__name__ == 'snow':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_angle, frame_density, video_random_state))
            elif corruption_fn.__name__ == 'rain':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_drop_size, frame_speed, frame_density, frame_nb_iterations, video_random_state))
            elif corruption_fn.__name__ == 'color_jitter':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_brightness, frame_contrast, frame_saturation, frame_hue, video_random_state))
            elif corruption_fn.__name__ == 'fog':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_fog_coef_lower, frame_fog_coef_upper, frame_alpha_coef, video_random_state))
            elif corruption_fn.__name__ == 'gauss_noise':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_var_limit, video_random_state))
            elif corruption_fn.__name__ == 'ISO_noise':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_color_shift, frame_intensity, video_random_state))
            elif corruption_fn.__name__ == 'resampling_blur':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_resize_factor, video_random_state))
            elif corruption_fn.__name__ == 'motion_blur':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_blur_limit, video_random_state))
            elif corruption_fn.__name__ == 'brightness':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_brightness, video_random_state))
            elif corruption_fn.__name__ == 'contrast':
                image_files.append((input_path, output_path, corruption_fn, severity, frame_contrast, video_random_state))
            else:
                image_files.append((input_path, output_path, corruption_fn, severity, video_random_state))
        for item in files:
            if item.startswith('._') or item.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            input_path = os.path.join(root, item)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_path, output_path)
    if image_files:
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(process_image, image_files), total=len(image_files), desc="Processing images"))
        total_images = len(image_files)
        processed_images = sum(results)
        print(f"Successfully processed {processed_images} out of {total_images} images")

def main():
    parser = argparse.ArgumentParser(description='Apply corruptions to images')
    parser.add_argument('--input_dir', type=str, default="./data/MOSE/train",
                      help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default="./data/MOSE-C/train",
                      help='Output directory for corrupted images')
    parser.add_argument('--corruptions', nargs='+', 
                      choices=['snow', 'fog', 'rain', 'gauss_noise', 'ISO_noise', 
                              'resampling_blur', 'motion_blur', 'color_jitter'],
                      default=['snow'],
                      help='List of corruptions to apply')
    parser.add_argument('--severity', type=int, default=5,
                      help='Severity level of corruption (default: 5)')
    parser.add_argument('--gradual_change', action='store_true', default=True,
                      help='Apply gradual severity change within each video (default: True)')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of worker processes (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    if args.num_workers is not None:
        global cpu_count
        cpu_count = lambda: args.num_workers

    corruptions = {
        'snow': snow,
        'fog': fog,
        'rain': rain,
        'gauss_noise': gauss_noise,
        'ISO_noise': ISO_noise,
        'resampling_blur': resampling_blur,
        'motion_blur': motion_blur,
        'brightness': brightness,
        'contrast': contrast,
        'color_jitter': color_jitter,
        # 'frosted_glass_blur': frosted_glass_blur,
        # 'impulse_noise': impulse_noise,
        # 'zoom_blur': zoom_blur,
        # 'compression': compression,
        # 'elastic_transform': elastic_transform
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for corruption_name in args.corruptions:
        if corruption_name not in corruptions:
            print(f"Warning: {corruption_name} is not a valid corruption. Skipping...")
            continue
            
        print(f"Applying {corruption_name} with severity {args.severity}...")
        output_dir = os.path.join(args.output_dir, "severe", corruption_name)
        process_directory(args.input_dir, output_dir, corruptions[corruption_name], args.severity, args.gradual_change)
        print(f"Completed {corruption_name}")

if __name__ == "__main__":
    main() 