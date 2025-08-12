import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import NMF
from tqdm import tqdm
from typing import Tuple, List
from pytorch_grad_cam.utils.image import scale_cam_image
import torch.nn.functional as F

def load_and_preprocess_images(image_dir: str, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    images = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, fname)
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
    return torch.stack(images)  # (N, 3, H, W)


def extract_layer_features(content_encoder, image_batch: torch.Tensor, layers: List[int]) -> List[torch.Tensor]:
    content_encoder.eval()
    features_by_layer = [[] for _ in layers]
    with torch.no_grad():
        for img in image_batch:
            x = img.unsqueeze(0).cuda()
            out = x
            layer_outputs = []
            for i in range(6):
                out = getattr(content_encoder, f'layer{i+1}')(out)
                if (i + 1) in layers:
                    layer_outputs.append(out.squeeze(0).cpu())
            for idx, feat in enumerate(layer_outputs):
                features_by_layer[idx].append(feat)
    return [torch.stack(feats) for feats in features_by_layer]


def dff_joint(images_activations: np.ndarray, n_components: int = 5):
    batch_size, channels, h, w = images_activations.shape
    reshaped = images_activations.transpose(1, 0, 2, 3).reshape(channels, -1)
    reshaped[np.isnan(reshaped)] = 0
    offset = reshaped.min(axis=-1)
    reshaped -= offset[:, None]
    model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=10000)
    W = model.fit_transform(reshaped)
    H = model.components_
    concepts = W + offset[:, None]
    explanations = H.reshape(n_components, batch_size, h, w).transpose(1, 0, 2, 3)
    return concepts, explanations


def normalize_dff_maps(maps: np.ndarray) -> np.ndarray:
    return np.array([scale_cam_image(m, (maps.shape[-1], maps.shape[-2])) for m in maps])


def build_composite_visualization(factor_map: np.ndarray, colormap='hsv') -> np.ndarray:
    k, H, W = factor_map.shape
    argmax_map = np.argmax(factor_map, axis=0)
    max_vals = np.max(factor_map, axis=0)
    cmap = plt.get_cmap(colormap, k)
    color_img = cmap(argmax_map)[..., :3]
    composite = color_img * max_vals[..., None]
    return (composite * 255).astype(np.uint8)


def run_dff_pipeline(
    image_dir: str,
    content_encoder,
    target_layers: List[int],
    n_components: int = 5,
    image_size: Tuple[int, int] = (256, 256)
):
    images = load_and_preprocess_images(image_dir, size=image_size)
    print(f"Loaded {len(images)} images.")
    features_by_layer = extract_layer_features(content_encoder, images, target_layers)

    all_dff_maps = []
    concepts_per_layer = []

    for i, feats in enumerate(features_by_layer):
        print(f"Running DFF on Layer {target_layers[i]} with shape {feats.shape}...")
        feats_np = feats.numpy()
        concepts, explanations = dff_joint(feats_np, n_components=n_components)
        normalized = normalize_dff_maps(explanations)
        all_dff_maps.append(normalized)
        concepts_per_layer.append(concepts)

    return images, all_dff_maps, concepts_per_layer


def visualize_composite(images: torch.Tensor, dff_maps: List[List[np.ndarray]], layer_idx: int, image_idx: int):
    factor_map = dff_maps[layer_idx][image_idx]
    composite = build_composite_visualization(factor_map)
    img = transforms.ToPILImage()(images[image_idx])
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(composite)
    ax[1].set_title(f"DFF Composite - Layer {layer_idx+1}")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()
  
def get_channels_per_concept(W: np.ndarray, threshold: float = 0.1) -> List[List[int]]:
    return [list(np.where(W[:, i] > threshold * W[:, i].max())[0]) for i in range(W.shape[1])]

def build_channel_mask(W: np.ndarray, keep_concepts: List[int], C: int) -> np.ndarray:
    # For each channel (row), find the concept (column) with the max weight
    channel_max_concept = np.argmax(W, axis=1)  # shape: (C,)

    keep_channels = set()
    for ch, concept in enumerate(channel_max_concept):
        if concept in keep_concepts:
            keep_channels.add(ch)

    mask = np.zeros(C, dtype=np.float32)
    mask[list(keep_channels)] = 1.0
    return mask

def compute_dff_loss(content_encoder, x, yhat, dff_masks, target_layers, device='cuda'):
    content_encoder.eval()
    loss = 0.0
    with torch.no_grad():
        fx = x
        fy = yhat
        for i in range(6):
            fx = getattr(content_encoder, f'layer{i+1}')(fx)
            fy = getattr(content_encoder, f'layer{i+1}')(fy)
            if (i + 1) in target_layers:
                layer_idx = target_layers.index(i + 1)
                mask = dff_masks[layer_idx].clone().detach().to(device).view(1, -1, 1, 1)
                fx_masked = fx * mask
                fy_masked = fy * mask
                loss += F.l1_loss(fx_masked, fy_masked)
    return loss
