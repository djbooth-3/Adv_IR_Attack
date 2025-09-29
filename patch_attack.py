from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from pathlib import Path
from typing import List, Tuple
import cv2
import time
import pandas as pd
from random import shuffle
from typing import List, Union


batch_size = 8

# --- GPU setup ---
# Don’t mask GPUs unless you want only some of them
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # <- optional if you want only specific GPUs

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    device = torch.device("cuda:0")   # root device
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

def resize_with_padding(img: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> Image.Image:
    """Resize the image to fit within the target size, adding black padding if necessary."""
    original_width, original_height = img.size
    target_width, target_height = target_size

    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    img_resized = img.resize((new_width, new_height))
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(img_resized, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return new_img

def process_image(img: Image.Image, target_size: tuple = (640, 640)) -> torch.Tensor:
    """Resize image with padding and convert it to a tensor."""
    img_resized = img.resize(target_size)
    img_tensor = torch.tensor(
        np.array(img_resized).transpose(2, 0, 1) / 255.0, dtype=torch.float32
    ).unsqueeze(0)
    return img_tensor


def process_images_in_batches(
    imgs: List[Image.Image],
    target_size: Tuple[int,int] = (640,640),
    device: str = "cuda"
):
    """
    Convert a list of PIL images into mini-batches of shape (B,C,H,W) on device.
    """
    tensors = []
    for img in imgs:
        arr = np.array(img.resize(target_size), dtype=np.float32) / 255.0  # HWC [0,1]
        arr = np.transpose(arr, (2,0,1))  # CHW
        tensors.append(torch.from_numpy(arr))

    # Stack all into (N,C,H,W)
    full_tensor = torch.stack(tensors, dim=0)

    # Split into mini-batches
    batches = []
    for s in range(0, len(full_tensor), batch_size):
        e = min(s+batch_size, len(full_tensor))
        batches.append(full_tensor[s:e].to(device))
    return batches

def collect_bbox_coordsV2(model, images: List[torch.Tensor]):
  """Collect gradients for the tracked object with id=0 in each image using model.track."""
  model = YOLO(model)
  y1x1_bboxes = []
  x1y1x2y2_coords = []
  skipped = [] 
  for img in images:
    img = img.clone().detach().requires_grad_(True)

    # Run tracking to get boxes and ids
    results = model.track(img)
    result = results[0]
    boxes = result.boxes
    ids = result.boxes.id if hasattr(result.boxes, 'id') else None

    if ids is None:
        skipped.append[img]
        continue
      #raise RuntimeError("Tracking results do not contain object IDs.")

    # Find the box with id == 1
    id_indices = (ids == 1).nonzero(as_tuple=True)[0]
    if len(id_indices) == 0:
      # No tracked object with id=1
      print("Warning: No tracked object with id=1 in this image.")
      continue

    boxes_xyxy = boxes.xyxy
    box_coords = boxes_xyxy[id_indices[0]]
    x1, y1, x2, y2 = map(int, box_coords.cpu().numpy().tolist())
    print(x1, y1, x2, y2)
    y1x1_bboxes.append((y1+30, x1)) # Adding +30 to lower the box
    x1y1x2y2_coords.append((x1,y1,x2,y2)) 

  #print(all_bboxes)

  return y1x1_bboxes, x1y1x2y2_coords, skipped

def extract_boxes(file_path, orig_w=336, orig_h=256, new_w=640, new_h=640):
    boxes = []
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            frame, track_id, cls, x, y, w, h = parts
            track_id = int(track_id)

            if track_id == 1:  # Filter only ID 1
                x, y, w, h = map(float, (x, y, w, h))
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)

                boxes.append((x1, y1, x2, y2))
    return boxes

def visualize_and_save_attacks_all(
    original_batches: Union[torch.Tensor, List[torch.Tensor]],   # (N,C,H,W) or list of (B,C,H,W)
    attacked_batches: Union[torch.Tensor, List[torch.Tensor]],   # same
    model_path: str,
    save_dir: str = "finetuned_output_30",

    class_filter: int | None = 0,   # set None to count all classes
    show_plots: bool = False,
):
    """
    Visualize original vs attacked for ALL images across multiple batches.
    Handles (N,C,H,W) or list-of-batches.
    """

    # --- Normalize inputs to single (N,C,H,W) tensors ---
    def _to_tensor(x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        elif isinstance(x, list):
            return torch.cat([b.detach().cpu() for b in x], dim=0)
        else:
            raise TypeError("Input must be Tensor or List[Tensor]")

    originals = _to_tensor(original_batches)  # (N,C,H,W)
    advs      = _to_tensor(attacked_batches)
    assert originals.shape == advs.shape, "Originals and attacked must have same shape"

    N, C, H, W = originals.shape
    print(f"Visualizing {N} images, shape {C}x{H}x{W}")

    os.makedirs(save_dir, exist_ok=True)
    attack_dir = os.path.join(save_dir, "attacked_images")
    os.makedirs(attack_dir, exist_ok=True)

    model = YOLO(model_path)
    failed_attacks = []

    for i in range(N):
        orig_np = originals[i].permute(1,2,0).numpy()
        adv_np  = advs[i].permute(1,2,0).numpy()

        orig_u8 = np.ascontiguousarray((orig_np * 255).clip(0,255).astype(np.uint8))
        adv_u8  = np.ascontiguousarray((adv_np  * 255).clip(0,255).astype(np.uint8))

       # --- perturbation visualization (normalized to [0,1] for display) ---
        pert = (orig_np - adv_np)
        pmin, pmax = float(pert.min()), float(pert.max())
        if pmax > pmin:
            pert_vis = (pert - pmin) / (pmax - pmin)
        else:
            pert_vis = np.zeros_like(pert, dtype=np.float32)

        # run detections
        result_orig = model.predict(orig_u8, imgsz=H, verbose=False)[0]
        result_adv  = model.predict(adv_u8,  imgsz=H, verbose=False)[0]
        result_orig.orig_img = np.ascontiguousarray(result_orig.orig_img)
        result_adv.orig_img  = np.ascontiguousarray(result_adv.orig_img)

        if class_filter is None:
            num_orig = len(result_orig.boxes)
            num_adv  = len(result_adv.boxes)
        else:
            num_orig = sum(int(b.cls) == class_filter for b in result_orig.boxes)
            num_adv  = sum(int(b.cls) == class_filter for b in result_adv.boxes)

        img_orig_bgr = result_orig.plot(line_width=2)
        img_adv_bgr  = result_adv.plot(line_width=2)

        # save attacked image
        attacked_image_path = os.path.join(attack_dir, f"attacked_{i}.png")
        Image.fromarray(adv_u8).save(attacked_image_path)

        # 4-panel plot
        fig, axs = plt.subplots(1, 4, figsize=(16,4))
        axs[0].imshow(orig_u8); axs[0].set_title("Original"); axs[0].axis("off")
        axs[1].imshow(img_orig_bgr[...,::-1]); axs[1].set_title("Orig + Dets"); axs[1].axis("off")
        axs[2].imshow(img_adv_bgr[...,::-1]);  axs[2].set_title("Adv + Dets");  axs[2].axis("off")
        axs[3].imshow((pert_vis))
        axs[3].set_title("Perturbation", fontsize=12); axs[3].axis("off")

        vis_path = os.path.join(save_dir, f"visualization_{i+1}.png")
        plt.tight_layout()
        plt.savefig(vis_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        if num_adv < num_orig:
            print(f"[{i}] Attack success: {num_orig}->{num_adv}")
        else:
            failed_attacks.append(i)

        print(f"[{i}] Saved {vis_path} and {attacked_image_path}")

    print("\nFailed attacks on images:", failed_attacks)

def apply_patch(images: torch.Tensor, coords: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
    
    # Full image size
    B, C, H, W = images.shape

    #resized_patches = []

    # If single-channel IR patch, broadcast to C
    if patch.shape[1] == 1 and C > 1:
        patch_for_apply = patch.expand(-1, C, -1, -1)   # (1,C,h0,w0)
    else:
        patch_for_apply = patch

    resized_patches = []
    for (x1, y1, x2, y2) in coords.tolist():
        h, w = max(0, y2 - y1), max(0, x2 - x1)
        if h <= 0 or w <= 0:
            resized_patches.append(torch.zeros((C, H, W), device=images.device, dtype=images.dtype))
            continue

        pr = F.interpolate(patch_for_apply, size=(h, w), mode="bilinear", align_corners=False)# (1,C,h,w)
        # Pad around patch, so the patch is right over the correct box
        pad_l, pad_r = x1, W - x2
        pad_t, pad_b = y1, H - y2
        pr_full = F.pad(pr, (pad_l, pad_r, pad_t, pad_b))  # (1,C,H,W)

        resized_patches.append(pr_full.squeeze(0))

    # Stack so they are compatible with original batch
    delta = torch.stack(resized_patches, dim=0)  # (B,C,H,W)
    adv = images + delta          
    return adv

def split_coords(coords: torch.Tensor):
    """
    Split (N,4) tensor of box coords into list of mini-batches.
    """
    coords_batches = []
    for s in range(0, coords.size(0), batch_size):
        e = min(s+batch_size, coords.size(0))
        coords_batches.append(coords[s:e])
    return coords_batches


class AdversarialYOLO(YOLO):
    def __init__(self, model_path: str, epsilon: float = 1 / 255, step_size: float = 1 / 255, num_steps: int = 1):
        super().__init__(model_path)
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.model.to(device=device)
            
    def attackV10(self, x: torch.Tensor, coords: torch.Tensor,patch: torch.Tensor,amp=True):
  
        patch = patch.detach().requires_grad_(True)

        adv_images = apply_patch(x,coords,patch)
        #adv_images.requires_grad_(True)
 
        with torch.cuda.amp.autocast(enabled=amp):
            preds = self.model(adv_images)[0]
            # loss: sum of confidences
            total_loss = preds[:,4:].sum()

            max_class = preds[:,4:].max().item()
            sum_class = preds[:,4:].sum().item()
            

        if patch.grad is not None:
            patch.grad.zero_()
        self.model.zero_grad()
        total_loss.backward()
 
        g = patch.grad
        grad = g.sum(dim=1,keepdim=True)

        # update universal patch
        with torch.no_grad():
            patch.add_(-self.step_size * grad.sign())
            patch.clamp_(0,1)
            patch.requires_grad_(True)

        del preds,adv_images,grad,total_loss
        torch.cuda.empty_cache()

        with torch.no_grad():
            adv_final = apply_patch(x,coords,patch)
        return adv_final.detach(), patch.detach()


def main():

    # ---------- LOADING MODEL ----------
    #model_path = Path("/data/adversarial/models/best11.pt")        
    model_path = Path("/data/adversarial/models/yolo11l.pt")
    adversarial_model = AdversarialYOLO(model_path)
    
    # ---------- LOADING DATA ----------
    #img_dir = Path("/data/adversarial/ir_images/Seq_30/images") 
    #img_label_dir = Path("/data/adversarial/ir_images/Seq_30/Seq30-IR.txt")
    
    img_dir = Path("/home/jperez/programming/adversarial/datasets/Seq_1/IR_seq_1")
    img_label_dir = Path("/home/jperez/programming/adversarial/datasets/Seq_1/ir_annot_1.txt")
    
    fpaths_train = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(".jpg")])[:216]

    original_images = [Image.open(img) for img in fpaths_train]
    original_batch = process_images_in_batches(original_images, target_size=(640,640),device=device)
    
    # Collect bounding box coordinates from ground truth 
    gt_coords = extract_boxes(img_label_dir)
    tensor_coords = torch.tensor(gt_coords[:216])
    
    # Shuffle data 
    #data_pairs = list(zip(original_images, tensor_coords.tolist()))
    #shuffle(data_pairs)
    #shuffled_images, shuffled_coords = zip(*data_pairs)
    #batches = process_images_in_batches(list(shuffled_images[:8]),device=device)
    #coords_batches = split_coords(torch.tensor(shuffled_coords[:8]))

    # ---------- ADVERSARIAL ATTACK ----------
    N,C,H,W = original_batch[0].shape
    x1,y1,x2,y2 = tensor_coords[0].tolist()
    h0, w0 = int(y2-y1), int(x2-x1)
    patch = torch.zeros((1,C,h0,w0), device=device, requires_grad=True)
    start_time = time.time()
    iteration = 50

    for i in range(epochs):
        print(f"\n=== Iteration {i+1}/{iteration} ===")

        # Shuffle data 
        data_pairs = list(zip(original_images, tensor_coords.tolist()))
        shuffle(data_pairs)
        shuffled_images, shuffled_coords = zip(*data_pairs)
        batches = process_images_in_batches(list(shuffled_images),device=device)
        coords_batches = split_coords(torch.tensor(shuffled_coords))
                               
        for i, (bat, coord) in enumerate(zip(batches, coords_batches), 1):
            
            bat   = bat.to(device, non_blocking=True)
            coord = coord.to(device, non_blocking=True)
        
            patch = patch.detach().to(device=device).requires_grad_(True)
            # Run one batch; make attackV10 detach its returns (see below)
            adv_b, patch = adversarial_model.attackV10(bat, coord, patch)
            # Proactively drop GPU memory before the next batch
            del bat, coord, adv_b
            torch.cuda.empty_cache()
               
    end_time = time.time()
    print('Elapsed time training= {:.4f} secs'.format(end_time-start_time))

    # ---------- LOADING ATTACK DATA ----------
    attacked_all = [] # will hold CPU tensors
    for bat,coord in zip(batches,coords_batches):
        bat = bat.to(device=device)
        coord = coord.to(device=device)
        adv_b = apply_patch(bat,coord,patch)
        attacked_all.append(adv_b.cpu())
        del bat,coord,adv_b

    # ---------- VISUALIZE ATTACK DATA ----------
    attacked_images = torch.cat(attacked_all, dim=0)
    visualize_and_save_attacks_all(batches,attacked_images,str(model_path))
    total_time = time.time()
    print('Elapsed time training= {:.4f} secs'.format(total_time-start_time))

if __name__ == "__main__":
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("⚠️ CUDA not available; falling back to CPU.")
    main()
