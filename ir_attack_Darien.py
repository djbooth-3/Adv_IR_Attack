from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import os
from pathlib import Path
from typing import List, Tuple
import cv2
import time
import pandas as pd

# --- GPU setup ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def visualize_and_save_attacks(
    original_images: List[Image.Image],
    attacked_images: List[torch.Tensor],
    model_path: str,
    #noise,
    save_dir: str = "output_db"):

    """Visualize and save results of adversarial attacks."""
    os.makedirs(save_dir, exist_ok=True)
    attack_dir = os.path.join(save_dir, "attacked_images")
    os.makedirs(attack_dir, exist_ok=True)

    model = YOLO(model_path)
    #noise_np = noise.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255
    #noise_np = noise_np.astype(np.uint8)

    #attacked_image_path = os.path.join(attack_dir, f"att_noise.png")
    #Image.fromarray(noise_np).save(attacked_image_path)

    failed_attacks = []

    for i, (orig, adv) in enumerate(zip(original_images, attacked_images)):
        print(np.array(orig.resize((640,640))).shape, adv.shape)
        perturbation = np.array(orig.resize((640,640)))/255  - adv.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        print(np.min(perturbation),np.max(perturbation))
        perturbation -= np.min(perturbation)
        perturbation /= np.max(perturbation)
        print(np.min(perturbation),np.max(perturbation))
        orig_resized = resize_with_padding(orig, target_size=(640, 640))
        orig_resized = orig.resize((640,640))

        result_orig = model.predict(orig_resized, imgsz=(640, 640))[0]
        boxes_orig = [box for box in result_orig.boxes if int(box.cls) == 0]
        num_person_orig = len(boxes_orig)

        result_adv = model.predict(adv, imgsz=(640, 640))[0]
        boxes_adv = [box for box in result_adv.boxes if int(box.cls) == 0]
        num_person_adv = len(boxes_adv)

        img_orig = result_orig.plot(line_width=2)
        img_adv = result_adv.plot(line_width=2)

        adv_img_np = adv.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255
        adv_img_np = adv_img_np.astype(np.uint8)

        attacked_image_path = os.path.join(attack_dir, f"attacked_{i}.png")
        Image.fromarray(adv_img_np).save(attacked_image_path)

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(orig_resized)
        axs[0].set_title("Original Image", fontsize=12)
        axs[0].axis("off")

        axs[1].imshow(img_orig[..., ::-1])
        axs[1].set_title("Original Image with Detection", fontsize=12)
        axs[1].axis("off")

        axs[2].imshow(img_adv)
        axs[2].set_title("Attacked Image with Detection", fontsize=12)
        axs[2].axis("off")

        axs[3].imshow(perturbation)
        axs[3].set_title("Perturbation", fontsize=12)
        axs[3].axis("off")

        visualization_path = os.path.join(save_dir, f"visualization_{i+1}.png")
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.show()
        #plt.close()

        if num_person_adv < num_person_orig:
            print(f"Attack successful: {num_person_orig} -> {num_person_adv} detections")
        else:
            failed_attacks.append(i+1)

        print(f"Visualization saved to {visualization_path}")
        print(f"Attacked image saved to {attacked_image_path}")

    print(f"\nFailed attacks on images: {failed_attacks}")

class AdversarialYOLO(YOLO):
    def __init__(self, model_path: str, epsilon: float = 2 / 255, step_size: float = 2 / 255, num_steps: int = 20):
        super().__init__(model_path)
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
    

    # Perform attack on bounding box with specified area
    def attackV9(self, img: List[torch.Tensor], x1: int, y1: int, x2: int, y2: int, area: float):

        img = img.clone().detach().requires_grad_(True).to(self.device)

        orig_w = x2 - x1
        orig_h = y2 - y1

        scale = area 
        new_w = orig_w * scale
        new_h = orig_h * scale

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        ax1 = int(cx - new_w / 2)
        ay1 = int(cy - new_h / 2)
        ax2 = int(cx + new_w / 2)
        ay2 = int(cy + new_h / 2)

        pred = self.model(img)[0]

        for i in range(200):
            pred = self.model(img)[0] # Shape: [Batches,N,Grid (8400)]

            stop_flag = False
            pred_boxes = pred.detach().cpu().numpy()[0] 
            target_box = np.array([x1, y1, x2, y2])

            xc, yc, w, h = pred_boxes[0], pred_boxes[1], pred_boxes[2], pred_boxes[3]
            conf = np.max(pred_boxes[4:], axis=0)
            conf_mask = conf > 0.25

            # Convert 
            i_x1 = np.maximum(x1, xc - w/2)
            i_y1 = np.maximum(y1, yc - h/2)
            i_x2 = np.minimum(x2, xc + w/2)
            i_y2 = np.minimum(y2, yc + h/2)

            i_w = np.maximum(i_x2 - i_x1, 0.0)
            i_h = np.maximum(i_y2 - i_y1, 0.0)
            i_area = i_w * i_h

            bA_area = (x2 - x1) * (y2 - y1)
            bB_area = w * h
            u_area = bA_area + bB_area - i_area

            iou_scores = np.where(u_area > 0, i_area / u_area, 0.0)

            # Apply IoU mask
            mask = conf_mask & (iou_scores > 0.40)
            stop_flag = not np.any(mask)

            # Compute loss 
            total_loss = pred[:,4:].sum()
            max_class = pred[:,4:].max().item()
            sum_class = pred[:,4:].sum().item()
            print(f'iteration {i} max class confidence: {max_class:6.4f} sum class confidence: {sum_class:6.4f}')
            if stop_flag: # Exit the loop if no boxes detected
                print(f'Exiting after {i} iterations')
                break

            # Backpropagate
            self.model.zero_grad()
            total_loss.backward()

            # Apply perturbation
            with torch.no_grad():
                im_grad = torch.sum(img.grad, dim=1, keepdim=True) # Make perturbation the same for all channels
                #grad_max =  torch.max(torch.abs(im_grad))
                #grad_max =  1
                perturbation = (1/255) * im_grad.sign()
                img[:, :,ay1:ay2, ax1:ax2] = torch.clamp(img[:, :,ay1:ay2, ax1:ax2] - perturbation[:, :,ay1:ay2, ax1:ax2], 0, 1).detach().requires_grad_(True)

        return img

def main():
    # Load Ultralytics model
    #model_path = Path("/data/adversarial/models/best11.pt")                 # <-- your weight
    model_path = Path("/data/adversarial/models/yolo11l.pt")
    
    
    adversarial_model = AdversarialYOLO(model_path)
    
    # ---------- LOADING DATA ----------
    img_dir = Path("/data/adversarial/ir_images/Seq_30/images")
    #img_dir = Path("/home/jperez/programming/adversarial/datasets/ir_image/images/val")
    #lbl_dir = Path("/home/jperez/programming/adversarial/datasets/ir_image/labels/val")

    img_label_dir = Path("/data/adversarial/ir_images/Seq_30/Seq30-IR.txt")

    fpaths_train = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(".jpg")])[:25]  #[:235]

    original_images = [Image.open(img) for img in fpaths_train]

    processed_images = [process_image(img) for img in original_images]

    # Collect bounding box coordinates from ground truth
    gt_coords = extract_boxes(img_label_dir)[:25]
    print(len(gt_coords))

    #y1x1_coords, x1y1x2y2_coords, skipped = collect_bbox_coordsV2(model_path,processed_images)
    #attacked_images = adversarial_model.attackV8(processed_images, x1y1x2y2_coords)
    attacked_images = [
        adversarial_model.attackV9(img, x1, y1, x2, y2, area = 1.0)
        for img, (x1, y1, x2, y2) in zip(processed_images, gt_coords)
    ]
    visualize_and_save_attacks(original_images,attacked_images,model_path)

if __name__ == "__main__":
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("⚠️ CUDA not available; falling back to CPU.")
    main()
