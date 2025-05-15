from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from src.dataset import Birddataset, base_transform
from src.config import device


def pseudo_label_unlabeled_data(model, unlabeled_loader, confidence_threshold=0.9):
    model.eval()
    pseudo_labeled_data = []
    
    # Get file paths of all unlabeled images
    unlabeled_dataset = unlabeled_loader.dataset
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(unlabeled_loader):
            # Images are already tensors from the dataloader
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            
            # Only keep high-confidence predictions
            confident_mask = max_probs > confidence_threshold
            
            for i, confident in enumerate(confident_mask):
                if confident:
                    # Calculate the index in the dataset
                    dataset_idx = batch_idx * unlabeled_loader.batch_size + i
                    if dataset_idx < len(unlabeled_dataset):
                        # Get image path and load original PIL image
                        img_info = unlabeled_dataset.images[dataset_idx]
                        img_path = img_info[0]
                        class_name = img_info[1]
                        full_path = os.path.join("./Noisy_birds", class_name, img_path)
                        
                        try:
                            # Load and transform the image to ensure consistent size
                            original_img = Image.open(full_path)
                            # Apply base transform to ensure consistent size
                            original_img = base_transform(original_img)
                            # Convert back to PIL for storage
                            original_img = transforms.ToPILImage()(original_img)
                            # Store the original PIL image and predicted label
                            pseudo_labeled_data.append((original_img, predicted[i].item()))
                        except Exception as e:
                            print(f"Error loading image {full_path}: {e}")
    
    return pseudo_labeled_data

# Iterative pseudo-labeling
def iterative_pseudo_labeling(model, unlabeled_loader, initial_threshold=0.95, 
                            min_threshold=0.8, iterations=3):
    all_pseudo_data = []
    threshold = initial_threshold
    
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}, Confidence threshold: {threshold}")
        pseudo_data = pseudo_label_unlabeled_data(model, unlabeled_loader, threshold)
        print(f"Generated {len(pseudo_data)} pseudo-labeled samples")
        all_pseudo_data.extend(pseudo_data)
        
        # Gradually lower threshold
        threshold = max(min_threshold, threshold - 0.05)
    
    return all_pseudo_data