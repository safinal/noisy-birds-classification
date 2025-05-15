import torch

from src.model import TeacherModel, StudentModel
from src.train import train
from src.evaluate import plot_losses
from src.dataset import create_train_val_unlabeled_datasets_and_loaders, create_combined_dataset_and_loader
from src.config import *
from src.utils import check_model_size
from src.pseudo_label import iterative_pseudo_labeling


train_dataset, train_loader, val_dataset, val_loader, unlabeled_dataset, unlabeled_loader = create_train_val_unlabeled_datasets_and_loaders()

teacher_model = TeacherModel()
teacher_model.to(device)
output = teacher_model(torch.randn(128,3,128,128).cuda())
assert output.shape==(128,4), "The output of your model does not have correct dimmensions"

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=teacher_lr, weight_decay=teacher_weight_decay)


teacher_model, training_scores = train(teacher_model, train_loader, val_loader, optimizer, criterion, num_epochs=teacher_num_epochs, patience=teacher_patience)
plot_losses(training_scores)


# Step 1: Create large teacher model for pseudo-labeling
teacher_model.to(device)

# Step 2: Generate pseudo-labels
print("Generating pseudo-labels...")
pseudo_labeled_data = iterative_pseudo_labeling(teacher_model, unlabeled_loader, initial_threshold=initial_threshold, min_threshold=min_threshold, iterations=iterations)

# Step 3: Create student model (smaller, within 70MB)
student_model = StudentModel()  # This should use smaller DINOv2
student_model.to(device)

# Step 4: Create combined dataset
combined_dataset, combined_loader = create_combined_dataset_and_loader(train_dataset, pseudo_labeled_data)

# Step 5: Train student model
print(f"Training on {len(train_dataset)} labeled + {len(pseudo_labeled_data)} pseudo-labeled samples")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(student_model.parameters(), lr=student_lr, weight_decay=student_weight_decay)

# Use lower learning rate for fine-tuning
student_model, scores = train(student_model, combined_loader, val_loader, optimizer, criterion, num_epochs=student_num_epochs, patience=student_patience)
plot_losses(scores)

# Check if model is within 70MB limit
model_size = check_model_size(student_model)
assert model_size <= 70, f"Model size {model_size:.2f} MB exceeds 70MB limit"

# model_save_path = "model.pth"
# torch.save(model.cpu().state_dict(), model_save_path)