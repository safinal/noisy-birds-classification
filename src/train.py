from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from typing import Tuple, List

from src.config import device


def train(model: nn.Module, dataloader: torch.utils.data.DataLoader, val_loader, optimizer, criterion, num_epochs: int = 30, patience: int = 30) -> Tuple[nn.Module, List[List[float]]]:
    """
    Function to train the model.

    Input:
        model: The CNN model to be trained.
        dataloader: The DataLoader that provides the training and validation data.
        num_epochs: Number of epochs to train the model for (default is 30).

    Output:
        model: Best version of the trained model.
        scores: A list containing two lists: [training_losses, validation_losses].
    """

    scores = [[], []]



    best_val_loss = float('inf')  # Initialize with a large value
    best_model_state = None
    counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Ensure the model is in training mode

        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_loss += loss.item() * len(inputs)

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss
        scores[0].append(avg_train_loss)

        # Evaluation
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # No need to track gradients for validation
            for inputs, labels in val_loader:  # Using the global val_loader

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                running_val_loss += loss.item() * len(inputs)
                # Store predictions and true labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average validation loss for the epoch
        avg_val_loss = running_val_loss
        scores[1].append(avg_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            counter = 0  # Reset counter if the validation loss improves
        else:
            counter += 1  # Increment counter if the validation loss does not improve

        # Early stopping check
        if counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

    # After training, load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print('Best model loaded based on validation loss.')

    all_preds = []
    all_labels = []
    with torch.no_grad():  # No need to track gradients for validation
        running_loss = 0.0
        for inputs, labels in val_loader:  # Using the global test_loader
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate F1 Score
    accuracy_counter=0
    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            accuracy_counter+=1
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Use 'micro' or 'macro' depending on your needs
    print(f'F1 Score on the Validation set: {f1:.4f}')
    print(f'Accuracy on the Validation set: {accuracy_counter/len(all_labels):.4f}')
    return model, scores
