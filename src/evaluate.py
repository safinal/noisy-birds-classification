import matplotlib.pyplot as plt

def plot_losses(scores):
    """
    Plot the training and validation losses.

    Parameters:
    - scores: A list containing two lists [training_losses, validation_losses]
    """
    train_losses = scores[0]
    val_losses = scores[1]

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
