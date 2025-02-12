import os
import sys
import random
import numpy as np
import scipy
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_confusion_matrix_percentage(preds, labels, num_classes):
    """
    Compute the confusion matrix with percentage values.

    Args:
    - preds (torch.Tensor): Predicted labels (1D Tensor).
    - labels (torch.Tensor): Ground truth labels (1D Tensor).
    - num_classes (int): Number of classes.

    Returns:
    - cm_percentage (np.ndarray): Confusion matrix with percentage values.
    """

    # Initialize the confusion matrix as a tensor of zeros
    confusion_matrix = torch.zeros(num_classes, num_classes)

    # Populate the confusion matrix
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Convert counts to percentage values
    cm_percentage = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True) * 100

    # Convert to numpy for easier display/interpretation
    cm_percentage = cm_percentage.numpy()

    return cm_percentage


def get_confusion_matrix(true_labels, predicted_labels):
    # Compute confusion matrix
    conf_mat_count = confusion_matrix(true_labels, predicted_labels)
    # Normalize the confusion matrix by dividing by the sum of each row (true labels)
    conf_mat_percentage = conf_mat_count.astype('float') / conf_mat_count.sum(axis=1)[:, np.newaxis]
    # Create a percentage format for each value in the heatmap
    conf_mat_percentage_rounded = np.round(conf_mat_percentage * 100, decimals=1)
    return conf_mat_percentage_rounded


def get_predictions_and_labels(model, test_loader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            #inputs = inputs.permute(0, 2, 1)  # Reshape if necessary (for CNNs)

            # Forward pass
            outputs = model(inputs)

            # Get the predicted class
            _, preds = torch.max(outputs, 1)

            # Append predictions and true labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_labels


def save_confusion_matrix_with_percentages(confusion_matrix, class_names, image_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    # Add axis labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Percentages)")

    # Save the figure as PNG
    plt.savefig(image_name + ".png", dpi=300, bbox_inches="tight")


def plot_confusion_matrix_with_percentages(confusion_matrix, class_names, image_name):
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    # Add axis labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Percentages)")

    # Show the plot
    plt.show()


def select_random_integers(start, end, count=3):
    if end - start + 1 < count:
        raise ValueError("Range is too small for the number of unique integers requested.")

    random_integers = random.sample(range(start, end + 1), count)
    return random_integers


def get_datasets(train_transform, val_transform, test_transform, cfg):
    train_dataset = datasets.ImageFolder(cfg.train_dataset, transform=train_transform)
    val_dataset = datasets.ImageFolder(cfg.validation_dataset, transform=val_transform)
    test_dataset = datasets.ImageFolder(cfg.test_dataset, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


def get_loaders(train_dataset, val_dataset, test_dataset, cfg):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def save_finetuned_model(model, cfg):
    name = "finetuned_" + cfg.cnn + ".pth"
    torch.save(model.state_dict(), name)


def modify_model_structure(model, cfg):
    if cfg.cnn == "vgg16" or "vgg19":
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, cfg.num_classes)
    elif cfg.cnn == "densenet169" or "densenet201":
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, cfg.num_classes)
    elif cfg.cnn == "resnet50" or "resnet152":
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, cfg.num_classes)
    elif cfg.cnn == "vit_b_16" or cfg.cnn == "vit_l_32":
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, cfg.num_classes)
    elif cfg.cnn == "swin_b" or cfg.cnn == "swin_s" or cfg.cnn == "swin_t":
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, cfg.num_classes)
    else:
        sys.exit("Not defined parameter")
    return model


def get_transforms():
    train_transform = transforms.Compose([transforms.Resize((224,224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transform, val_transform, test_transform


def get_device_torch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def delete_all_files_from_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            os.remove(os.path.join(root, file))


def create_folder_structure_for_H2H(folder_path):
    os.makedirs(folder_path)
    os.makedirs(os.path.join(folder_path, "01")); os.makedirs(os.path.join(folder_path, "02"))
    os.makedirs(os.path.join(folder_path, "03")); os.makedirs(os.path.join(folder_path, "04"))
    os.makedirs(os.path.join(folder_path, "05")); os.makedirs(os.path.join(folder_path, "06"))
    os.makedirs(os.path.join(folder_path, "07")); os.makedirs(os.path.join(folder_path, "08"))
    os.makedirs(os.path.join(folder_path, "09")); os.makedirs(os.path.join(folder_path, "10"))
    os.makedirs(os.path.join(folder_path, "11")); os.makedirs(os.path.join(folder_path, "12"))


def convert_name(name):
    new_name = name[1:]
    if(len(new_name)==1):
        new_name = "0"+new_name
    else:
        pass
    return new_name


def get_number_of_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn=nn*s
        pp+=nn
    return pp


def load_torchvision_model(cfg):
    if cfg.cnn == "vgg16":
        from torchvision.models import vgg16
        model = vgg16(pretrained=True)
    elif cfg.cnn == "vgg19":
        from torchvision.models import vgg19
        model = vgg19(pretrained=True)
    elif cfg.cnn == "densenet169":
        from torchvision.models import densenet169
        model = densenet169(pretrained=True)
    elif cfg.cnn == "densenet201":
        from torchvision.models import densenet201
        model = densenet201(pretrained=True)
    elif cfg.cnn == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=True)
    elif cfg.cnn == "resnet152":
        from torchvision.models import resnet152
        model = resnet152(pretrained=True)
    elif cfg.cnn == "vit_b_16":
        from torchvision.models import vit_b_16
        model = vit_b_16(pretrained=True)
    elif cfg.cnn == "vit_l_32":
        from torchvision.models import vit_l_32
        model = vit_l_32(pretrained=True)
    elif cfg.cnn == "swin_b":
        from torchvision.models import swin_b
        model = swin_b(pretrained=True)
    elif cfg.cnn == "swin_s":
        from torchvision.models import swin_s
        model = swin_s(pretrained=True)
    elif cfg.cnn == "swin_t":
        from torchvision.models import swin_t
        model = swin_t(pretrained=True)
    else:
        sys.exit("Not defined parameter")
    return model


def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="same")


def gaussian_weights(window_size=10, sigma=1.0):
    tail = np.arange(-window_size // 2, window_size // 2 + 1)
    weights = np.exp(-tail**2 / (2 * sigma**2))
    weights /= weights.sum()
    return weights


def gaussian_moving_average(data, window_size=10, sigma=1.0):
    weights = gaussian_weights(window_size, sigma)
    return scipy.signal.convolve(data, weights, mode="same")


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    if min_val == max_val:
        return np.zeros(arr.shape)

    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr


def get_model_size_megabytes(model):
    model_size_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    model_size_mb=model_size_bytes/(1024*1024)
    return model_size_mb


if __name__ == "__main__":
    confusion_matrix = np.array([[83.3, 0.0, 14.8, 0.0, 1.9, 0.0, 0.0],
                                 [0.9, 73.4, 15.9, 9.4, 0.0, 0.0, 0.4],
                                 [0.5, 76.6, 0.0, 1.5, 0.5, 0.5, 20.4],
                                 [0.0, 2.1, 0.0, 90.5, 0.0, 0.0, 7.4],
                                 [9.5, 0.0, 0.0, 0.9, 74.8, 14.9, 0.0],
                                 [6.3, 8.7, 1.0, 1.5, 10.7, 66.0, 5.8],
                                 [5.7, 0.0, 0.0, 36.1, 3.6, 10.3, 44.3]])

    class_names = ["bend","fall","lie down","run","sitdown","standup","walk"]

    save_confusion_matrix_with_percentages(confusion_matrix, class_names, "BLSTM_wrt2humans")