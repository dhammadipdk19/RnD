# Function to convert RGB images to L, a, and b channels in Lab color space
def rgb_to_lab(images):
    l_channels = []
    ab_channels = []
    for img in images:
        img = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and move to CPU
        lab_image = rgb2lab(img)  # Convert to CIE-Lab

        # Normalize L, a, and b channels
        L_channel = lab_image[:, :, 0] / 100.0  # Normalize L channel to [0, 1]
        a_channel = (lab_image[:, :, 1] + 128) / 255.0  # Normalize a channel to [0, 1]
        b_channel = (lab_image[:, :, 2] + 128) / 255.0  # Normalize b channel to [0, 1]

        l_channels.append(L_channel)
        ab_channels.append(np.stack((a_channel, b_channel), axis=-1))  # Stack a and b

    # Convert to PyTorch tensors
    L = torch.tensor(np.stack(l_channels), dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, H, W)
    ab = torch.tensor(np.stack(ab_channels), dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (N, 2, H, W)
    return L, ab

# CIFAR10 Dataset Loader
def load_cifar10_dataset(batch_size=8, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet/DenseNet input size
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    ])

    # Load the CIFAR-10 dataset
    train_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# Function to convert predicted 'L' and 'ab' channels to an RGB image
def lab_to_rgb(L, ab):
    L = L * 100  # Denormalize L channel from [0, 1] to [0, 100]
    ab = ab * 255 - 128  # Denormalize ab channels from [0, 1] to [-128, 128]

    lab = np.concatenate((L, ab), axis=2)  # Combine L and ab channels
    rgb_img = lab2rgb(lab)  # Convert Lab to RGB
    return rgb_img

