class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()

        # Pre-trained ResNet50 encoder (modified to accept 1 channel input)
        self.encoder_resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.encoder_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_resnet = nn.Sequential(*list(self.encoder_resnet.children())[:-2])  # Remove the fully connected layer

        # Pre-trained DenseNet121 encoder (modified to accept 1 channel input)
        self.encoder_densenet = models.densenet121(weights='IMAGENET1K_V1')
        self.encoder_densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_densenet = nn.Sequential(*list(self.encoder_densenet.children())[:-1])  # Use all layers except the classifier

        # Pooling layer to downsample DenseNet output to 7x7
        self.downsample_densenet = nn.AdaptiveAvgPool2d((7, 7))

        # 1x1 Convolution to reduce ResNet output to 1024 channels for averaging
        self.resnet_conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1)

        # 1x1 Convolution to match DenseNet's output to 1024 channels if needed (keeping symmetry)
        self.densenet_conv1x1 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Fusion Blocks
        self.fusion_block1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fusion_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fusion_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fusion_block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder Blocks (same as in the concatenation model)
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=3, padding=1),
            nn.Tanh(),  # Use Tanh to match output range [-1, 1]
            nn.Upsample(scale_factor=2)  # Upsample to 224 x 224
        )

    def forward(self, x):
        # Encoder
        x_resnet = self.encoder_resnet(x)  # ResNet output
        x_resnet = self.resnet_conv1x1(x_resnet)  # Reduce ResNet output to 1024 channels
        x_densenet = self.encoder_densenet(x)  # DenseNet output
        x_densenet = self.downsample_densenet(x_densenet)  # Downsample DenseNet output
        x_densenet = self.densenet_conv1x1(x_densenet)  # Match DenseNet output to 1024 channels

        # Fusion by averaging
        fb1_input = (x_resnet + x_densenet) / 2  # Average fusion instead of concatenation
        fb1_output = self.fusion_block1(fb1_input)

        fb2_input = fb1_output  # Use previous output only (no need to concatenate)
        fb2_output = self.fusion_block2(fb2_input)

        fb3_input = fb2_output  # Use previous output only (no need to concatenate)
        fb3_output = self.fusion_block3(fb3_input)

        fb4_input = fb3_output  # Use previous output only (no need to concatenate)
        fb4_output = self.fusion_block4(fb4_input)

        # Decoder
        db1_output = self.decoder_block1(fb4_output)
        db2_output = self.decoder_block2(db1_output)
        db3_output = self.decoder_block3(db2_output)
        db4_output = self.decoder_block4(db3_output)

        output = self.decoder_block5(db4_output)

        return output


# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            images = batch[0].to(device)  # Move images to device

            # Convert RGB to L and ab channels
            L, ab_target = rgb_to_lab(images)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass: model predicts 'a' and 'b' channels from 'L' channel
            ab_pred = model(L)  # The input is now just the L channel

            # Compute loss between predicted ab channels and ground truth ab channels
            loss = criterion(ab_pred, ab_target)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Print loss every 500 batches
            if batch_idx % 500 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Example usage
if __name__ == "__main__":
    # Load the CIFAR10 dataset
    batch_size = 8
    train_loader, test_loader = load_cifar10_dataset(batch_size=batch_size)

    # Initialize the model
    model = ColorizationModel().to(device)  # Move model to GPU if available

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Loss for comparing predicted ab with ground truth ab
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 2
    train(model, train_loader, criterion, optimizer, num_epochs)
 