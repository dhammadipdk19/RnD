def visualize_colorization(model, test_loader, num_images=4):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)  # Get the original RGB images
            print(f"Original RGB Images Shape: {images.shape}")  # Debugging statement

            # Get the L and ab channels
            L, ab = rgb_to_lab(images)
            print(f"L Channel Shape: {L.shape}, ab Shape: {ab.shape}")  # Debugging statement

            # L already has shape (N, 1, H, W)
            # No need to change L

            # Predict ab channels from L
            ab_pred = model(L)  # Forward pass with L channel
            print(f"Predicted ab Shape: {ab_pred.shape}")  # Debugging statement

            # Convert the first few images to numpy for visualization
            for i in range(num_images):
                # Prepare the original RGB image
                original_rgb = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
                print(f"Original RGB Image Shape: {original_rgb.shape}")  # Debugging statement

                # Prepare the L channel
                L_cpu = L[i].cpu().numpy().squeeze() * 100  # Squeeze to (H, W) and denormalize to [0, 100]
                print(f"L Channel Shape After Squeeze: {L_cpu.shape}")  # Debugging statement

                # Prepare the predicted ab channel
                ab_cpu_pred = ab_pred[i].cpu().numpy() * 255 - 128  # Denormalize predicted ab channels
                print(f"Predicted ab Channel Shape: {ab_cpu_pred.shape}")  # Debugging statement

                # Reshape ab_cpu_pred to (224, 224, 2)
                ab_cpu_pred_reshaped = ab_cpu_pred.transpose(1, 2, 0)  # Change shape to (224, 224, 2)
                print(f"Reshaped Predicted ab Channel Shape: {ab_cpu_pred_reshaped.shape}")  # Debugging statement

                # Create LAB image and convert to RGB
                lab_image = np.concatenate((L_cpu[:, :, np.newaxis], ab_cpu_pred_reshaped), axis=2)  # Concatenate L and ab
                print(f"LAB Image Shape: {lab_image.shape}")  # Debugging statement

                rgb_colorized = lab2rgb(lab_image)  # Convert LAB back to RGB

                # Plot the images
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 3, 1)
                plt.title('Original RGB Image')
                plt.imshow(original_rgb.astype(np.uint8))
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title('L Channel')
                plt.imshow(L_cpu, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title('Colorized Image')
                plt.imshow(rgb_colorized)
                plt.axis('off')

                plt.show()

            break  # Remove this if you want to visualize more than one batch

# Testing and visualization
if __name__ == "__main__":
    # Visualize the model's colorization results with original images
    visualize_colorization(model, test_loader, num_images=4)