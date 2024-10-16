# Function to evaluate the model
def evaluate_model(model, test_loader, lpips_loss_fn):
    model.eval()  # Set the model to evaluation mode
    mse_loss_fn = torch.nn.MSELoss()  # MSE loss for evaluation
    total_loss, total_mse, total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0, 0.0, 0.0
    num_images = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in test_loader:
            images = batch[0].to(device)  # Get the original RGB images
            print(f"Original RGB Images Shape: {images.shape}")  # Debugging statement

            # Convert RGB to L and ab channels
            L, ab = rgb_to_lab(images)
            print(f"L Channel Shape: {L.shape}, ab Shape: {ab.shape}")  # Debugging statement

            # Add channel dimension to L (Nx1xHxW) if not already present
            if L.ndim == 3:  # Only unsqueeze if L doesn't already have the channel dimension
                L = L.unsqueeze(1)
            print(f"L Channel Shape After Unsqueeze: {L.shape}")  # Debugging statement

            # Predict ab channels from L
            ab_pred = model(L)
            print(f"Predicted ab Shape: {ab_pred.shape}")  # Debugging statement

            # Resize ground truth ab to match ab_pred size
            ab_resized = F.interpolate(ab, size=ab_pred.shape[2:], mode='bilinear', align_corners=False)
            print(f"Resized Ground Truth ab Shape: {ab_resized.shape}")  # Debugging statement

            # Compute loss (MSE between predicted and ground truth ab channels)
            loss = mse_loss_fn(ab_pred, ab_resized)
            total_loss += loss.item()

            # Compute MSE for evaluation
            mse = F.mse_loss(ab_pred, ab_resized)
            total_mse += mse.item()

            # Convert to numpy for PSNR, SSIM, and LPIPS evaluation
            L_cpu = L.squeeze(1).cpu().numpy() * 100  # Denormalize L from [0, 1] to [0, 100]
            ab_cpu_gt = ab.cpu().numpy() * 255 - 128  # Denormalize ab from [0, 1] to [-128, 128]
            ab_cpu_pred = ab_pred.cpu().numpy() * 255 - 128  # Denormalize predicted ab
            print(f"L_cpu Shape: {L_cpu.shape}, ab_cpu_gt Shape: {ab_cpu_gt.shape}, ab_cpu_pred Shape: {ab_cpu_pred.shape}")  # Debugging statement

            for i in range(L.shape[0]):  # Loop over batch images
                # Reconstruct ground truth and predicted LAB images
                lab_gt = np.concatenate([L_cpu[i][:, :, np.newaxis], ab_cpu_gt[i].transpose(1, 2, 0)], axis=2)
                lab_pred = np.concatenate([L_cpu[i][:, :, np.newaxis], ab_cpu_pred[i].transpose(1, 2, 0)], axis=2)

                print(f"LAB GT Shape: {lab_gt.shape}, LAB Pred Shape: {lab_pred.shape}")  # Debugging statement

                # Convert Lab to RGB
                rgb_gt = lab2rgb(lab_gt)
                rgb_pred = lab2rgb(lab_pred)

                # Compute PSNR
                psnr_value = psnr(torch.tensor(rgb_pred).permute(2, 0, 1).unsqueeze(0),
                                  torch.tensor(rgb_gt).permute(2, 0, 1).unsqueeze(0), data_range=1.0)
                total_psnr += psnr_value.item()

                # Compute SSIM
                ssim_value = ssim(torch.tensor(rgb_pred).permute(2, 0, 1).unsqueeze(0),
                                  torch.tensor(rgb_gt).permute(2, 0, 1).unsqueeze(0), data_range=1.0)
                total_ssim += ssim_value.item()

                # Compute LPIPS
                lpips_value = lpips_loss_fn(torch.tensor(rgb_pred).permute(2, 0, 1).unsqueeze(0).to(device),
                                            torch.tensor(rgb_gt).permute(2, 0, 1).unsqueeze(0).to(device))
                total_lpips += lpips_value.item()

                num_images += 1  # Track number of evaluated images

    # Calculate averages
    avg_loss = total_loss / len(test_loader)
    avg_mse = total_mse / num_images
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_lpips = total_lpips / num_images

    return avg_loss, avg_mse, avg_psnr, avg_ssim, avg_lpips


if __name__ == "__main__":
    # Evaluate the model and print results
    avg_loss, avg_mse, avg_psnr, avg_ssim, avg_lpips = evaluate_model(model, test_loader, lpips_loss_fn)

    # Print the evaluation results
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
  