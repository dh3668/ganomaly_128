import torch
from torch.optim import Adam
import torch.nn.functional as F
from model_modified import ConditionalGANomaly
from dataset import get_dataloader
from config import CONFIG
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def save_reconstruction(model, dataloader, device, output_dir="reconstructions", condition_dim=8, epoch=0):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            c = F.one_hot(labels, num_classes=condition_dim).float().to(device)

            z, reconstructed_images, z_hat, feature_real, feature_fake, disc_real, disc_fake = model(images, c)

            batch_size = images.size(0)
            fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
            if batch_size == 1:
                axes = [axes]

            for i in range(batch_size):
                axes[i][0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
                axes[i][0].set_title("Original")
                axes[i][0].axis("off")

                axes[i][1].imshow(reconstructed_images[i].permute(1, 2, 0).cpu().numpy())
                axes[i][1].set_title("Reconstructed")
                axes[i][1].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"epoch_{epoch+1}_batch_{idx + 1}.png"))
            plt.close()
            if idx >= 0:
                break

def train(config):
    device = config["device"]
    input_channels = config["input_channels"]
    latent_dim = config["latent_dim"]
    condition_dim = config["condition_dim"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    dataset_path = config["dataset_path"]
    image_size = config["image_size"]  # (H, W) 예: (224,224)

    os.makedirs("logs_modified", exist_ok=True)

    train_loader, valid_loader = get_dataloader(dataset_path, batch_size, image_size)

    model = ConditionalGANomaly(input_channels, latent_dim, condition_dim, H=image_size[0], W=image_size[1]).to(device)
    optimizer_G = Adam(model.generator.parameters(), lr=learning_rate)
    optimizer_D = Adam(model.discriminator.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []
    best_valid_loss = float("inf")
    best_model_path = "logs_modified/best_model.pth"

    for epoch in range(epochs):
        model.train()
        image_recon_loss_total = 0
        latent_recon_loss_total = 0
        feature_loss_total = 0
        g_loss_total = 0
        d_loss_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            c = F.one_hot(labels, num_classes=condition_dim).float().to(device)

            #================= Generator Training =================#
            model.generator.train()
            model.discriminator.eval()

            z, x_hat, z_hat = model.generator(images, c)
            feature_real, disc_real = model.discriminator(images, c)
            feature_fake, disc_fake = model.discriminator(x_hat, c)

            optimizer_G.zero_grad()

            x_recon_loss = F.l1_loss(x_hat, images, reduction="mean")
            z_recon_loss = F.mse_loss(z_hat, z, reduction="mean")
            feature_loss = F.mse_loss(feature_fake, feature_real, reduction="mean")

            g_loss = 50*x_recon_loss + z_recon_loss + feature_loss
            g_loss.backward()
            optimizer_G.step()

            #=============== Discriminator Training ===============#
            model.generator.eval()
            model.discriminator.train()

            with torch.no_grad():
                z, x_hat, z_hat = model.generator(images, c)
            feature_real, disc_real = model.discriminator(images, c)
            feature_fake, disc_fake = model.discriminator(x_hat, c)

            optimizer_D.zero_grad()

            real_labels = torch.ones(images.size(0), 1, 1, 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1, 1, 1).to(device)

            d_loss_real = F.binary_cross_entropy(disc_real, real_labels)
            d_loss_fake = F.binary_cross_entropy(disc_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            image_recon_loss_total += x_recon_loss.item()
            latent_recon_loss_total += z_recon_loss.item()
            feature_loss_total += feature_loss.item()
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

            avg_image_recon_loss = image_recon_loss_total / len(train_loader)
            avg_latent_recon_loss = latent_recon_loss_total / len(train_loader)
            avg_feature_loss = feature_loss_total / len(train_loader)
            avg_g_loss = g_loss_total / len(train_loader)
            avg_d_loss = d_loss_total / len(train_loader)

            progress_bar.set_postfix({
                "Image Recon Loss": avg_image_recon_loss,
                "Latent Recon Loss": avg_latent_recon_loss,
                "Feature Loss": avg_feature_loss,
                "G Loss": avg_g_loss,
                "D Loss": avg_d_loss
            })

        train_losses.append(50*avg_image_recon_loss + avg_latent_recon_loss)

        # Validation
        model.eval()
        valid_image_recon_loss_total = 0
        valid_latent_recon_loss_total = 0
        valid_feature_loss_total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                c = F.one_hot(labels, num_classes=condition_dim).float().to(device)

                z, x_hat, z_hat, feature_real, feature_fake, disc_real, disc_fake = model(images, c)
                valid_image_recon_loss = F.l1_loss(x_hat, images, reduction="mean")
                valid_image_recon_loss_total += valid_image_recon_loss.item()

                valid_latent_recon_loss = F.mse_loss(z_hat, z, reduction="mean")
                valid_latent_recon_loss_total += valid_latent_recon_loss.item()

                valid_feature_loss = F.mse_loss(feature_fake, feature_real, reduction="mean")
                valid_feature_loss_total += valid_feature_loss.item()

        avg_valid_image_recon_loss = valid_image_recon_loss_total / len(valid_loader)
        avg_valid_latent_recon_loss = valid_latent_recon_loss_total / len(valid_loader)
        avg_valid_feature_loss = valid_feature_loss_total / len(valid_loader)

        valid_loss = 50*avg_valid_image_recon_loss + avg_valid_latent_recon_loss
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch + 1}/{epochs}  Valid Image Recon Loss: {avg_valid_image_recon_loss:.5f}    Valid Latent Recon Loss: {avg_valid_latent_recon_loss:.5f}    Valid Feature Loss: {avg_valid_feature_loss:.5f}")
        print("")

        save_reconstruction(model, valid_loader, device, output_dir="reconstructions_modified", condition_dim=condition_dim, epoch=epoch)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} with loss {best_valid_loss:.5f}")

    # Save final model
    final_model_path = config["model_save_path"]
    torch.save(model.state_dict(), final_model_path)
    print("Final model saved at", final_model_path)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Valid Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig("logs/loss_plot.png")
    plt.show()
