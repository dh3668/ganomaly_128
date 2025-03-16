import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionEmbedding(nn.Module):
    def __init__(self, condition_dim, H, W):
        super(ConditionEmbedding, self).__init__()
        self.linear = nn.Linear(condition_dim, H * W)
        self.H = H
        self.W = W

    def forward(self, c):
        x = self.linear(c)  # [B, H*W]
        x = x.view(-1, 1, self.H, self.W)  # [B,1,H,W]
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W)
        proj_key   = self.key_conv(x).view(B, -1, H*W)
        energy = torch.bmm(proj_query.permute(0,2,1), proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(B, -1, H*W)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(B, C, H, W)
        out = self.gamma*out + x
        return out

class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)     
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.attn1 = SelfAttention(512)

        self.conv11 = nn.Conv2d(512, latent_dim, kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.bn8(self.conv8(x)), 0.2)
        x = F.leaky_relu(self.bn9(self.conv9(x)), 0.2)
        x = F.leaky_relu(self.bn10(self.conv10(x)), 0.2)

        x_att = self.attn1(x)

        z = self.conv11(x_att)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(512)

        self.attn1 = SelfAttention(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1) # skip e2
        self.bn3 = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.deconv5 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1) # skip e1
        self.bn5 = nn.BatchNorm2d(128)

        self.deconv6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.deconv7 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.deconv8 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(32)

        self.deconv9 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        self.deconv10 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn10 = nn.BatchNorm2d(16)

        self.deconv11 = nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = F.relu(self.bn1(self.deconv1(z)))
        x = self.attn1(x)

        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))

        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))

        x = F.relu(self.bn6(self.deconv6(x)))
        x = F.relu(self.bn7(self.deconv7(x)))

        x = F.relu(self.bn8(self.deconv8(x)))
        x = F.relu(self.bn9(self.deconv9(x)))

        x = F.relu(self.bn10(self.deconv10(x)))
        x = torch.sigmoid(self.deconv11(x))

        return x


class Generator(nn.Module):
    def __init__(self, input_channels, latent_dim, condition_dim, H=224, W=224):
        super(Generator, self).__init__()
        self.condition_embedding = ConditionEmbedding(condition_dim, H, W)
        self.ae_encoder = Encoder(input_channels+1, latent_dim)
        self.ae_decoder = Decoder(latent_dim, input_channels)
        self.encoder = Encoder(input_channels+1, latent_dim)

    def forward(self, x, c):
        c_map = self.condition_embedding(c) # [B,1,H,W]
        x_cond = torch.cat([x, c_map], dim=1)
        z = self.ae_encoder(x_cond)
        x_hat = self.ae_decoder(z)
        x_hat_cond = torch.cat([x_hat, c_map], dim=1)
        z_hat = self.encoder(x_hat_cond)
        return z, x_hat, z_hat


class Discriminator(nn.Module):
    def __init__(self, input_channels, condition_dim):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(1)

        self.condition_embedding = ConditionEmbedding(condition_dim, 7, 7)

        self.conv12 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=0)

    def forward(self, x, c):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.bn8(self.conv8(x)), 0.2)
        x = F.leaky_relu(self.bn9(self.conv9(x)), 0.2)
        x = F.leaky_relu(self.bn10(self.conv10(x)), 0.2)

        feature = F.leaky_relu(self.bn11(self.conv11(x)), 0.2)      # [B,1,7,7]

        c_map = self.condition_embedding(c)
        f_cond = torch.cat([feature, c_map], dim=1)

        disc = torch.sigmoid(self.conv12(f_cond))

        return feature, disc

class ConditionalGANomaly(nn.Module):
    def __init__(self, input_channels, latent_dim, condition_dim, H=224, W=224):
        super(ConditionalGANomaly, self).__init__()
        self.generator = Generator(input_channels, latent_dim, condition_dim, H, W)
        self.discriminator = Discriminator(input_channels, condition_dim)

    def forward(self, x, c):
        z, x_hat, z_hat = self.generator(x, c)
        feature_real, disc_real = self.discriminator(x, c)
        feature_fake, disc_fake = self.discriminator(x_hat, c)
        return z, x_hat, z_hat, feature_real, feature_fake, disc_real, disc_fake
