import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def show_images(data, num_samples=20, cols=4):
    """ Plots some samples for the dataset"""
    plt.figure(figsize=(15,15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(num_samples/cols + 1, cols, i + 1)
        plt.imshow(img[0])

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(1, t.cpu())
    return out.reshapes(batch_size, ((1) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_alphas_cumprod_t, t, x_0/shape()
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

T = 200
data = torchvision.datasets.StanfordCars(root='.', download=True)
show_images(data)
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)



# class SinPosEmbed(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
    
#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings
        
IMG_SIZE = 64
BATCH_SIZE = 128

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root='.', download=True, transform=data_transform)

    test = torchvision.datasets.StanfordCars(root='.', download=True, transform=data_transform, split='test')

    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2)
        transforms.Lambda(lambda t: t.permute(1, 2, 0))
        transforms.Lambda(lambda t: t * 255.)
        transforms.Lambda(lambda t: t.numpy().astype(np.vint8)),
        trasnforms.ToPILImage(),
        transforms.
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
image = next(iter(dataloader))[0]

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images + 1, (idx/stepsize) + 1)
    image, noise = forward_diffusion_sample(image, t)
    show_tensor_image(image)


# class Block(nn.Module):
#     def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
#         super().__init__()
#         self.time_mlp = nn.Linear(time_emb_dim, out_ch)
#         if up:
#             self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
#             self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
#         else:
#             self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
#             self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
#         self.conv2 = nn.Conv2d(out_ch, out_ch,)
#         self.relu = nn.ReLU()
#         self.bnorm = nn.BatchNorm2d(out_ch)
#         self.pool = nn.MaxPool2d(3, stride=2)

#     def forward(self, x, t, ):
#         h = self.bnorm(self.relu(self.conv1(x)))
#         time_emb = self.relu(self.time_mlp(t))
#         time_emb = time_emb[(..., ) + (None, ) * 2]
#         h = h + time_emb
#         h = self.bnorm(self.relu(self.conv2(h)))
#         return self.transform(h)

    

            

# class SimpleUnet(nn.Module):

#     def __init__(self):
#         super().__init__()
#         image_channels = 3
#         down_channels = (64, 128, 256, 512, 1024)
#         up_channels = (1024, 512, 256, 128, 64)
#         out_dim = 1
#         time_emb_dim = 32

#         self.time_mlp = nn.Sequential(
#             SinPosEmbed(time_emb_dim),
#             nn.Linear(time_emb_dim, time_emb_dim)
#             nn.ReLU()
#         )

#         self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

#         self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \ 
#                                           time_em_dim) \
#                                     for i in range(len(down_channels) - 1))])
#         self.ups = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \ 
#                                           time_em_dim) \
#                                     for i in range(len(down_channels) - 1))])
        
#         self.output = nn.Conv2d(up_channels[-1], 1, out_dim)

#     def forward(self, x, timestep):
#         t = self.time_mlp(timestep)

#         x = self.conv0(x)

#         residual_inputs = []
#         for down in self.downs:
#             x = down(x, t)
#             residual_inputs.append(x)
#         for up in self.ups:
#             residual_x = residual_inputs.pop()
#             x = torch.cat((x, residual_x), dim=1)
#             x = up(x, t)
#         self.output(x)


# def get_loss(model, x_0, t):
#     x_noisy, noise = forward_diffusion_sample(x_0, t, device)
#     noise_pred = model(x_noisy, t)
#     return F.l1_loss(noise, noise_pred)

# @torch.no_grad
# def sample_timestep(x, t):

#     betas_t = get_index_from_list(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod_t, t, x.shape
#                                                           )






# model = SimpleUnet()
# print("Num params: ", sum((p.numel() for p in model.parameters())))