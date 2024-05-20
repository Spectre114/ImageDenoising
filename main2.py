# denoising_with_fastapi.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import io
import os
import torchvision as tv
import torch.utils.data as td
import nntools as nt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

dataset_root_dir = "dataset/BSDS300/images"


class NoisyBSDSDataset(td.Dataset):
    def __init__(self, root_dir, mode="train", image_size=(180, 180), sigma=30):
        super(NoisyBSDSDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})".format(
            self.mode, self.image_size, self.sigma
        )

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert("RGB")
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])

        clean = clean.crop(
            [i, j, i + self.image_size[0], j + self.image_size[1]])
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        clean = transform(clean)

        noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)
        return noisy, clean


class NNRegressor(nt.NeuralNetwork):
    def __init__(self):
        super(NNRegressor, self).__init__()
        self.mse = nn.MSELoss()

    def criterion(self, y, d):
        return self.mse(y, d)


class DnCNN(NNRegressor):
    def __init__(self, D, C=64):
        super(DnCNN, self).__init__()
        self.D = D

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity="relu")
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        for i in range(D):
            h = F.relu(self.bn[i](self.conv[i + 1](h)))
        y = self.conv[D + 1](h) + x
        return y


class DenoisingStatsManager(nt.StatsManager):
    def __init__(self):
        super(DenoisingStatsManager, self).__init__()

    def init(self):
        super(DenoisingStatsManager, self).init()
        self.running_psnr = 0

    def accumulate(self, loss, x, y, d):
        super(DenoisingStatsManager, self).accumulate(loss, x, y, d)
        n = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        self.running_psnr += 10 * torch.log10(4 * n / (torch.norm(y - d) ** 2))

    def summarize(self):
        loss = super(DenoisingStatsManager, self).summarize()
        psnr = self.running_psnr / self.number_update
        return {"loss": loss, "PSNR": psnr.cpu()}


def myimshow(tensor_image, ax):
    image = tensor_image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(image)
    ax.axis("off")


def denoise_custom_image(model, custom_image):
    transform = transforms.Compose([transforms.ToTensor()])
    custom_image_tensor = transform(custom_image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        noisy_image = custom_image_tensor
        denoised_image = model(noisy_image)
    images = [
        custom_image_tensor.squeeze(0),
        denoised_image.squeeze(0),
    ]
    titles = ["Uploaded", "denoise"]
    fig, axes = plt.subplots(ncols=2, figsize=(9, 5),
                             sharex="all", sharey="all")
    for i, img in enumerate(images):
        myimshow(img, ax=axes[i])
        axes[i].set_title(titles[i])
    plt.show()


# Model setup
lr = 1e-3
net = DnCNN(6).to(device)
adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = DenoisingStatsManager()
exp1 = nt.Experiment(
    net,
    NoisyBSDSDataset(dataset_root_dir),
    NoisyBSDSDataset(dataset_root_dir, mode="test", image_size=(320, 320)),
    adam,
    stats_manager,
    batch_size=4,
    output_dir="checkpoints/denoising1",
    perform_validation_during_training=True,
)
model = torch.load(
    "checkpoints/denoising1/checkpoint.pth.tar",
    map_location="cpu",
)

# FastAPI code
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def main():
    with open("index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/denoise/")
async def denoise_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        denoise_custom_image(exp1.net.to(device), image)
        return JSONResponse(status_code=200, content={"message": "Image processed successfully"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"An error occurred: {str(e)}"})

# To run the server, use the command:
# uvicorn denoising_with_fastapi:app --reload
