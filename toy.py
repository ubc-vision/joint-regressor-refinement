'''
current status: overfitting with linearized sampler can generate good gradients,
but overfitting with bilinear sample can generate good error values but wrong gradients.
This is counter my past experience, so we want to do a simple toy example verification.

Wei Jiang
2021.01.22
'''
import os.path as osp
import math
import colorsys

import imageio
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm

from utils import utils
from warp.sampling_helper import DifferentiableImageSampler

IMG_SIZE = 1000
CENTER = [500, 500]
X_MIN = 400
X_MAX = 600
Y_MIN = 400
Y_MAX = 600
CROP_SIZE = 256
HALF_CROP_SIZE = 128


class ToyDataset(data.Dataset):
    def __init__(self, img):
        self.img = img

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        x = torch.randint(low=X_MIN, high=X_MAX, size=(1,))[0]
        y = torch.randint(low=Y_MIN, high=Y_MAX, size=(1,))[0]
        crop = self.img[y - HALF_CROP_SIZE:y + HALF_CROP_SIZE, x - HALF_CROP_SIZE:x + HALF_CROP_SIZE]
        crop = utils.np_img_to_torch_img(crop) / 255.0
        error = torch.tensor([torch.norm(torch.tensor([y, x]).float() - torch.tensor(CENTER).float(), keepdim=True)]) / torch.tensor([math.sqrt(200**2)])
        return crop.float(), error.float()


class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 12, 3),
            nn.AvgPool2d(2),
            nn.GroupNorm(12, 12),
            nn.ReLU(),

            nn.Conv2d(12, 24, 3),
            nn.AvgPool2d(2),
            nn.GroupNorm(24, 24),
            nn.ReLU(),

            nn.Conv2d(24, 48, 3),
            nn.AvgPool2d(2),
            nn.GroupNorm(48, 48),
            nn.ReLU(),

            nn.Conv2d(48, 64, 3),
            nn.AvgPool2d(2),
            nn.GroupNorm(64, 64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3),
            nn.AvgPool2d(2),
            nn.GroupNorm(128, 128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3),
            nn.AvgPool2d(2),
            nn.GroupNorm(256, 256),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


def train(net, loader, optimizer, iterations=1000):
    i = 0
    all_loss = []
    while True:
        for batch_idx, batch in tqdm.tqdm(
                enumerate(loader), total=len(loader),
                desc=f'iteration={i}...', ncols=80,
                leave=False):
            i += 1
            if i >= iterations:
                break
            x = batch[0].cuda()
            y = batch[1].cuda()
            y_ = net(x)
            loss = torch.nn.functional.mse_loss(y_, y)
            loss.backward()
            optimizer.step()
            # print(loss)
            all_loss.append(loss.item())
        if i >= iterations:
            break
    plt.plot(all_loss)
    plt.title('Training loss')
    plt.show()
    # exec(utils.embed_breakpoint())


def vec2mat_for_trans_scale(vec):
    assert vec.shape[1] == 4
    _len = vec.shape[0]
    O = torch.zeros([_len], dtype=torch.float32, device=vec.device)
    I = torch.ones([_len], dtype=torch.float32, device=vec.device)

    p1, p2, p3, p4 = torch.unbind(vec, dim=1)
    sx = p1
    sy = p2
    dx = p3
    dy = p4
    S = torch.stack([torch.stack([sx, O, O], dim=-1),
                     torch.stack([O, sy, O], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    T = torch.stack([torch.stack([I, O, dx], dim=-1),
                     torch.stack([O, I, dy], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    transformation_mat = torch.bmm(S, T)

    return transformation_mat


def angle_to_color(angle):
    red_hue, _, _ = colorsys.rgb_to_hsv(1, 0, 0)
    green_hue, _, _ = colorsys.rgb_to_hsv(0, 1, 0)
    cur_hue = np.interp(angle, (0, np.pi), (green_hue, red_hue))
    cur_color = colorsys.hsv_to_rgb(cur_hue, 1, 1)
    return cur_color


def draw_error_and_gradient(net, img):
    dots = []
    dirs = []
    sampler = DifferentiableImageSampler('bilinear', 'zeros')
    img_torch = utils.np_img_to_torch_img(img)[None].float().cuda() / 255.0
    for y in np.linspace(Y_MIN, Y_MAX, 20):
        for x in np.linspace(X_MIN, X_MAX, 20):
            n_y, n_x = y / IMG_SIZE * 2 - 1, x / IMG_SIZE * 2 - 1
            s = CROP_SIZE / IMG_SIZE
            cur_pose = torch.tensor([s, s, n_x, n_y]).float().cuda()[None].requires_grad_(True)
            H = vec2mat_for_trans_scale(cur_pose)
            crop = sampler.warp_image(img_torch, H, out_shape=(CROP_SIZE, CROP_SIZE))
            est_e = net(crop)
            est_e.backward()
            dir = -1 * cur_pose.grad[0][2:].detach().cpu().numpy()
            dots.append(np.array([x, y, est_e.item()]))
            dirs.append(np.concatenate([np.array([x, y]), dir]))

    dots = np.array(dots)
    plt.imshow(img)
    plt.scatter(dots[:, 0], dots[:, 1], c=dots[:, 2], cmap=plt.cm.jet, s=10)
    plt.title('Error plot')
    plt.show()

    dirs = np.array(dirs)
    _, ax = plt.subplots()
    ax.imshow(img)

    for dir in dirs:
        gt_dir = np.array(CENTER) - dir[:2]
        angle = utils.angle_between(dir[2:], gt_dir)
        c = angle_to_color(angle)
        ax.arrow(dir[0], dir[1], dir[2], dir[3], head_width=1, head_length=1, color=c)
    plt.title('Gradient plot')
    plt.show()


def main():
    img = imageio.imread('./sample_data/circle.png')
    dset = ToyDataset(img)
    net = ToyNet()
    net = net.cuda()
    loader = DataLoader(dset, batch_size=32,
                        shuffle=True, num_workers=0)

    optim_list = [{"params": filter(
        lambda p: p.requires_grad, net.parameters()), "lr": 1e-6}]
    optim = torch.optim.Adam(optim_list)
    if not osp.isfile('./sample_data/checkpoint.pth.tar'):
        train(net, loader, optim, 10000)
        torch.save({
            'optim_state_dict': optim.state_dict(),
            'model_state_dict': net.state_dict(),
        }, osp.join('./sample_data/checkpoint.pth.tar'))
    else:
        weight = torch.load('./sample_data/checkpoint.pth.tar')['model_state_dict']
        net.load_state_dict(weight)
    draw_error_and_gradient(net, img)


if __name__ == "__main__":
    main()
