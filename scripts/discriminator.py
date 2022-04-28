import torch
from torch import nn

from scripts.args import args


class Discriminator(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.num_inputs = 24

        self.conv_operations = nn.Sequential(
            torch.nn.Conv2d(6, 32, 1),
            nn.ReLU(),
            torch.nn.Conv2d(32, 32, 1),
            nn.ReLU(),
        ).to(args.device)

        self.linears = nn.ModuleList(
            [nn.Linear(32, 1) for i in range(self.num_inputs)])

        self.linear_operations = nn.Sequential(
            torch.nn.Linear(32*self.num_inputs, 1024),
            nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(),
            torch.nn.Linear(1024, 1),
        ).to(args.device)

    def forward(self, rot6d):

        rot6d = rot6d.permute(0, 2, 1).unsqueeze(-1)

        conv = self.conv_operations(rot6d)
        conv = conv.permute(0, 2, 1, 3)

        preds = []

        linear_operations = self.linear_operations(conv.reshape(-1, 24*32))

        preds.append(linear_operations)

        for i in range(self.num_inputs):
            this_joint_est = self.linears[i](conv[:, i].reshape(-1, 32))

            # print("this_joint_est.shape")
            # print(this_joint_est.shape)
            preds.append(this_joint_est)

        preds = torch.stack(preds, dim=1)

        return nn.Sigmoid()(preds)


class Shape_Discriminator(nn.Module):
    # def __init__(self, num_inputs, num_joints):
    def __init__(self):
        super(Shape_Discriminator, self).__init__()

        self.shape_operations = nn.Sequential(
            torch.nn.Linear(10, 10),
            nn.ReLU(),
            torch.nn.Linear(10, 5),
            nn.ReLU(),
            torch.nn.Linear(5, 1),
        ).to(args.device)

    def forward(self, shapes):

        shape_disc = self.shape_operations(shapes)

        return nn.Sigmoid()(shape_disc)
