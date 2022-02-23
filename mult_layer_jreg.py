import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

from args import args

from utils import utils



# get the non-zero indeces from the initial JReg
class DL_JReg(nn.Module):
    def __init__(self, initial_jreg):
        super(DL_JReg, self).__init__()

        self.num_joints = initial_jreg.shape[0]

        # get the non zero indeces for each joint
        self.joint_operations = []
        self.joint_indeces = []

        for i in range(self.num_joints):
            non_zero = torch.nonzero(initial_jreg[i])
            self.joint_indeces.append(non_zero)

            num_non_zero = non_zero.shape[0]

            this_joint_operations = nn.Sequential(
                torch.nn.Linear(num_non_zero*3, num_non_zero*3),
                nn.ReLU(),
                torch.nn.Linear(num_non_zero*3, 3),
            ).to(args.device)

            self.joint_operations.append(this_joint_operations)

        self.joint_operations = nn.ModuleList(self.joint_operations)



    def forward(self, vertices):

        est_joints = []

        for i in range(self.num_joints):

            these_vertices = vertices[:, self.joint_indeces[i]]
            

            num_joints = self.joint_indeces[i].shape[0]

            


            these_vertices = torch.reshape(these_vertices, (-1, num_joints, 3))

            # subtract the mean

            mean = torch.mean(these_vertices, 1)

            these_vertices -= mean.unsqueeze(1).expand(these_vertices.shape)

            these_vertices = torch.reshape(these_vertices, (-1, num_joints*3))

            est_joint = self.joint_operations[i](these_vertices)


            est_joint += mean

            est_joints.append(est_joint)

        est_joints = torch.stack(est_joints, dim=1)

        return est_joints


# get the non-zero indeces from the initial JReg
class OLD_JReg(nn.Module):
    def __init__(self, initial_jreg):
        super(OLD_JReg, self).__init__()

        self.num_joints = initial_jreg.shape[0]

        # get the non zero indeces for each joint
        self.joint_operations = []
        self.joint_indeces = []

        for i in range(self.num_joints):
            non_zero = torch.nonzero(initial_jreg[i])
            self.joint_indeces.append(non_zero)

            num_non_zero = non_zero.shape[0]

            this_joint_operations = nn.Sequential(
                torch.nn.Linear(num_non_zero*3, num_non_zero*3),
                nn.ReLU(),
                torch.nn.Linear(num_non_zero*3, 3),
            ).to(args.device)

            self.joint_operations.append(this_joint_operations)

        self.joint_operations = nn.ModuleList(self.joint_operations)



    def forward(self, vertices):

        est_joints = []

        for i in range(self.num_joints):

            these_vertices = vertices[:, self.joint_indeces[i]]
            

            num_joints = self.joint_indeces[i].shape[0]

            


            these_vertices = torch.reshape(these_vertices, (-1, num_joints, 3))

            # subtract the mean

            mean = torch.mean(these_vertices, 1)

            these_vertices -= mean.unsqueeze(1).expand(these_vertices.shape)

            these_vertices = torch.reshape(these_vertices, (-1, num_joints*3))

            est_joint = self.joint_operations[i](these_vertices)


            est_joint += mean

            est_joints.append(est_joint)

        est_joints = torch.stack(est_joints, dim=1)

        return est_joints



