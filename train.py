from args import args
import wandb
import torch
from torch import nn, optim

from tqdm import tqdm

from crop_model import Crop_Model

from data import load_data, data_set
from visualizer import draw_gradients

def train_crop_model():

    model = Crop_Model().to(args.device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    data_dict = load_data("train")

    val_data_dict = load_data("validation")
    
    this_data_set = data_set(data_dict)

    val_data_set = data_set(val_data_dict)

    loss_function = nn.MSELoss()

    loader = torch.utils.data.DataLoader(this_data_set, batch_size = args.training_batch_size, num_workers=4, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size = args.training_batch_size, num_workers=1, shuffle=True)

    for epoch in range(args.train_epochs):

        total_loss = 0

        iterator = iter(loader)

        val_iterator = iter(val_loader)

        for iteration in tqdm(range(len(loader))):

            batch = next(iterator)

            for item in batch:
                batch[item] = batch[item].to(args.device) 

            batch['estimated_j3d'].requires_grad = True

            optimizer.zero_grad()
            estimated_loss = model.forward(batch)

            estimated_error_loss = loss_function(estimated_loss, batch['mpjpe'])

            pred_grad = torch.autograd.grad(
                                estimated_error_loss, batch['estimated_j3d'], 
                                retain_graph=True, 
                                create_graph=True
                                )[0]

            # batch['estimated_j3d'].requires_grad = False

            gradient_loss = loss_function(pred_grad, batch['gt_gradient'])

            loss = estimated_error_loss+args.grad_loss_weight*gradient_loss

            total_loss += loss.item()

            loss.backward()

            if(args.wandb_log):

                wandb.log({"loss": loss.item(), "gradient_loss": gradient_loss.item(), "estimated_error_loss": estimated_error_loss.item()})

            optimizer.step()


            del batch

            if(iteration%10==0):

                model.eval()
                val_batch = next(val_iterator)

                for item in val_batch:
                    val_batch[item] = val_batch[item].to(args.device) 

                estimated_loss = model.forward(val_batch)

                val_loss = loss_function(estimated_loss, val_batch['mpjpe'])

                if(args.wandb_log):
                    wandb.log({"validation loss": val_loss.item()}, commit=False)

                model.train()

                del val_batch

        print(f"epoch: {epoch}, loss: {total_loss}")

        if(args.wandb_log):
            draw_gradients(model, "train", "train")
            draw_gradients(model, "validation", "validation")

        torch.save(model.state_dict(), f"models/linearized_model_{args.crop_scalar}_epoch{epoch}.pt")

        # scheduler.step()
    
    
    return model

