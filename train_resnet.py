import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
#from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

# ==============================
# Prepare Hyperparameters
# ==============================
BATCH_SIZE = 100 
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3

def colossal_dataloader(batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase):
    # transform_train
    transform_train = transforms.Compose([
        transforms.Pad(4), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(32),  
        transforms.ToTensor()  
    ])
    # transform_test
    transform_test = transforms.Compose([
        transforms.ToTensor()  
    ])
    
    # Use CIFAR-10 dataset
    os.environ["DATA"] = "./autodl-tmp"
    data_path = os.environ.get("DATA", "./data")
    # Download CIFAR10 datasets with priority in a distributed setting 
    with coordinator.priority_execution():
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, transform=transform_train, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, transform=transform_test, download=True
        )
        
    # Prepare dataloaders with specific plugin configurations
    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = plugin.prepare_dataloader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dataloader, test_dataloader

    

@torch.no_grad()
def evaluate(model: nn.Module, test_dataloader: DataLoader, coordinator: DistCoordinator) -> float:
    model.eval()
    correct = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())
    total = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())
    for images, labels in test_dataloader:
        # Move to GPU
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    # # Use all_reduce for distributed accuracy calculation
    dist.all_reduce(correct)
    dist.all_reduce(total)
    accuracy = correct.item() / total.item()
    
    # Log accuracy if current process is the master process
    if coordinator.is_master():
        print(f"Accuracy of the model on the test images: {accuracy * 100:.2f} %")
    return accuracy


def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    model.train()
    is_progress_bar_enabled = coordinator.is_master()
    total_loss = 0.0
    num_batches = 0
    with tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", disable=not is_progress_bar_enabled) as pbar:
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward 
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Backward and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update total loss
            total_loss += loss.item()
            num_batches += 1

            # Print log info
            pbar.set_postfix({"loss": loss.item()})
            
    # Calculate average loss for the epoch
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "low_level_zero"],
        help="plugin to use",
    )
    parser.add_argument("-r", "--resume", type=int, default=-1, help="resume from the epoch's checkpoint")
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoint", help="checkpoint directory")
    parser.add_argument("-i", "--interval", type=int, default=5, help="interval of saving checkpoint")
    parser.add_argument(
        "--target_acc", type=float, default=None, help="target accuracy. Raise exception if not reached"
    )
    args = parser.parse_args()

    # ==============================
    # Prepare Checkpoint Directory
    # ==============================
    if args.interval > 0:
        Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    # update the learning rate with linear scaling
    # old_gpu_num / old_lr = new_gpu_num / new_lr
    global LEARNING_RATE
    LEARNING_RATE *= coordinator.world_size

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)

    booster = Booster(plugin=plugin)

    # ==============================
    # Prepare Dataloader
    # ==============================
    train_dataloader, test_dataloader = colossal_dataloader(BATCH_SIZE, coordinator, plugin)

    # ====================================
    # Prepare model, optimizer, criterion
    # ====================================
    # resent18
    model = resnet18(num_classes=10)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)

    # lr scheduler
    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=criterion, lr_scheduler=lr_scheduler
    )

    # ==============================
    # Resume from checkpoint
    # ==============================
    if args.resume >= 0:
        booster.load_model(model, f"{args.checkpoint}/model_{args.resume}.pth")
        booster.load_optimizer(optimizer, f"{args.checkpoint}/optimizer_{args.resume}.pth")
        booster.load_lr_scheduler(lr_scheduler, f"{args.checkpoint}/lr_scheduler_{args.resume}.pth")

    # ==============================
    # Train model
    # ==============================
    train_losses = []
    
    start_epoch = args.resume if args.resume >= 0 else 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        avg_loss = train_epoch(epoch, model, optimizer, criterion, train_dataloader, booster, coordinator)
        train_losses.append(avg_loss)
        
        lr_scheduler.step()

        # save checkpoint
        if args.interval > 0 and (epoch + 1) % args.interval == 0:
            booster.save_model(model, f"{args.checkpoint}/model_{epoch + 1}.pth")
            booster.save_optimizer(optimizer, f"{args.checkpoint}/optimizer_{epoch + 1}.pth")
            booster.save_lr_scheduler(lr_scheduler, f"{args.checkpoint}/lr_scheduler_{epoch + 1}.pth")

    accuracy = evaluate(model, test_dataloader, coordinator)
    if args.target_acc is not None:
        assert accuracy >= args.target_acc, f"Accuracy {accuracy} is lower than target accuracy {args.target_acc}"
    
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.plugin}_train_loss.png") 
    plt.show()


if __name__ == "__main__":
    main()
