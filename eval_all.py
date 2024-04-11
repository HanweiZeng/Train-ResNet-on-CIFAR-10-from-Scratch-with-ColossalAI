import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoint", help="checkpoint directory")
args = parser.parse_args()


def evaluate(checkpoint_path, model, test_loader):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


model = torchvision.models.resnet18(num_classes=10).cuda()


transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.CIFAR10(root="./data/", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


results = {}

checkpoints = [f for f in os.listdir(args.checkpoint) if f.startswith('model_') and f.endswith('.pth')]
checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


for checkpoint in checkpoints:
    epoch_num = int(checkpoint.split('_')[1].split('.')[0])
    checkpoint_path = os.path.join(args.checkpoint, checkpoint)
    accuracy = evaluate(checkpoint_path, model, test_loader)
    results[epoch_num] = accuracy
    print(f"Epoch {epoch_num}: Accuracy = {accuracy*100:.2f}%")


sorted_epochs = sorted(results.keys())
accuracies = [results[epoch] for epoch in sorted_epochs]


plt.figure(figsize=(10, 5))
plt.plot(sorted_epochs, accuracies, marker='o')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)


max_accuracy = max(accuracies)
max_epoch = sorted_epochs[accuracies.index(max_accuracy)]


plt.scatter(max_epoch, max_accuracy, color='red') 
plt.text(max_epoch, max_accuracy, f'{max_accuracy:.2%}', fontsize=9, verticalalignment='bottom')

plt.savefig(f"{args.checkpoint}_accuracy_over_epochs.png")
plt.show()