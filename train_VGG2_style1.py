import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import wandb
import GPUtil
from time import time

wandb.login(key='ef483be48f529db684a143aae8b6c3580b5853ef')
wandb.init(project="vgg16-cifar100-training", name="VGG2-style1")

class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        return x

model = VGG16(classes=100).to('cuda' if torch.cuda.is_available() else 'cpu')

params = {"batch_size": 16, "lr": 0.001, "epochs": 40}
wandb.params.update(params)

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transformation)
train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transformation)
test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)

def train(epoch_count, loader):
    model.train()
    start_time = time()
    gpu_stats = []

    for epoch in range(epoch_count):
        for i, (data, target) in enumerate(loader):
            data, target = data.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            current_gpu_load = [gpu.load for gpu in GPUtil.getGPUs()]
            gpu_stats.extend(current_gpu_load)
            wandb.log({"loss": loss.item(), "GPU Utilization": current_gpu_load[0] * 100 if current_gpu_load else 0})

        print(f'Epoch {epoch+1}/{epoch_count}, Loss: {loss.item()}, Avg GPU Load: {sum(gpu_stats) / len(gpu_stats) * 100:.2f}%')

    training_duration = time() - start_time
    print(f'Total Training Time: {training_duration:.2f} seconds')
    print(f'Average GPU Utilization: {sum(gpu_stats) / len(gpu_stats) * 100:.2f}%')
    wandb.log({"Total Training Time": training_duration, "Average GPU Utilization": sum(gpu_stats) / len(gpu_stats) * 100})

train(params["epochs"], train_loader)
wandb.finish()
