import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import GPUtil
import time

wandb.login(key='ef483be48f529db684a143aae8b6c3580b5853ef')

wandb.init(project="vgg16-cifar100-training", name="VGG6")

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

model = VGG16(num_classes=100)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

params = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 45
}
wandb.config.update(params)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=True)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9)

def train(model, criterion, optimizer, epochs, trainloader):
    total_start_time = time.time() 
    total_gpu_usage = []

    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            gpu_loads = [gpu.load for gpu in GPUtil.getGPUs()]
            total_gpu_usage.extend(gpu_loads) 

            wandb.log({"loss": loss.item(), "GPU Utilization": gpu_loads[0] * 100 if gpu_loads else 0})

        current_gpu_avg = sum(total_gpu_usage) / len(total_gpu_usage) if total_gpu_usage else 0
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Avg GPU Util: {current_gpu_avg * 100:.2f}%')

    total_time = time.time() - total_start_time
    average_gpu_usage = sum(total_gpu_usage) / len(total_gpu_usage) if total_gpu_usage else 0

    print(f'Total training time: {total_time:.2f} seconds')
    print(f'Average GPU utilization: {average_gpu_usage * 100:.2f}%')

    wandb.log({"Total Training Time": total_time, "Average GPU Utilization": average_gpu_usage * 100})

train(model, criterion, optimizer, params["epochs"], trainloader)

wandb.finish()
