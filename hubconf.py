
# practice torchhub on other ds..
# using cifar10..

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
           
        

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            nn.BatchNorm2d(256),

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def get_lossfn_and_optimizer(mymodel):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=1e-3)
    return loss_fn, optimizer

def load_data():

    # Download training data from open datasets.
    # using cifar10 dataset...
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform_train,
    )

    # Download test data from open datasets.
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform_test,
    )
    
    return training_data, test_data

def create_dataloaders(training_data, test_data, batch_size=128):

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=2)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

        
    return train_dataloader, test_dataloader

def get_model():
    
    model = NeuralNetwork().to(device)

    return model

def _train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def _test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train(train_dataloader, test_dataloader, model1, loss_fn1, optimizer1, epochs=5):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train(train_dataloader, model1, loss_fn1, optimizer1)
        _test(test_dataloader, model1, loss_fn1)
    print("Done!")
    return model1

def save_model(model1,mypath="model.pth"):
    torch.save(model1.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

def load_model(mypath="model.pth"):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    return model

def sample_test(model1, test_data):
    model1.eval()
    x, y = test_data.__getitem__(0)
    
    with torch.no_grad():
        pred = model1(x.cuda())
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

