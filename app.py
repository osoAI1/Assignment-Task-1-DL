#osama mostafa elemam 4221166
#Assignment Task 1
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CPU")


trainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

validTranform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trainDs  = Flowers102(root="./data", split="train", download=True, transform=trainTransform)
testDS   = Flowers102(root="./data", split="test",  download=True, transform=validTranform)
validaDs = Flowers102(root="./data", split="val",   download=True, transform=validTranform)

trainLoader = DataLoader(trainDs,  batch_size=8, shuffle=True)
testLoader  = DataLoader(testDS,   batch_size=8, shuffle=False)
validLoader = DataLoader(validaDs, batch_size=8, shuffle=False)

numClass = 102

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, numClass)
model = model.to(device)

cross = nn.CrossEntropyLoss()

train_loss_hist = []
train_acc_hist  = []
val_acc_hist    = []

for name, p in model.named_parameters():
    if "fc" not in name:
        p.requires_grad = False

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
print("Head only")

for epoch in range(5):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = cross(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in validLoader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    print(f"epoch {epoch+1}: loss={train_loss:.4f}, trainAcc={train_acc:.4f}, validAcc={val_acc:.4f}")


for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.layer3.parameters(): 
    p.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

print("fine-tuning: layer3+layer4")
for epoch in range(5):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = cross(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in validLoader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    print(f"[FT] epoch {epoch+1}: loss={train_loss:.4f}, trainAcc={train_acc:.4f}, validAcc={val_acc:.4f}")


model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for x, y in testLoader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        test_correct += (out.argmax(1) == y).sum().item()
        test_total += y.size(0)

test_acc = test_correct / test_total
print(f"Test accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "flowers102_resnet50_finetun.pth")
print("Model saved")


epochs = list(range(1, len(train_loss_hist) + 1))

plt.figure()
plt.plot(epochs, train_loss_hist, label="Train Loss")
plt.axvline(x=5, linestyle="--", label="Stage 2 start")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs, train_acc_hist, label="Train Accuracy")
plt.plot(epochs, val_acc_hist, label="Validation Accuracy")
plt.axvline(x=5, linestyle="--", label="Stage 2 start")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

test_incorrect = test_total - test_correct
plt.figure()
plt.pie([test_correct, test_incorrect],
        labels=["Correct", "Incorrect"],
        autopct="%1.1f%%",
        startangle=90)
plt.title(f"Test Results (Acc={test_acc:.4f})")
plt.show()