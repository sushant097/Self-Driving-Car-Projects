import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model import create_model


def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms

    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.3)
    train_x = torch.tensor(train_x.reshape(-1, 1, 100, 100))
    test_x = torch.tensor(test_x.reshape(-1, 1, 100, 100))

    train_y = torch.tensor(train_y.reshape(train_y.shape[0], 1))
    test_y = torch.tensor(test_y.reshape(test_y.shape[0], 1))

    transform = transforms.Compose(
        [transforms.Normalize((0.5,), (0.5))])
    train_dataset = CustomTensorDataset(tensors=(train_x, train_y), transform=transform)
    test_dataset = CustomTensorDataset(tensors=(test_x, test_y), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = create_model(100, 100)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 12
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_loader)

    for epoch in range(n_epochs):
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx,(data_, target_) in enumerate(train_loader):


            # optimizer zero gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in (test_loader):
                data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss/len(test_loader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
            # Saving the best weight 
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model, 'autopilot_model.pt')
                print('Detected network improvement, saving current model')

        model.train()


train()