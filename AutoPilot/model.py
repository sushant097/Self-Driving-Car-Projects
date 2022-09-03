import torch
import torch.nn as nn
import torchvision


def create_model(image_h, image_w):
    layers = []
    layers.append(torchvision.transforms.Resize((image_h, image_w)))
    # Conv -> relu -> maxpool
    layers.append(nn.Conv2d(1, 32, (3, 3), padding='same'))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d((2,2)))
    layers.append(nn.Conv2d(32, 32, (3, 3), padding='same'))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d((2,2)))

    #
    layers.append(nn.Conv2d(32, 64, (3, 3), padding='same'))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d((2,2)))
    layers.append(nn.Conv2d(64, 64, (3, 3), padding='same'))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d((2,2)))

    #
    layers.append(nn.Conv2d(64, 128, (3, 3), padding='same'))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d((2,2)))
    layers.append(nn.Conv2d(128, 128, (3, 3), padding='same'))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d((2,2)))

    layers.append(nn.Flatten())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(128, 1024))
    layers.append(nn.Linear(1024, 256))
    layers.append(nn.Linear(256, 64))
    layers.append(nn.Linear(64, 1))

    model = nn.Sequential(*layers)

    return model


if __name__ == "__main__":
    model = create_model(100, 100)
    print(model)
    img = torch.rand((1, 1, 224, 224))
    output = model(img)
    print(output.shape)