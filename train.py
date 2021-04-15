import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import wandb

from model import Model
import utils

def train(args):

    # Set device 
    if torch.cuda.is_available():
        device = "cuda:" + str(args.gpu)
    else:
        device = "cpu"

    print(f"set_device: {device}")

    # Initialize project
    wandb.init(project="AAA534", entity='yejilee', config=args)

    # Define data loader
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False)

    # Define model
    model = Model(layers=[2, 2, 2], num_classes=10, drop_prob=args.dropout).to(device)
    model.reset_params(args.initialize)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    progress = tqdm(range(1, 1 + args.epochs))

    # intial value
    min_loss = 10**5
    max_test = 0.0

    for epoch in progress: 
        for i, [image, label] in enumerate(trainloader, 0):
            model.train()
            x, y = image.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            wandb.log({'loss': loss})
            loss.backward()
            optimizer.step()

            if loss.item() < min_loss :
                min_loss = loss.item()

            with torch.no_grad():
                model.eval()
                total = 0.0
                correct = 0.0
                for k, [image, label] in enumerate(testloader):
                    x = image.to(device)
                    y = label.to(device)
                    
                    outputs = model(x)
                    _, output_index = torch.max(outputs, 1)          

                    total += label.size(0)
                    correct += (output_index == y).sum().float().item()
                    
                    
                    
                test_score = (correct/total)*100;
                wandb.log({'test': test_score})

            if test_score > max_test:
                max_test = test_score

            progress.write(f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Test: {test_score:.2f}%')


    wandb.config.update({"best_loss": min_loss, "best_test": max_test})
    wandb.finish()

if __name__=="__main__":

    args = utils.set_arguments()
    utils.control_randomness(args.seed)

    train(args)
