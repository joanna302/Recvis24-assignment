import argparse
import os
import json 

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.optim import lr_scheduler 
from torchvision import datasets

from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="../data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default="5",
        metavar="ES",
        help="number of epochs before stopping the training if the loss_val is not decreasing",
    )
    parser.add_argument(
        "--freeze",
        type=str,
        default="T",
        metavar="F",
        help="if true only the classifier is trained",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        metavar="F",
        help="optimizer used. SGD, Adam, AdamW",
    )
    parser.add_argument(
        "--data_aug",
        type=str,
        default="F",
        metavar="F",
        help="if T, use data augmentation",
    )
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    return loss.data.item()


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for i, (data, target) in enumerate(val_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= i

    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return validation_loss


def main():
    """Default Main Function."""
    # optionsval_values
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms, data_augm_transforms = ModelFactory(args.model_name).get_all()

    #freeze the layers before the classifier if requested 
    if args.freeze == "T" :
        for param in model.model.parameters(): 
            param.requires_grad=False 
        for param in model.classifier.parameters(): 
            param.requires_grad=True 
        experiment_path = f"{args.experiment}/{args.model_name}/freeze"
    else : 
        for param in model.model.parameters(): 
            param.requires_grad=True 
        experiment_path = f"{args.experiment}/{args.model_name}/unfreeze"
    if args.data_aug == "T" : 
        experiment_path = f"{experiment_path}/data_aug"

    os.makedirs(experiment_path, exist_ok = True) 

    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    #Data initialization and loading
    if args.data_aug == "F" : 
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    else : 
        # If data augmentation 
        # Load the dataset without augmentation 
        original_train_dataset = datasets.ImageFolder(args.data + "/train_images", 
                                                    transform=data_transforms, 
                                                    )

        # Load the dataset with augmentation 
        augmented_train_dataset = datasets.ImageFolder(args.data + "/train_images", 
                                                    transform=data_augm_transforms, 
                                                    )

        # Both datasets contatenation 
        combined_dataset = torch.utils.data.ConcatDataset([original_train_dataset, augmented_train_dataset])

        train_loader = torch.utils.data.DataLoader(combined_dataset, 
                                                batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    if args.optimizer=="SGD" :
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer=="Adam": 
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer=="AdamW": 
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented")

    # Setup scheduler 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma = 0.9)

    # Loop over the epochs
    best_val_loss = 1e8
    val_loss_tab = []
    training_loss_tab = []
    es = False 

    for epoch in range(1, args.epochs + 1):
        # training loop
        train_loss = train(model, optimizer, train_loader, use_cuda, epoch, args)
        training_loss_tab.append(train_loss)
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        val_loss_tab.append(val_loss)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = experiment_path + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)

        # also save the model every epoch
        #model_file = experiment_path + "/model_" + str(epoch) + ".pth"
        #torch.save(model.state_dict(), model_file)
        # print(
        #     "Saved model to "
        #     + model_file
        #     + f". You can run `python evaluate.py --model_name {args.model_name} --model "
        #     + best_model_file
        #     + "` to generate the Kaggle formatted csv file\n"
        # )
        
        print(
            f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
        if epoch > args.early_stopping : 
            if all(val_loss > val_loss_tab[-i] for i in range(2, args.early_stopping+1)) : 
                es = True 
                print("Early stopping")
                break
        scheduler.step()

    # Save the logs 
    log_file = experiment_path + "/log.txt"
    
    results_dict = {}
    results_dict['best_val_loss'] = best_val_loss 
    results_dict['val_loss'] = val_loss 
    results_dict['train_loss'] = train_loss 
    results_dict['early_stopping'] = es 
    results_dict['nb_epochs'] = epoch

    save_dict = {}
    save_dict['arg'] = args.__dict__
    save_dict['results'] = results_dict 

    with open(log_file, 'w') as f:
        json.dump(save_dict, f, indent=4)

    # Plot the training and validation loss 
    plt.plot(training_loss_tab, label='Training Loss')
    plt.plot(val_loss_tab, label='Validation Loss')
    plt.legend() 
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show() 
    plt.savefig(experiment_path+"/graph.pdf")

if __name__ == "__main__":
    main()
