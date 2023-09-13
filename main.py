"""Main script for the workshop."""
from joblib import dump, load
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchvision.transforms as transforms

from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import SGDClassifier
from scipy.stats import loguniform
from secai.torch_models import LinearRegression, MultiLayerPerceptron, \
    ConvolutionalNeuralNetwork, LSTMModel, EarlyStopping

import medmnist
from medmnist import INFO

import matplotlib.pyplot as plt
import seaborn as sns


def train_model(torch_model, torch_optimizer, patience, n_epochs, path,
                training_dataloader, validation_dataloader):
    """
    Train a PyTorch torch_model with early stopping.

    Parameters
    ----------
    torch_model : PyTorch torch_model
        The torch_model to be optimized.
    torch_optimizer : PyTorch optimizer
        The optimizer to minimize the bce_loss and update the weights after
        each iteration.
    patience : int
        Number of subsequent training epochs on which the validation accuracy
        does not decrease. If this is fulfilled, the training is stopped, and
        the best torch_model so far returned.
    n_epochs : int
        Maximum number of training epochs.
    path : Path
        Location where to store intermediate models.
    training_dataloader : DataLoader
        The training input_data loader.
    validation_dataloader : DataLoader
        The validation input_data loader for early stopping.

    Returns
    -------
    torch_model : PyTorch torch_model
        The optimized torch_model.
    optimizer : PyTorch optimizer
        The optimizer that is used.
    current_epoch : int
        The current_epoch at which the training has stopped.
    bce_loss : float
        The final validation bce_loss.
    avg_training_losses : list[float]
        The training losses after each current_epoch.
    avg_validation_losses : list[float]
        The validation losses after each current_epoch.
    """
    loss_function = nn.CrossEntropyLoss()
    # to track the training bce_loss as the torch_model trains
    training_losses = []
    # to track the validation bce_loss as the torch_model trains
    validation_losses = []
    # to track the average training bce_loss per current_epoch as the
    # torch_model trains
    avg_training_losses = []
    # to track the average validation bce_loss per current_epoch as the
    # torch_model trains
    avg_validation_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
    for current_epoch in range(1, n_epochs + 1):
        ###################
        # train the torch_model #
        ###################
        torch_model.train()  # prep torch_model for training
        for batch, (input_data, target) in enumerate(training_dataloader, 1):
            input_data = input_data.to(DEVICE)
            target = target.to(DEVICE)
            # clear the gradients of all optimized variables
            torch_optimizer.zero_grad()
            # forward pass: compute predicted outputs
            output = torch_model(input_data)
            # calculate the bce_loss
            target = target.squeeze().long()
            bce_loss = loss_function(output, target)
            # backward pass: gradient of the bce_loss with respect to
            # parameters
            bce_loss.backward()
            # perform a single optimization step (parameter update)
            torch_optimizer.step()
            # record training bce_loss
            training_losses.append(bce_loss.item())
        ######################
        # validate the torch_model #
        ######################
        validation_outputs = []
        validation_targets = []
        torch_model.eval()  # prep torch_model for evaluation
        for input_data, target in validation_dataloader:
            input_data = input_data.to(DEVICE)
            target = target.to(DEVICE)
            # forward pass: compute predicted outputs
            output = torch_model(input_data)
            # calculate the bce_loss
            target = target.squeeze().long()
            bce_loss = loss_function(output, target)
            # record validation bce_loss
            validation_losses.append(bce_loss.item())
            validation_targets.append(target.cpu().detach().numpy().flatten())
            validation_outputs.append(
                output.cpu().detach().numpy().argmax(axis=1))
        # print training/validation statistics
        # calculate average bce_loss over an current_epoch
        training_loss = np.average(training_losses)
        validation_loss = np.average(validation_losses)
        avg_training_losses.append(training_loss)
        avg_validation_losses.append(validation_loss)
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{current_epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
                     f'train_loss: {training_loss:.5f} '
                     f'valid_loss: {validation_loss:.5f}')
        print(print_msg)
        # early_stopping needs the validation bce_loss to check if it has
        # decresed,
        # and if it has, it will make a checkpoint of the current torch_model

        early_stopping(-accuracy_score(np.hstack(validation_targets),
                                       np.hstack(validation_outputs)),
                       torch_model, torch_optimizer, current_epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return (torch_model, torch_optimizer, early_stopping.epoch, bce_loss,
            avg_training_losses, avg_validation_losses)


def test_model(torch_model, training_dataloader, test_dataloader):
    """
    Train a PyTorch torch_model with early stopping.

    Parameters
    ----------
    torch_model : PyTorch torch_model
        The torch_model to be optimized.
    training_dataloader : DataLoader
        The training input_data loader.
    test_dataloader : DataLoader
        The test input_data loader for early stopping.

    Returns
    -------
    training_targets : list[int]
        The training targets.
    training_outputs : list[int]
        The predictions on the training set.
    test_targets : list[int]
        The test targets.
    test_outputs : list[int]
        The predictions on the test set.
    """
    # to track the average accuracy scores on the training dataset
    training_outputs = []
    training_targets = []
    # to track the average accuracy scores on the test dataset
    test_outputs = []
    test_targets = []
    # initialize the early_stopping object
    torch_model.eval()  # prep torch_model for inference
    with torch.no_grad():
        for batch, (input_data, target) in enumerate(training_dataloader, 1):
            input_data = input_data.to(DEVICE)
            target = target.to(DEVICE)
            # forward pass: compute predicted outputs
            output = torch_model(input_data)
            # Prepare the target output
            target = target.squeeze().long()
            target = target.float().resize_(len(target), 1)
            training_targets.append(target.cpu().detach().numpy().flatten())
            training_outputs.append(
                output.cpu().detach().numpy().argmax(axis=1))
        for batch, (input_data, target) in enumerate(test_dataloader, 1):
            input_data = input_data.to(DEVICE)
            target = target.to(DEVICE)
            # forward pass: compute predicted outputs
            output = torch_model(input_data)
            # Prepare the target output
            target = target.squeeze().long()
            target = target.float().resize_(len(target), 1)
            test_targets.append(target.cpu().detach().numpy().flatten())
            test_outputs.append(output.cpu().detach().numpy().argmax(axis=1))
    return training_targets, training_outputs, test_targets, test_outputs


sns.set_theme(context="notebook")

DEVICE = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

data_flag = 'pathmnist'

DATASET_INFO = INFO['pathmnist']
TASK = DATASET_INFO['task']
LABELS = DATASET_INFO['label']
N_CHANNELS = DATASET_INFO['n_channels']
N_CLASSES = len(DATASET_INFO['label'])
N_PIXELS = 28*28
BATCH_SIZE = 256
PATIENCE = 5
NUM_EPOCHS = 200

# preprocessing
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# load the data
DataClass = getattr(medmnist, DATASET_INFO['python_class'])
train_dataset = DataClass(split='train', transform=data_transform, as_rgb=True)
validation_dataset = DataClass(split='val', transform=data_transform,
                               as_rgb=True)
test_dataset = DataClass(split='test', transform=data_transform, as_rgb=True)

training_input = []
training_target = []
for data in tqdm(utils.data.DataLoader(dataset=train_dataset, shuffle=True)):
    training_input.append(data[0].numpy().reshape(-1, N_PIXELS))
    training_target.append(data[1].numpy().flatten())

training_df = pd.DataFrame(np.vstack(training_input),
                           columns=[f"Pixel {k+1}" for k in range(N_PIXELS)])
training_df["Target"] = [LABELS[str(d)] for d in np.hstack(training_target)]
training_df["Numeric target"] = np.hstack(training_target)

validation_input = []
validation_target = []
for data in tqdm(utils.data.DataLoader(dataset=validation_dataset,
                                       shuffle=True)):
    validation_input.append(data[0].numpy().reshape(-1, N_PIXELS))
    validation_target.append(data[1].numpy().flatten())

validation_df = pd.DataFrame(np.vstack(validation_input),
                             columns=[f"Pixel {k+1}" for k in range(N_PIXELS)])
validation_df["Target"] = [
    LABELS[str(d)] for d in np.hstack(validation_target)]
validation_df["Numeric target"] = np.hstack(validation_target)

test_input = []
test_target = []
for data in tqdm(utils.data.DataLoader(dataset=test_dataset, shuffle=True)):
    test_input.append(data[0].numpy().reshape(-1, N_PIXELS))
    test_target.append(data[1].numpy().flatten())

test_df = pd.DataFrame(np.vstack(test_input),
                       columns=[f"Pixel {k+1}" for k in range(N_PIXELS)])
test_df["Target"] = [LABELS[str(d)] for d in np.hstack(test_target)]
test_df["Numeric target"] = np.hstack(test_target)

_, axs = plt.subplots(3, 3, sharex="all", sharey="all")
for k in range(9):
    sns.heatmap(
        data=training_df.loc[
            k, [f"Pixel {k+1}" for k in range(N_PIXELS)]
        ].values.astype(float).reshape(28, 28).T, ax=axs.flatten()[k],
        square=True, xticklabels=False, yticklabels=False)
plt.tight_layout()

_, axs = plt.subplots(2, 1, sharex="all")
sns.histplot(data=training_df.loc[:, [f"Pixel {k+1}" for k in range(0, 5)]],
             ax=axs[0])
sns.boxplot(data=training_df.loc[:, [f"Pixel {k+1}" for k in range(0, 5)]],
            ax=axs[1], orient="h")
plt.tight_layout()

pca = PCA().fit(training_df.loc[:, [f"Pixel {k+1}" for k in range(N_PIXELS)]])

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, 785), y=np.cumsum(pca.explained_variance_ratio_),
             ax=axs)
axs.axhline(y=0.95, c="r")
axs.set_xlabel("Pixel k")
axs.set_ylabel("Accumulated explained variance")
axs.set_xlim((0, N_PIXELS+5))
axs.set_ylim((0.55, 1.01))
plt.tight_layout()

# preprocessing
data_transform.transforms.append(transforms.Normalize(mean=[.5], std=[.5]))

# load the data
DataClass = getattr(medmnist, DATASET_INFO['python_class'])
train_dataset = DataClass(split='train', transform=data_transform, as_rgb=True)
validation_dataset = DataClass(split='val', transform=data_transform,
                               as_rgb=True)
test_dataset = DataClass(split='test', transform=data_transform, as_rgb=True)

training_input = []
training_target = []
for data in utils.data.DataLoader(dataset=train_dataset, shuffle=True):
    training_input.append(data[0].numpy().reshape(-1, N_PIXELS))
    training_target.append(data[1].numpy().flatten())

training_df = pd.DataFrame(np.vstack(training_input),
                           columns=[f"Pixel {k+1}" for k in range(N_PIXELS)])
training_df["Target"] = [LABELS[str(d)] for d in np.hstack(training_target)]
training_df["Numeric target"] = np.hstack(training_target)

validation_input = []
validation_target = []
for data in utils.data.DataLoader(dataset=validation_dataset, shuffle=True):
    validation_input.append(data[0].numpy().reshape(-1, N_PIXELS))
    validation_target.append(data[1].numpy().flatten())

validation_df = pd.DataFrame(np.vstack(validation_input),
                             columns=[f"Pixel {k+1}" for k in range(N_PIXELS)])
validation_df["Target"] = [
    LABELS[str(d)] for d in np.hstack(validation_target)]
validation_df["Numeric target"] = np.hstack(validation_target)

test_input = []
test_target = []
for data in utils.data.DataLoader(dataset=test_dataset, shuffle=True):
    test_input.append(data[0].numpy().reshape(-1, N_PIXELS))
    test_target.append(data[1].numpy().flatten())

test_df = pd.DataFrame(np.vstack(test_input),
                       columns=[f"Pixel {k+1}" for k in range(N_PIXELS)])
test_df["Target"] = [LABELS[str(d)] for d in np.hstack(test_target)]
test_df["Numeric target"] = np.hstack(test_target)

cv_training_df = pd.concat((training_df, validation_df))
X_train = cv_training_df.loc[
          :, [f"Pixel {k+1}" for k in range(N_PIXELS)]].to_numpy()
y_train = cv_training_df.loc[:, "Numeric target"].to_numpy()
test_fold = [-1] * len(training_df) + [1] * len(validation_df)

X_test = test_df.loc[:, [f"Pixel {k+1}" for k in range(N_PIXELS)]].to_numpy()
y_test = test_df.loc[:, "Numeric target"].to_numpy()

cv = PredefinedSplit(test_fold=test_fold)

try:
    clf = load("results/sklearn_linear_model_baseline.joblib")
except FileNotFoundError:
    clf = SGDClassifier(loss="log_loss",
                        early_stopping=True).fit(X=X_train, y=y_train)
    dump(clf, "results/sklearn_linear_model_baseline.joblib")

print(f"{clf.score(X=X_train, y=y_train)}")
print(f"{clf.score(X=X_test, y=y_test)}")

try:
    clf = load("results/sklearn_linear_model.joblib")
except FileNotFoundError:
    clf = RandomizedSearchCV(
        estimator=SGDClassifier(loss="log_loss"), n_iter=50, n_jobs=-1,
        cv=cv, verbose=10, param_distributions={
            "alpha": loguniform(a=1e-5, b=1e1)}).fit(X=X_train, y=y_train)
    dump(clf, "results/sklearn_linear_model.joblib")

_, axs = plt.subplots()
sns.lineplot(data=pd.DataFrame(clf.cv_results_), x="param_alpha",
             y="mean_test_score", ax=axs)
axs.set_xscale("log")
axs.set_xlim((1e-5, 1e1))
axs.set_xlabel("Ridge parameter alpha")
axs.set_ylabel("Validation accuracy")
plt.tight_layout()

print(f"{clf.score(X=X_train, y=y_train)}")
print(f"{clf.score(X=X_test, y=y_test)}")

train_loader = utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=BATCH_SIZE, shuffle=True,
                                     num_workers=2, pin_memory=True)
validation_loader = utils.data.DataLoader(dataset=validation_dataset,
                                          batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2, pin_memory=True)
test_loader = utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=2, pin_memory=True)

model = LinearRegression(in_features=N_PIXELS, num_classes=N_CLASSES)
model = model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-2)

model, optimizer, epoch, loss, average_training_losses, \
    average_validation_losses = train_model(model, optimizer, PATIENCE,
                                            NUM_EPOCHS,
                                            "results/torch_linear_model.pt",
                                            train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_linear_model.pt")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")

model = MultiLayerPerceptron(hidden_layer_sizes=(N_PIXELS, ),
                             num_classes=N_CLASSES)
model = model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=1e-2)

model, optimizer, epoch, loss, average_training_losses, \
    average_validation_losses = train_model(model, optimizer, PATIENCE,
                                            NUM_EPOCHS,
                                            "results/torch_naive_mlp_model.pt",
                                            train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_naive_mlp_model.pt")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")

model = MultiLayerPerceptron(hidden_layer_sizes=(N_PIXELS, 128, 64, ),
                             num_classes=N_CLASSES)
model = model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=1e-2)

model, optimizer, epoch, loss, average_training_losses, \
    average_validation_losses = train_model(model, optimizer, PATIENCE,
                                            NUM_EPOCHS,
                                            "results/torch_deep_mlp_model.pt",
                                            train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_deep_mlp_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")

model = ConvolutionalNeuralNetwork(in_channels=1, num_classes=N_CLASSES)
model = model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=1e-1)

model, optimizer, epoch, loss, average_training_losses, \
    average_validation_losses = train_model(model, optimizer, PATIENCE,
                                            NUM_EPOCHS,
                                            "results/torch_cnn_model.pt",
                                            train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_cnn_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")

model = LSTMModel(input_size=28, hidden_size=100, num_layers=1,
                  bidirectional=False, dropout=0., num_classes=N_CLASSES)
model = model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=1e-1)

model, optimizer, epoch, loss, average_training_losses, \
    average_validation_losses = train_model(model, optimizer, PATIENCE,
                                            NUM_EPOCHS,
                                            "results/torch_1L_lstm_model.pt",
                                            train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_1L_lstm_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")

model = LSTMModel(input_size=28, hidden_size=100, num_layers=2,
                  bidirectional=False, dropout=0., num_classes=N_CLASSES)
model = model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=1e-1)

model, optimizer, epoch, loss, average_training_losses, \
    average_validation_losses = train_model(model, optimizer, PATIENCE,
                                            NUM_EPOCHS,
                                            "results/torch_2L_lstm_model.pt",
                                            train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_2L_lstm_model.pt")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")

model = LSTMModel(input_size=28, hidden_size=100, num_layers=1,
                  bidirectional=True, dropout=0., num_classes=N_CLASSES)
model = model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=1e-1)

model, optimizer, epoch, loss, average_training_losses,\
    average_validation_losses = train_model(
        model, optimizer, PATIENCE, NUM_EPOCHS,
        "results/torch_1L_bi_lstm_model.pt", train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_1L_bi_lstm_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")

model = LSTMModel(input_size=28, hidden_size=100, num_layers=2,
                  bidirectional=True, dropout=0., num_classes=N_CLASSES)
model = model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=1e-1)

model, optimizer, epoch, loss, average_training_losses, \
    average_validation_losses = train_model(
        model, optimizer, PATIENCE, NUM_EPOCHS,
        "results/torch_2L_bi_lstm_model.pt", train_loader, validation_loader)

_, axs = plt.subplots()
sns.lineplot(x=np.arange(1, len(average_training_losses) + 1),
             y=average_training_losses, ax=axs, label="Training loss")
sns.lineplot(x=np.arange(1, len(average_validation_losses) + 1),
             y=average_validation_losses, ax=axs, label="Validation loss")
axs.set_xlim((0, round(len(average_training_losses) / 5) * 5))
axs.set_xlabel("Number of epochs")
axs.set_ylabel("Loss")
plt.tight_layout()

checkpoint = torch.load("results/torch_2L_bi_lstm_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(DEVICE)
model.eval()

y_train_true, y_train_pred, y_test_true, y_test_pred = test_model(
    torch_model=model, training_dataloader=train_loader,
    test_dataloader=test_loader)

print(f"{accuracy_score(np.hstack(y_train_true), np.hstack(y_train_pred))}")
print(f"{accuracy_score(np.hstack(y_test_true), np.hstack(y_test_pred))}")
