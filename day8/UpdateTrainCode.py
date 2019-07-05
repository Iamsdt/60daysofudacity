import os
from glob import glob
import numpy as np  # linear algebra
import torch
from PIL import Image
from matplotlib import pyplot as plt


def load_model(model, name):
    model.load_state_dict(torch.load(f'{name}.pt'))
    return model


def check_overfitted(loss):
    train_loss_data, valid_loss_data = loss
    plt.plot(train_loss_data, label="Training loss")
    plt.plot(valid_loss_data, label="validation loss")
    plt.legend(frameon=False)


def train(model, epochs, total_class, train_loader,
          test_loader, device, criterion, optimizer, label_class_size=16,
          valid_loss_min=np.Inf):
    n_epochs = epochs

    # compare over fitted
    train_loss_data, valid_loss_data = [], []

    class_correct = list(0. for i in range(total_class))
    class_total = list(0. for i in range(total_class))

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_loader:
            # Move input and label tensors to the default device
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()  # *data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in test_loader:
            # Move input and label tensors to the default device
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss
            valid_loss += loss.item()  # *data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(label_class_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(test_loader.dataset)

        # calculate train loss and running loss
        train_loss_data.append(train_loss)
        valid_loss_data.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss
        ))
        print('\t\tTest Accuracy: %4d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('\t\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

        print("\n")

    loss = (train_loss_data, valid_loss_data)

    return model, loss, valid_loss_min


def test(model, test_loader, device, criterion, classes,
         total_class, label_class_size=16):
    test_loss = 0.0
    class_correct = list(0. for i in range(total_class))
    class_total = list(0. for i in range(total_class))

    model.eval()  # prep model for evaluation

    for data, target in test_loader:
        # Move input and label tensors to the default device
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(label_class_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(total_class):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


def visualize(loader, classes):

    data_iter = iter(loader)
    images, labels = data_iter.next()

    fig = plt.figure(figsize=(25, 5))
    for idx in range(2):
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
        # unnormolaize first
        img = images[idx] / 2 + 0.5
        npimg = img.numpy()
        img = np.transpose(npimg, (1, 2, 0))  # transpose
        ax.imshow(img, cmap='gray')
        ax.set_title(classes[labels[idx]])


def check(path, model, train_loader, transformer, device):
    classes = train_loader.dataset.class_to_idx

    file = glob(os.path.join(path, '*.jpg'))

    for i in file[:5]:
        with Image.open(i) as f:
            img = transformer(f).unsqueeze(0)
            with torch.no_grad():
                out = model(img.to(device)).cpu().numpy()
                for key, value in classes.items():
                    if value == np.argmax(out):
                        print(key)
                plt.imshow(np.array(f))
                plt.show()
