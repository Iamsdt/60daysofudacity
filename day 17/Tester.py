import torch
import numpy as np


def test(model, loader, device, criterion):
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        model.eval()

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Test Loss:{:.6f}..".format(test_loss),
          "\tAccuracy: {:.4f}".format(accuracy / len(loader) * 100))


def test2(model, loader, device, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(102))
    class_total = list(0. for i in range(102))

    with torch.no_grad():
        model.eval()
        # iterate over test data
        for data, target in loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update test loss
            test_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(16):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(loader.dataset)
    print('Test Loss: {:.6f}'.format(test_loss))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


def test3(model, loader, device, criterion):
    valid_loss = 0
    total = 0
    correct = 0

    with torch.no_grad:
        model.eval()  # prep model for evaluation
        for data, target in loader:
            # Move input and label tensors to the default device
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss_p = criterion(output, target)
            # update running validation loss
            valid_loss += loss_p.item() * data.size(0)
            # calculate accuracy
            proba = torch.exp(output)
            top_p, top_class = proba.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print("\tValid Loss:{:.6f}..".format(valid_loss),
              "\tAccuracy: {:.4f}".format(correct / total * 100))
