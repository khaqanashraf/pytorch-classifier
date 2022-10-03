import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.model_selection import KFold

class Classifier():
  def __init__(self, train_data, class_names, create_model, model_name='./model'):
    self.model_name = model_name
    self.train_data = train_data
    self.class_names = class_names
    self.create_model = create_model
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def train_val_dataset(self, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(self.train_data))), test_size=val_split)
    train = torch.utils.data.dataset.Subset(self.train_data, train_idx)
    valid = torch.utils.data.dataset.Subset(self.train_data, val_idx)
    return train, valid

  def validate_model(self, model, validloader, criterion):
    model.eval()
    correct, total = 0, 0
    running_loss = 0
    actual_targets = []
    predicted_labels = []
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(validloader, 0):

        # Get inputs
        inputs, targets = data[0].to(self.device), data[1].to(self.device)

        # Generate outputs
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        running_loss += loss.item()

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        actual_targets.extend(np.array(targets.cpu()))
        predicted_labels.extend(np.array(predicted.cpu()))
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

      return running_loss/len(validloader), correct / total, actual_targets, predicted_labels

  def train_model(self, model, e, trainloader, optimizer, criterion, validloader=None, early_stopping=None):
    # setup the model for training and initialize the early stopping criteria if the experments is required
    # Note that early stopping will work only if the validation data is given
    model.train()
    validation_losses = []
    training_losses = []
    for epoch in range(e):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        running_accuracy = 0.0
        
        # Iterate over the training data for each epoch
        for i, data in enumerate(trainloader, 0):

            # get data batch and its labels and cast them to device for GPU or CPU
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            
            # Initialize optimizer to zero
            optimizer.zero_grad()

            # get model output for given input batch
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Back propagate the loss
            loss.backward()

            # Update gradients for the given model
            optimizer.step()

            # Add up the loss for all training batches
            running_loss += loss.item()

            # Get the output class predicted by the given model
            _, preds = torch.max(outputs.data, 1)
            running_accuracy += (preds == labels).sum().item()

        # Average the running loss to get training loss for given epoch
        training_loss = running_loss/len(trainloader)
        training_losses.append(training_loss)
        print(f"Epoch: {epoch}; Training Loss: {training_loss}; Running Accuracy: {running_accuracy/len(trainloader.dataset)}; Time Taken: {time.time() - start_time}")
        # Check if the validation data is given for experimentation
        # if given then calculate validation loss and validation accuracy and use early stopping if required
        if validloader != None:
          validation_loss, validation_accuracy, valid_targets, valid_predictions = self.validate_model(model, validloader, criterion)
          validation_losses.append(validation_loss)
          # Print accuracy
          print(f'validation loss: {validation_loss}, validation accuracy: {validation_accuracy}')
          torch.save(model.state_dict(), self.model_name)

          # Early stopping
          if early_stopping!=None:
            if early_stopping(validation_loss, training_loss):
              print('Early Stopping....')
              print(classification_report(valid_targets, valid_predictions, target_names=self.class_names))
              return model, {'train': training_losses, 'valid': validation_losses}

            print(f'Trigger times: {early_stopping.count}')
          if epoch==e-1:
            print(classification_report(valid_targets, valid_predictions, target_names=self.class_names))

    return model, {'train': training_losses, 'valid': validation_losses}



  def kfold(self, epochs=10, k=5, batch_size=4, early_stopping=None):
    validation_losses = {}
    kfold = KFold(n_splits=k, shuffle=True)
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(self.train_data)):
        print('fold:'+ str(fold) +' ======================================')
        model, criterion, optimizer = self.create_model()

        model.to(self.device)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, sampler=valid_subsampler)

        model, losses = self.train_model(model, epochs, trainloader, optimizer, criterion, validloader, early_stopping)

        validation_losses[fold] = losses['valid']
        

    return validation_losses

  def experiments(self, epochs=10, split=0.2, batch_size=4, early_stopping=None):
    model, criterion, optimizer = self.create_model()

    model.to(self.device)

    train_subset, valid_subset = self.train_val_dataset(split)

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=True)

    model, losses = self.train_model(model, epochs, trainloader, optimizer, criterion, validloader, early_stopping)
    return losses

# EarlyStopper class is used to stop training if the validation loss going to in crease more to training loss
class EarlyStopper():
  # Initialize early stopper by giving patience level and minimum loss difference that should be considered
  def __init__(self, patience=2, min_delta=0.2) -> None:
      self.count = 0
      self.patience = patience
      self.min_delta = min_delta

  # if the validation loss increased by min delta by patience time then this method will return true
  def __call__(self, valid_loss, train_loss) -> bool:
      current_delta = valid_loss-train_loss
      if current_delta>self.min_delta:
        self.count += 1
      else:
        self.count = 0
        
      return self.count>=self.patience

