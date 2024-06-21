import sys, os
import argparse
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import pickle
from time import perf_counter
from datetime import datetime
from model.cvit import CViT
from helpers.loader import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CViT model definition
model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
            dim=1024, depth=6, heads=8, mlp_dim=2048)
model.to(device)

print(f'using {device}')

def train(dir_path, num_epochs, test_model, batch_size, lr, weight_decay):
    dataloaders, dataset_sizes = load_data(dir_path, batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    min_val_loss=10000
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_loss = min_val_loss

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for idx, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if idx%100==0:
                print(f'Train loss: {loss.item()}')
                print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    idx * batch_size, dataset_sizes['train'], \
                    100. * idx*batch_size / dataset_sizes['train'], loss.item()))

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.float() / dataset_sizes['train']

        train_loss.append(epoch_loss)
        train_accu.append(epoch_acc)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        for idx, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0) 
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['validation']
        epoch_acc = running_corrects.float() / dataset_sizes['validation']

        val_loss.append(epoch_loss)
        val_accu.append(epoch_acc)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('validation', epoch_loss, epoch_acc))

        if epoch_loss < min_loss:
            print('\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_loss, min_loss))
            min_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())    
        
        scheduler.step()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    # load best model weights
    model.load_state_dict(best_model_wts)

    with open('weight/cvit_deepfake_detection_v2.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)

    state = {'epoch': num_epochs+1, 
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'min_loss':epoch_loss}
    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    torch.save(state, f'weight/cvit_deepfake_detection_{curr_time}.pth')

    if test_model:
        test(model, dataloaders, dataset_sizes)

    return train_loss,train_accu,val_loss,val_accu, min_loss

def test(model, dataloaders, dataset_sizes):
    model.eval()

    correct_predictions = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs).to(device).float()   
        
        _,prediction = torch.max(output,1)
        correct_predictions += (prediction == labels).sum().item()
        
    print('Test Set Prediction: ', (correct_predictions/dataset_sizes['test'])*100,'%')

def gen_parser():
    parser = optparse.OptionParser("Train CViT model.")

    parser.add_option("-e", "--epoch", type=int, dest='epoch', help='Number of epochs used for training the CViT model.')
    parser.add_option("-d", "--dir", dest='dir', help='Training data path.')
    parser.add_option("-b", "--batch", type=int, dest='batch', help='Batch size.')
    parser.add_option("-l", "--rate",  type=float, dest='rate', help='Learning rate.')
    parser.add_option("-w", "--wdecay", type=float, dest='wdecay', help='Weight decay.')
    parser.add_option("-t", "--test", type=str, dest='test', help='Test on test set.')

    (options, _) = parser.parse_args()

    dir_path = options.dir
    num_epochs = options.epoch if options.epoch else 1
    test_model = "y" if options.test else None
    batch_size = options.batch if options.batch else 32
    lr = float(options.rate) if options.rate else 0.0001
    weight_decay = float(options.wdecay) if options.wdecay else 0.0000001

    return dir_path, num_epochs, test_model, int(batch_size), lr, weight_decay

def main():
    start_time = perf_counter()
    dir_path, num_epochs, test_model, batch_size, lr, weight_decay = gen_parser()
    print('Training Configuration:')
    print(f'\npath: {dir_path}')
    print(f'\nepoch: {num_epochs}')
    print(f'\ntest_model: {test_model}')
    print(f'\nbatch_size: {batch_size}')

    train(dir_path, num_epochs, test_model, batch_size, lr, weight_decay)

    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    main()
