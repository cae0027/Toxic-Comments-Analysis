import  numpy as np
from numpy.random import choice
import torch
from model import ToxicComment
from preprocessing import CleanData

preprocess = CleanData()
preprocess.pre_process_train()
preprocess.pre_process_test()

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()
# train_on_gpu = False      # test the model before moving to GPU
if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# Instantiate the model with hyperparameters
vocab_size = preprocess.review_vocab_size + 2 # +1 for the 0 padding and +1 for OOV words
output_size = 41
embedding_dim = 200
hidden_dim = 128
n_layers = 2
lr=0.000004

vocab_size = preprocess.review_vocab_size + 2 # +1 for the 0 padding and +1 for 
class Training:
    def __init__(self, vocab_size=vocab_size, output_size=41, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers, lr=lr):
        # Instantiate the model with hyperparameters
        self.net = ToxicComment(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        # instantiate loss and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
    def train(self, epochs=30, print_every=500, clip=5):
        train_loader, _ , valid_loader = preprocess.to_dataloader()
        counter = 0
        print_every = print_every
        clip = clip  # gradient clipping

        # move model to GPU, if available
        if(train_on_gpu):
            self.net.cuda()

        self.net.train()
        # train for some number of epochs
        for e in range(epochs):
            # initialize hidden state
            h = self.net.init_hidden(preprocess.batch_size)

            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                self.net.zero_grad()

                # get the output from the model
                output, h = self.net(inputs, h)

                # calculate the loss and perform backprop
                # print(output.shape)
                loss = self.criterion(output.squeeze(), labels)
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                self.optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = self.net.init_hidden(preprocess.batch_size)
                    val_losses = []
                    self.net.eval()
                    for inputs, labels in valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if(train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output, val_h = self.net(inputs, val_h)
                        val_loss = self.criterion(output.squeeze(), labels)

                        val_losses.append(val_loss.item())

                    self.net.train()
                    print("Epoch: {}/{}...".format(e+1, epochs),
                        "Step: {}...".format(counter),
                        "Loss: {:.6f}...".format(loss.item()),
                        "Val Loss: {:.6f}".format(np.mean(val_losses)))
        # save the trained model
        torch.save(self.net, './net_trained.pt')

    def test(self):
        _, test_loader, _ = preprocess.to_dataloader()
        # Get test data loss and accuracy

        test_losses = [] # track loss
        num_correct = 0

        # init hidden state
        h = self.net.init_hidden(preprocess.batch_size)

        self.net.eval()
        # iterate over test data
        for inputs, labels in test_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # get predicted outputs
            output, h = self.net(inputs, h)
            
            # calculate loss
            test_loss = self.criterion(output.squeeze(), labels)
            test_losses.append(test_loss.item())
            
            # convert output probabilities to predicted class (0 or 1)
            pred = torch.argmax(output.squeeze(), dim=1)
            
            # compare predictions to true label
            correct_tensor = pred.eq(labels.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)


        # -- stats! -- ##
        # avg test loss
        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        # accuracy over all test data
        test_acc = num_correct/len(test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))
        return test_acc


if __name__ == "__main__":
    import time
    import pickle
    # training = Training()
    # training.train()
    # training.test()
    now = time.time()
    l_rate = np.linspace(1e-06, 0.01, 700)
    no_layers = np.arange(1,10)
    embedding_dimen = np.arange(20, 1000, 5)
    hidden_dimen = np.arange(20, 1000, 5)
    batch_size = np.arange(20, 200, 10)
    seq_length = np.arange(20, 600, 20)
    epochs = np.arange(2, 30)
    

    myparams = ["l_rate", "no_layers", "embedding_dimen", "hidden_dimen", "batch_size", "seq_length", "epochs"]
    
    params_re = {name:[] for name in myparams}
    accuracy = []

    m = 1000
    params = [l_rate, no_layers, embedding_dimen, hidden_dimen, batch_size, seq_length, epochs]
    for _ in range(m):
        # print("Say something")
        param_choice = [choice(name) for name in params]
        intermediate = []
        print(param_choice)
        for j in range(10):
            preprocess = CleanData(seq_length=param_choice[5], batch_size=int(param_choice[4]))
            preprocess.pre_process_train()
            preprocess.pre_process_test()
            vocab_size = preprocess.review_vocab_size + 2 # +1 for the 0 padding and +1 for OOV words
            training = Training(vocab_size=vocab_size, output_size=41, embedding_dim=param_choice[2], hidden_dim=param_choice[3], n_layers=param_choice[1], lr=lr)
            training.train(epochs=param_choice[6])
            test_out = training.test()
            intermediate.append(test_out)
            print(f"Iteration ==================> {j+1}/10")
        for i, name in enumerate(myparams):
            params_re[name] = param_choice[i]
            print(name + ": ", param_choice[i], end="  ") 
        print(f"Parameter set: {_}/{m}")                                      
        accuracy.append(sum(intermediate)/len(intermediate))
    with open("grid_search_output.pickle", 'wb') as saveto:
        pickle.dump(params_re, saveto, protocol=pickle.HIGHEST_PROTOCOL)
    with open("grid_search_accuracy.pickle", 'wb') as f:
        pickle.dump(accuracy, f, protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print("Total time for the grid search is: {} hours".format((end - now)/3600))
                                


    