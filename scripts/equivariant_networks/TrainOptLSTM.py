# in this script to fellow fairly closely Zou et al
# but the create a equivariant descriptor for each type 
# of tree topology from the beggining
# We use a an embedding "across sequences" instead to
# within sequences. 
# this is an optimized version of the lstm network 

import numpy as np
# import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)

import h5py
import torch.nn as nn
from torch.utils import data
import itertools
import json
import sys
import time

from os import path
sys.path.insert(0, '../')

from utilities import SequenceDataSet

import gc

gc.collect()
nameScript = sys.argv[0].split('/')[-1]

# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson, flush = True)
print("=================================================")

# opening Json file 
jsonFile = open(nameJson) 
dataJson = json.load(jsonFile)   

# loading the input data from the json file
ngpu = dataJson["ngpu"]                  # number of GPUS
lr = dataJson["lr"]                      # learning rate
batch_size = dataJson["batchSize"]       # batch size

dataRoot = dataJson["dataRoot"]          # data folder
modelRoot = dataJson["modelRoot"]        # folder to save the data

labelFiles = dataJson["labelFile"]        # file with labels
sequenceFiles = dataJson["matFile"]            # file with sequences

#n_train_samples = dataJson["nTrainSamples"]
#n_test_samples = dataJson["nTestSamples"]

nEpochs  = dataJson["nEpochs"]           # number of epochs

if "summaryFile" in dataJson:
    summary_file = dataJson["summaryFile"]   # file in which we 
                                             # summarize the end result
else :
    summary_file = "summary_file.txt"

print("=================================================\n")

print("Learning Rate {} ".format(lr))
print("Batch Size {} \n".format(batch_size))

print("=================================================")

print("Loading Sequence Data in " + sequenceFiles, flush = True)
print("Loading Label Data in " + labelFiles, flush = True)

# we read the labels as list of strings
with open(dataRoot+labelFiles, 'r') as f:
    label_char = f.readlines()

# we read the sequence as a list of strings
with open(dataRoot+sequenceFiles, 'r') as f:
    seq_string = f.readlines()

# extracting the number of samples
n_samples = len(label_char)
print(n_samples)
# extracting the sequence lenghth
seq_length = len(seq_string[0])-1

# function to convert string to numbers
def convert_string_to_numbers(str, dict):
    ''' str: is the string to convert,
        dict: dictionary with the relative ordering of each char'''

    # create a map iterator using a lambda function
    numbers = map(lambda x: dict[x], str)

    return np.fromiter(numbers, dtype=np.int64)

# We need to extract the dictionary with the relative positions
# fo each aminoacid

# first we need to extract all the different chars
strL = ""
for c in seq_string[0][:-1]:
    if not c in strL:
        strL += c

# we sort them
strL = sorted(strL)

# we give them a relative order
dict_amino = {}
for ii, c in enumerate(strL):
    dict_amino[c] = ii

# looping over the labels and create array
# here each element of the label_char has
# the form "1\n", so we only take the first one
labels = np.fromiter(map(lambda x: int(x[0])-1,
                         label_char), dtype= np.int64)

mats = np.zeros((len(seq_string), seq_length), dtype = np.int64)


# this is pretty slow (optimize in numba)
for ii, seq in enumerate(seq_string):
    # note each line has a \n character at the end so we remove it
    mats[ii,:] = convert_string_to_numbers(seq[:-1], dict_amino).reshape((1,seq_length))


mats = mats.reshape((n_samples, -1, seq_length))    
# dims of mats is (N_samples, n_sequences, seq_length)

print("Total number of samples: {}".format(labels.shape[0]))
#print("Number of training samples: {}".format(n_train_samples))
#print("Number of testing samples: {}".format(n_test_samples))

#assert n_train_samples + n_test_samples <=  n_samples

# we perform the training/validation splitting
# we need to truncate a bit the lenght of the sequences
trunc_length = 1550

# we perform the training/validation splitting
#outputTrain = torch.from_numpy(labels[0:n_train_samples])
#inputTrain  = torch.from_numpy(mats[0:n_train_samples, :, :trunc_length])

#outputTest = torch.from_numpy(labels[-n_test_samples:])
#inputTest  = torch.from_numpy(mats[-n_test_samples:, :,:trunc_length])

# creating the dataset objects (we use regular data sets)
#datasetTrain = data.TensorDataset(inputTrain, outputTrain) 
#datasetTest = data.TensorDataset(inputTest, outputTest) 


##############################################################
# We specify the networks (this are quite simple, we should be
# able to build some more complex)


## copy paste from the Zou 2019 model
# here is the residue block
class _ResidueModule(torch.nn.Module):

    def __init__(self, channel_count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)


class _DoubleEmbedding(torch.nn.Module):
    # we use first an embedding for the 

    def __init__(self, length_dict, embeding_dim, trunc_length = 1550):
        super().__init__()
        
        self.embedding_layer = nn.Embedding(length_dict, embeding_dim)
        self._res_module_1 = _ResidueModule(embeding_dim)
        self._res_module_2 = _ResidueModule(embeding_dim)

    def forward(self, x):
        # (none, 4, 1550)

        # input will be not hot-encoded 
        x = self.embedding_layer(x).permute([0, 1, 3, 2])
        # (none, 4, 1550, chn_dim) without permute
        # (none, 4, chn_dim, 1550) with permutation

        d0 =  self._res_module_1(x[:,0,:,:])
        d1 =  self._res_module_1(x[:,1,:,:])     
        d2 =  self._res_module_1(x[:,2,:,:])     
        d3 =  self._res_module_1(x[:,3,:,:])   

        # Quartet 1 (12|34)
        # d01 = d0 + d1
        d01 = self._res_module_2(d0 + d1)

        # d23 = d2 + d3
        d23 = self._res_module_2(d2 + d3)

        G1 = d01 + d23

        #Quartet 2 (13|24)
        # d02 = d0 + d2
        d02 = self._res_module_2(d0 + d2)

        # d13 = d1 + d3
        d13 = self._res_module_2(d1 + d3)

        # F56 = F5 + F6
        G2 = d02 + d13

        # Quartet 3 (14|23)
        # d03 = d0 + d3
        d03 = self._res_module_2(d0 + d3)

        # d12 = d1 + d2
        d12 = self._res_module_2(d1 + d2)

        # F34 = F3 + F4
        G3 = d03 + d12

        x = torch.cat([torch.unsqueeze(G1,1), 
                       torch.unsqueeze(G2,1), 
                       torch.unsqueeze(G3,1)], dim = 1)

        # (none, 3, emb_dim, 1550)
        return x


class _Model(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self, embeding_dim = 80, hidden_dim = 20, 
                      num_layers = 3, output_size = 20, 
                      dropout = 0.0):
        """Create a neural network model."""
        super().__init__()

        self.embedding_layer = _DoubleEmbedding(20, embeding_dim)
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.classifier = torch.nn.Linear(self.output_size, 1)
        self.rnn = nn.LSTM(embeding_dim, hidden_dim, 
                           num_layers, dropout=dropout,
                           batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, self.output_size)

        # flatenning the parameters
        self.rnn.flatten_parameters()


    def forward(self, x):
        """Predict phylogenetic trees for the given sequences.
        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequences.
        Returns
        -------
        torch.Tensor
            The predicted adjacency trees.
        """
        device = x.device
        batch_size = x.size()[0]
        
        # counte the number of equal proteins in a site 

        # this is the structure preserving embedding
        g =  self.embedding_layer(x)
        
        x1 = g[:,0,:,:]
        x2 = g[:,1,:,:]
        x3 = g[:,2,:,:]
        # (none,embeding_dim, 1550)

        # contanenation in the batch dimesion
        # (3*none, 80, 1550)
        X = torch.cat([x1, x2, x3], dim  = 0)

        # (3*none, 1550, hidden_dim)
        r_output, hidden = self.rnn(X.permute([0, 2, 1]))

        # TODO: perhaps add an attention layer here!
        # extracting only the last in the sequence
        # (3*none, hidden_dim)
        r_output_last = r_output[:, -1, :] 

        # not sure if this helps
        out = r_output_last.contiguous().view(-1, self.hidden_dim)
        
        # (3*none, out_put_dimensions)
        output = self.fc(out)

 
        X_combined = self.classifier(output) 
        # (3*none, 1)

        X_combined = X_combined.view(3,batch_size)


        return torch.permute(X_combined, [1, 0])


###############################################

#dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,
#                                              batch_size=batch_size,
#                                              shuffle=True,
#                                              pin_memory=True)

#dataloaderTest = torch.utils.data.DataLoader(datasetTest,
#                                             batch_size=batch_size,
#                                             shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# specify loss function
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
# criterion = torch.nn.CrossEntropyLoss()

# dropout /TODO: add to the input json file
dropout = 0.2

# define the model
model = _Model(dropout = 0.2).to(device)
# model = torch.jit.script(_Model(dropout = dropout)).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("number of parameters is %d"%count_parameters(model))

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# specify scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)


print("Starting Training Loop")
maxAccuracy = 0
maxTAccuracy = 0
getEpoch =[]
getAcc = []
trainEpoch = []
trainLoss = []
testEpoch = []
testLoss = []
for epoch in range(1, nEpochs + 1):
    # monitor training loss
    train_loss = 0.0
    start = time.time()
    model.train()
    Ttotal,Tcorrect = 0, 0
    #get the training data by sampling 2000 samples per epoch
    sample_slice = (epoch-1)*2%53
    #slice the 2000 training sequences and 2000 training labels
    if(sample_slice < 26):
      inputTrain  = torch.from_numpy(mats[sample_slice*1000:(sample_slice+2)*1000, :, :trunc_length])
      outputTrain = torch.from_numpy(labels[sample_slice*1000:(sample_slice+2)*1000])
    elif(sample_slice == 26):
      firstseq_tensor = torch.from_numpy(mats[26000:27000, :, :trunc_length])
      secondseq_tensor = torch.from_numpy(mats[0:1000, :, :trunc_length])
      inputTrain = torch.cat((firstseq_tensor,secondseq_tensor),0)
      firstlab_tensor = torch.from_numpy(labels[26000:27000])
      secondlab_tensor = torch.from_numpy(labels[0:1000])
      outputTrain = torch.cat((firstlab_tensor,secondlab_tensor),0)
    else:
      inputTrain  = torch.from_numpy(mats[(sample_slice-27)*1000:(sample_slice-25)*1000, :, :trunc_length])
      outputTrain = torch.from_numpy(labels[(sample_slice-27)*1000:(sample_slice-25)*1000])
    #print(inputTrain.size())
    #print(outputTrain.size())
    datasetTrain = data.TensorDataset(inputTrain, outputTrain)
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True)
    
    
    ###################
    # train the model #
    ###################
    for genes, quartets_batch in dataloaderTrain:
        # send to the device (either cpu or gpu)
        genes, quartets_batch = genes.to(device), quartets_batch.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        quartetsNN = model(genes)
        _, predicted = torch.max(quartetsNN, 1)
        #calculate training accuracy
        Ttotal += quartets_batch.size(0)
        Tcorrect += (predicted == quartets_batch).sum().item()
        # calculate the loss
        loss = criterion(quartetsNN, quartets_batch)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
    trainEpoch.append(epoch)
    trainLoss.appedn(train_loss)
    end = time.time()
    # print avg training statistics
    TrainaccuracyTest = Tcorrect / Ttotal
    print(Tcorrect)
    print(Ttotal)
    print(TrainaccuracyTest)
    train_loss = train_loss/len(dataloaderTrain)
    print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f} \tTime Elapsed: {:.6f}[s]'.format(
        epoch, 
        optimizer.param_groups[0]['lr'],
        train_loss,
        end - start
        ), flush=True)
    if TrainaccuracyTest >maxTAccuracy:
      maxTAccuracy = TrainaccuracyTest
    # advance the step in the scheduler
    exp_lr_scheduler.step() 
    
    # we compute the test accuracy every 10 epochs 
    if epoch % 10 == 0 :
        #get the testing data by  collecting 50 samples per lba dataset
        test_loss = 0.0
        testlabel = []
        testsequence = []
        for index in range(0,27):
          eachlab_lba = labels[index*1000:(index+1)*1000]
          eachseq_lba = mats[index*1000:(index+1)*1000, :, :trunc_length]
          random_index = np.random.uniform(0,1000,50)
          random_int_index = random_index.astype(int)
          testlabel = eachlab_lba[random_int_index]
          testsequence = eachseq_lba[random_int_index,:,:trunc_length]
        outputTest = torch.from_numpy(testlabel)
        inputTest  = torch.from_numpy(testsequence)
        datasetTest = data.TensorDataset(inputTest, outputTest)
        dataloaderTest = torch.utils.data.DataLoader(datasetTest,
                                                     batch_size=batch_size,
                                                     shuffle=True)
                                                   
        model.eval()
        correct, total = 0, 0

        start = time.time()
        for genes, quartets_batch in dataloaderTest:
            #send to the device (either cpu or gpu)
            genes, quartets_batch = genes.to(device), quartets_batch.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            quartetsNN = model(genes)
            # calculate the loss
            _, predicted = torch.max(quartetsNN, 1)
             # calculate the loss
            loss = criterion(quartetsNN, quartets_batch)
              # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            test_loss += loss.item()
            total += quartets_batch.size(0)
            correct += (predicted == quartets_batch).sum().item()
        testEpoch.apppend(epoch)
        testLoss.apppend(test_loss)
        end = time.time()
        accuracyTest = correct/total
        getEpoch.append(epoch)
        getAcc.append(accuracyTest)
        print('Epoch: {} \tTest accuracy: {:.6f}  \tTime Elapsed: {:.6f}[s]'.format(epoch, 
                                                         accuracyTest,
                                                         end - start))

        if accuracyTest > maxAccuracy:
            maxAccuracy = accuracyTest
            torch.save(model.state_dict(), modelRoot  +
                "saved_{}_{}_lr_{}_batch_{}_lba_best.pth".format(nameScript.split(".")[0],
                                                                 nameJson.split(".")[0],
                                                                 str(lr), 
                                                                 str(batch_size)))

        model.eval()
        correct, total = 0, 0

        for genes, quartets_batch in dataloaderTrain:
            #send to the device (either cpu or gpu)
            genes, quartets_batch = genes.to(device), quartets_batch.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            quartetsNN = model(genes)
            # calculate the loss
            _, predicted = torch.max(quartetsNN, 1)
            
            total += quartets_batch.size(0)
            correct += (predicted == quartets_batch).sum().item()

        accuracyTrain = correct/total

        print('Epoch: {} \tTrain accuracy: {:.6f}'.format(epoch, 
                                                         accuracyTrain))



torch.save(model.state_dict(), modelRoot  +
           "saved_{}_{}_lr_{}_batch_{}_lba_last.pth".format(nameScript.split(".")[0],
                                                            nameJson.split(".")[0],
                                                            str(lr), 
                                                            str(batch_size)))

if not path.exists(summary_file):
    with open(summary_file, 'w') as f:
        f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {}\n".format("Script name",
                                    " Json file",
                                    "label file",
                                    "lerning rate", 
                                    "batch size", 
                                    "max testing accuracy", 
                                    "train loss", 
                                    "N epoch", 
                                    "chnl_dim",
                                    "embd_dim"))
info_output = np.c_[getEpoch,getAcc]
np.savetxt("EpochAccuracy1300.csv", info_output, delimiter=",")     
test_output = np.c_[testEpoch,testLoss]
np.savetxt("testLoss1300.csv", test_output, delimiter=",")    
train_output = np.c_[trainEpoch,trainLoss]
np.savetxt("trainLoss1300.csv", train_output, delimiter=",")    
# we write the last data to a file
with open(summary_file, 'a') as f:
    f.write("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n".format(nameScript.split(".")[0],
                                    nameJson.split(".")[0],
                                    labelFiles,
                                    str(lr), 
                                    str(batch_size), 
                                    str(maxAccuracy), 
                                    str(train_loss),
                                    str(nEpochs)))