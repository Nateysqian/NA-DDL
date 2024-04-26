import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.metrics import euclidean_distances
from torch.utils.data import random_split
from torchvision.datasets import FashionMNIST, EMNIST, MNIST,  SVHN
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms as transforms

device = torch.device("cuda")

# Deep Dictionary Configurations
input_dim = 784 # the input dimensions to be expected
dd_layer_config = [784//2]  # the layer configuration for the deep dictionary
# dd_layer_config = [392//2,392//4,392//8]
# dd_layer_config = [int(input_dim * 1.5), int(input_dim * 2), int(input_dim * 1.5)]  #  int(input_dim * 0.6), int(input_dim * 0.4)
sparse_cff = 1e-1  # regularization to enusure sparseness in the dictionary representation
epoch_per_level = 50  # the number of epochs to train for each layer of deep dictionary

# MLP Configurations
batch_size_train = 128   # the batch size of the MLP model (optimized via Adam)
batch_size_valid = 128
epoch_mlp = 250              # the number of epochs to train the MLP for
num_classes = 47  # the number of classes for classification (10 for MNIST)
mlp_lr = 3.5e-3  # the learning rate for the Adam optimizer to optimize the MLP model


# # EMNIST dataset
# emnist_train_data = EMNIST('./data/', split='balanced', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# train_size = 30000
# train_data, valid_data = random_split(emnist_train_data, [train_size, len(emnist_train_data) - train_size], generator=torch.Generator().manual_seed(0))
# train_loader_dd = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False, pin_memory=True)
# train_loader_mlp = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, pin_memory=True)
# valid_loader_mlp = torch.utils.data.DataLoader(valid_data, batch_size=batch_size_valid, shuffle=True, pin_memory=True)
# test_data = EMNIST('./data/', split='balanced', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# test_loader_mlp = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=True)

# #MNIST dataset
# mnist_train_data = MNIST('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# train_size = 30000
# train_data, valid_data = random_split(mnist_train_data, [train_size, len(mnist_train_data) - train_size], generator=torch.Generator().manual_seed(0))
# train_loader_dd = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False, pin_memory=True)
# train_loader_mlp = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, pin_memory=True)
# valid_loader_mlp = torch.utils.data.DataLoader(valid_data, batch_size=batch_size_valid, shuffle=True, pin_memory=True)
# test_data = MNIST('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# test_loader_mlp = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=True)

#SVHN dataset
# svhn_data = SVHN('./data/',download=True, transform=transforms.Compose(
#     [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# train_dataset = datasets.SVHN(root='./data',  download=True, transform=transform)
# train_size = 30000
# train_data, valid_data = random_split(svhn_data, [train_size, len(train_dataset) - train_size], generator=torch.Generator().manual_seed(0))
# train_loader_dd  = torch.utils.data.DataLoader(train_data,batch_size=batch_size_train, shuffle=True, pin_memory=True)
# train_loader_mlp = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, pin_memory=True)
# valid_loader_mlp = torch.utils.data.DataLoader(valid_data, batch_size=batch_size_valid, shuffle=True, pin_memory=True)
# test_data = SVHN('./data/', download=True,
#                     transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))
# test_loader_mlp = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=True)
# test_loader_mlp = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=True)

#fashion MNIST dataset
fashion_mnist_train_data = FashionMNIST('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_size = 30000
train_data, valid_data = random_split(fashion_mnist_train_data, [train_size, len(fashion_mnist_train_data) - train_size], generator=torch.Generator().manual_seed(0))
train_loader_dd = DataLoader(train_data, batch_size=len(train_data), shuffle=False, pin_memory=True)
train_loader_mlp = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, pin_memory=True)
valid_loader_mlp = DataLoader(valid_data, batch_size=batch_size_valid, shuffle=True, pin_memory=True)

fashion_mnist_test_data = FashionMNIST('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader_mlp = DataLoader(fashion_mnist_test_data, batch_size=len(fashion_mnist_test_data), shuffle=True, pin_memory=True)

# Function Class
class Identity:
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def inverse(x):
        return x


class ELUInv:

    alpha = 1.5

    @staticmethod
    def forward(x):
        return (x > 0).float()*x + (x <= 0).float()*torch.log((x/ELUInv.alpha)+1)

    @staticmethod
    def inverse(x):
        return (x > 0).float() * x + (x <= 0).float() * ELUInv.alpha * (torch.exp(x) - 1)


class TanhInv:

    alpha = 10

    @staticmethod
    def forward(x):
        return TanhInv.alpha*torch.atanh(x)

    @staticmethod
    def inverse(x):
        return torch.tanh(x/TanhInv.alpha)

class ElliotSig:

    alpha = 2.5

    @staticmethod
    def forward(x):
        return ElliotSig.alpha*x/(1 + torch.abs(x))

    @staticmethod
    def inverse(x):
        return (x >= 0).float()*(x/(ElliotSig.alpha-x)) + (x < 0).float()*(x/(ElliotSig.alpha+x))


def EuDist2(fea_a, fea_b=None, b_sqrt=True):
    # Efficiently compute the Euclidean Distance Matrix
    if fea_b is None:
        aa = torch.sum(fea_a * fea_a, dim=1)
        ab = torch.matmul(fea_a, fea_a.t())

        D = aa.unsqueeze(1) + aa - 2 * ab
        D[D < 0] = 0

        if b_sqrt:
            D = torch.sqrt(D)

        D = torch.max(D, D.t())
    else:
        aa = torch.sum(fea_a * fea_a, dim=1)
        bb = torch.sum(fea_b * fea_b, dim=1)
        ab = torch.matmul(fea_a, fea_b.t())

        D = aa.unsqueeze(1) + bb.unsqueeze(0) - 2 * ab
        D[D < 0] = 0

        if b_sqrt:
            D = torch.sqrt(D)
    return D


def constructW(fea, options):
    bSpeed = 1
    if 'options' not in locals():
        options = {}

    if 'Metric' in options:
        print("Warning: This function has been changed and the Metric is no longer supported")

    if 'bNormalized' not in options:
        options['bNormalized'] = 0

    if 'NeighborMode' not in options:
        options['NeighborMode'] = 'KNN'

    if options['NeighborMode'].lower() == 'knn':
        if 'k' not in options:
            options['k'] = 5
    elif options['NeighborMode'].lower() == 'supervised':
        if 'bLDA' not in options:
            options['bLDA'] = 0

        if options['bLDA']:
            options['bSelfConnected'] = 1

        if 'k' not in options:
            options['k'] = 0

        if 'gnd' not in options:
            raise ValueError("Label(gnd) should be provided under 'Supervised' NeighborMode!")

        if fea is not None and len(options['gnd']) != fea.shape[0]:
            raise ValueError("gnd doesn't match with fea!")
    else:
        raise ValueError("NeighborMode does not exist!")

    if 'WeightMode' not in options:
        options['WeightMode'] = 'HeatKernel'

    bBinary = 0
    bCosine = 0

    if options['WeightMode'].lower() == 'binary':
        bBinary = 1
    elif options['WeightMode'].lower() == 'heatkernel':
        if 't' not in options:
            nSmp = fea.shape[0]
            D = np.linalg.norm(fea[np.random.choice(nSmp, min(3000, nSmp), replace=False)], axis=1)
            options['t'] = np.mean(np.mean(D))
    elif options['WeightMode'].lower() == 'cosine':
        bCosine = 1
    else:
        raise ValueError("WeightMode does not exist!")
    # ===================================
    if 'bSelfConnected' not in options:
        options['bSelfConnected'] = 0

    if 'gnd' in options:
        nSmp = len(options['gnd'])
    else:
        nSmp = fea.shape[0]
    maxM = 62500000  # 500M
    BlockSize = maxM // (nSmp * 3)
    fea_cpu = fea.cpu()
    Normfea = normalize(fea_cpu.numpy())
    # if 'gnd' in options:
    #     Label = np.unique(options['gnd'])
    #     nLabel = len(Label)

    if options['NeighborMode'].lower() == 'knn':
            if options['k'] > 0:
                G = np.zeros((nSmp * (options['k'] + 1), 3))
                for i in range(1, math.ceil(nSmp / BlockSize) + 1):
                    if i == math.ceil(nSmp / BlockSize):
                        smpIdx = np.arange((i - 1) * BlockSize, nSmp)
                        dist = euclidean_distances(fea_cpu[smpIdx, :], fea_cpu)
                        dump, idx = np.sort(dist, axis=1)[:, :options['k'] + 1], np.argsort(dist, axis=1)[:,:options['k'] + 1]

                        if not bBinary:
                            if bCosine:
                                dist = np.dot(Normfea[smpIdx, :], Normfea.T).todense()
                                linidx = np.arange(1, idx.size + 1)
                                dump = dist.flat[idx.flat - 1]
                            else:
                                dump = np.exp(-dump / (2 * options['t'] ** 2))

                        G[(i - 1) * BlockSize * (options['k'] + 1):nSmp * (options['k'] + 1), 0] = np.tile(smpIdx,
                                                                                                           options[
                                                                                                               'k'] + 1)
                        G[(i - 1) * BlockSize * (options['k'] + 1):nSmp * (options['k'] + 1), 1] = idx.ravel()
                        if not bBinary:
                            G[(i - 1) * BlockSize * (options['k'] + 1):nSmp * (options['k'] + 1), 2] = dump.ravel()
                        else:
                            G[(i - 1) * BlockSize * (options['k'] + 1):nSmp * (options['k'] + 1), 2] = 1

                    else:
                        smpIdx = np.arange((i - 1) * BlockSize, i * BlockSize)
                        dist = euclidean_distances(fea_cpu[smpIdx, :], fea_cpu)
                        dump, idx = np.sort(dist, axis=1)[:, :options['k'] + 1], np.argsort(dist, axis=1)[:, :options['k'] + 1]

                        if not bBinary:
                            if bCosine:
                                dist = np.dot(Normfea[smpIdx, :], Normfea.T).todense()
                                linidx = np.arange(1, idx.size + 1)
                                dump = dist.flat[idx.flat - 1]
                            else:
                                dump = np.exp(-dump / (2 * options['t'] ** 2))

                        G[(i - 1) * BlockSize * (options['k'] + 1):i * BlockSize * (options['k'] + 1), 0] = np.tile(
                            smpIdx,
                            options[
                                'k'] + 1)
                        G[(i - 1) * BlockSize * (options['k'] + 1):i * BlockSize * (options['k'] + 1), 1] = idx.ravel()
                        if not bBinary:
                            G[(i - 1) * BlockSize * (options['k'] + 1):i * BlockSize * (options['k'] + 1),
                            2] = dump.ravel()
                        else:
                            G[(i - 1) * BlockSize * (options['k'] + 1):i * BlockSize * (options['k'] + 1), 2] = 1

                W = sp.csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))
            else:
                raise ValueError("Invalid k for 'KNN' NeighborMode.")
    # else:
    #     raise ValueError("NeighborMode does not exist!")
    W=EuDist2(fea,None,0)
    return W

class SCnetAGL:

    alpha = 2.5

    @staticmethod
    def forward(x):
        torch.cuda.empty_cache()
        sita=torch.tensor(np.ones(x.shape[0]) / x.shape[0])
        options={'NeighborMode': 'KNN', 'k': 3, 'WeightMode': 'HeatKernel','t':1}
        transposedX = torch.transpose(x, 0, 1)
        W = constructW(transposedX, options)
        dim_Reduce = 10
        lambda1 = 1e-1
        lambda2 = 1e2
        device = x.device  # Get the device of the input data
        sita=sita.to(device)
        # Convert other tensors to NumPy arrays and move to CPU
        # sita_np = sita.cpu().numpy()
        # W_np = W.cpu().numpy()

        L = torch.diag(torch.sum(W, dim=1)) - W
        sita= sita.to(x.dtype)
        temp_sitadata = torch.diag(sita) @ x
        temp_sitadata = temp_sitadata.to(device)
        temp_sitadata = temp_sitadata.to(torch.float16)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
        # A = temp_sitadata @ temp_sitadata.t() + lambda2 * torch.eye(x.size(0),device=device)
        chunk_size = 10
        num_chunks = x.size(0) // chunk_size
        A = torch.zeros(x.size(0), x.size(0), device=device)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            temp_result = temp_sitadata[start_idx:end_idx] @ temp_sitadata[start_idx:end_idx].t()
            A[start_idx:end_idx, start_idx:end_idx] = temp_result + torch.eye(chunk_size, device=device)
        A2 = torch.eye(x.size(1), device=device) + lambda1 * L - (temp_sitadata.t() @ torch.linalg.pinv(A)) @ temp_sitadata

        Y, eigvalue = torch.linalg.eig(A2)
        eigvalue = eigvalue.real  # Extract real parts of eigenvalues

        _, indices = torch.sort(eigvalue)
        eigvalue = eigvalue[indices]
        Y = Y[indices]

        if dim_Reduce < len(eigvalue):
            Y = Y[:, :dim_Reduce]
            eigvalue = eigvalue[:dim_Reduce]

        Y = Y.t()

        return torch.Tensor(Y).to(device), eigvalue
        # return ElliotSig.alpha*x/(1 + torch.abs(x))

    @staticmethod
    def inverse(x):
        return (x >= 0).float()*(x/(ElliotSig.alpha-x)) + (x < 0).float()*(x/(ElliotSig.alpha+x))


class DeepDictionary():

    def __init__(self, input_dim, **kwargs):
        """
        :param input_dim: the input dimension to be expected
        :param layer_config: the configuration for the layer dimensions of each dictionary layer
        :param activation: the activation function (default: None)
        :param sparseness_coeff: the sparseness coefficient
        """

        self.input_dim = input_dim

        self.layer_config = kwargs.get('layer_config', [self.input_dim, self.input_dim//2])  # list of layer dimensions
        if self.layer_config[0] != self.input_dim:  # in case where the layer config is given as hidden dimensions only
            self.layer_config = [self.input_dim] + self.layer_config
        assert len(self.layer_config) >= 2, "Error in specifying layer configuration, not enough layers"

        self.activation = kwargs.get('activation', SCnetAGL)    # default implies linear
        self.spar_cff = kwargs.get('sparseness_coeff', 0)   # default sparseness coefficient

        # construct dictionary
        self.dictionary_layers = [torch.rand((self.layer_config[i+1], self.layer_config[i]), requires_grad=False).to(device)
                                  for i in range(len(self.layer_config)-1)]

    def eval_layer(self, x, layer, concat_prev=False):
        """
        :param x: input of dimension (batch x dim)
        :param layer: the layer to be evaluated at (value between 0 and len(self.dictionary_layers)-1)
        :return: (Z_i, Z_i-1) 'Z' at layer 'i' and 'i-1' (batch x dim[layer])
        """

        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        # compute initial layer z_0
        d_i = self.dictionary_layers[0]
        z_i_prev = x
        z_i = z_i_prev@d_i.T@torch.inverse(d_i@d_i.T)  # first obtain Z_0

        if concat_prev:
            concat_z = z_i.clone()

        # process intermediate layer (if specified)
        for i in range(1, layer+1):  # iterate through next few layers to compute 'z_i' (if needed)
            d_i = self.dictionary_layers[i]  # obtain dictionary for this layer
            z_i_prev = z_i  # make a copy of previous z_i
            z_i_prev_ia = self.activation.inverse(z_i_prev)  # compute the inverse activated z_i_prev

            if i == len(self.dictionary_layers) - 1:  # if is last layer of deep model
                z_i = z_i_prev_ia@d_i.T@torch.inverse(d_i@d_i.T + self.spar_cff*torch.eye((len(d_i)), device=device))   # enforce sparseness
            else:
                z_i = z_i_prev_ia@d_i.T@torch.inverse(d_i@d_i.T)    # otherwise, treat as regular

            if concat_prev:
                concat_z = torch.cat([concat_z, z_i], dim=-1)

        if concat_prev:
            return z_i, z_i_prev, concat_z
        return z_i, z_i_prev


    def optimize_layer(self, x, layer):
        """
        :param x: input of dimension (batch x dim)
        :param layer: the layer to be trained (value between 0 and len(self.dictionary_layers)-1)
        :return: None
        """

        # only optimize specified layer (previous layers assumed constant during this specific layer optimization)

        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        z_layer, z_layer_prev = self.eval_layer(x, layer)  # obtain z_layer for the specified layer

        if layer == 0:  # layer '0' has no activations
            d_layer = torch.inverse(z_layer.T @ z_layer) @ z_layer.T @ z_layer_prev  # get optimal dictionary
        else:
            z_layer_prev_ia = self.activation.inverse(z_layer_prev)  # compute the inverse activated z_layer_prev
            d_layer = torch.inverse(z_layer.T@z_layer)@z_layer.T@z_layer_prev_ia

        self.dictionary_layers[layer] = d_layer  # update model dictionary

    def layer_reconstruction(self, x, layer):
        """
        Performs layer reconstruction of the previous latent 'z'
        :param x: input of dimension (batch x dim)
        :param layer: the layer to perform the reconstruction on
        :return: a reconstruction of 'x' based on learned dictionaries
        """

        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        z_layer, z_layer_prev = self.eval_layer(x, layer)
        layer_dict = self.dictionary_layers[layer]

        if layer == 0:
            z_layer_rec = z_layer @ layer_dict  # if first layer, linear activation
        else:
            z_layer_rec = self.activation.forward(z_layer@layer_dict)  # otherwise, non-linear activation

        return z_layer_rec, z_layer_prev

    def reconstruction(self, x):
        """
        Performs total reconstruction of the input image
        :param x: input of dimension (batch x dim)
        :return: a reconstruction of 'x' based on learned dictionaries
        """

        z_layer, _ = self.eval_layer(x, len(self.dictionary_layers)-1)

        # intermediate layers
        for dicts in self.dictionary_layers[:0:-1]:  # going reverse order, skip dictionary[0]
            z_layer = self.activation.forward(z_layer@dicts)

        # layer 0
        x_rec = z_layer@self.dictionary_layers[0]

        return x_rec


# define the models of the Deep Dictionary and MLP models

dd = DeepDictionary(input_dim=input_dim, layer_config=dd_layer_config,
                          activation=SCnetAGL, sparseness_coeff=sparse_cff)

mlp_input_dim = sum(dd_layer_config)

dd_mlp = nn.Sequential(
    nn.Linear(mlp_input_dim, mlp_input_dim//2),
    nn.Sigmoid(),
    nn.Linear(mlp_input_dim//2, mlp_input_dim//4),
    nn.Sigmoid(),
    nn.Linear(mlp_input_dim//4, num_classes)).to(device)

mlp_opt = torch.optim.Adam(dd_mlp.parameters(), lr=mlp_lr)
opt_schd = torch.optim.lr_scheduler.MultiStepLR(mlp_opt, [35, 50], gamma=0.25)

# begin model trainings
print('BEGIN TRAINING THE DEEP DICTIONARY MODEL')
for layer_i in range(len(dd.dictionary_layers)):
    for epoch in range(epoch_per_level):
        for batch_i, (img_dd, labels) in enumerate(train_loader_dd):
            # img_dd is batch of images - (batch x 1 x 28 x 28)
            # labels is batch of labels - (batch)

            img_dd = img_dd.to(device)
            batch_size, _, img_h, img_w = img_dd.shape
            img_dd = img_dd.view(batch_size, -1)

            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)

            # optimization
            dd.optimize_layer(img_dd, layer_i)
            img_rec = dd.reconstruction(img_dd)
            z_rec, z_prev = dd.layer_reconstruction(img_dd, layer_i)
            z_i = dd.eval_layer(img_dd, layer_i)

            # eval
            total_loss = torch.sum((img_dd - img_rec) ** 2) / batch_size
            lat_rec_loss = torch.sum((z_prev - z_rec) ** 2) / batch_size
            print(f'Layer: {layer_i} | Epoch:{epoch} - Batch {batch_i} - '
                  f'| Total Loss: {total_loss:.4f} | Latent Loss: {lat_rec_loss:.4f}')


print('BEGIN TRAINING THE MLP MODEL')

best_metric, best_model_state_dict = 0, None

plt_y_train=[]
plt_y_test=[]

def show_loss(max_x,max_y,train_data,test_data):
    x=[i for i in range(epoch_mlp)]

    #build table graph
    plt.figure(figsize=(10,6))

    #draw line graph
    plt.plot(x,train_data,label="train",color='red')
    plt.plot(x,test_data,label="test",color="blue")

    #add title and labels
    plt.title('Data Visualization')
    plt.xlabel('iter times')
    plt.ylabel('accuracy')

    #set the range number of x and y
    plt.xlim(0,max_x)
    plt.ylim(0,max_y)

    #show table graph
    plt.legend()
    plt.show()

for epoch in range(epoch_mlp):

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    # Training MLP
    dd_mlp.train()
    for batch_i, (img, labels) in enumerate(train_loader_mlp):
        mlp_opt.zero_grad()

        # compute latent variables
        with torch.no_grad():
            img = img.to(device)
            labels = labels.to(device)
            batch_size, _, img_h, img_w = img.shape
            img = img.view(batch_size, -1)

            _, _, concat_z = dd.eval_layer(img, len(dd.dictionary_layers) - 1, concat_prev=True)
            concat_z = concat_z.detach()


        # prediction and compute loss
        class_logits = dd_mlp(concat_z)  # class_logits : (batch size x num_classes)
        ce_loss = nn.functional.cross_entropy(class_logits, labels)

        # optimization
        ce_loss.backward()
        mlp_opt.step()

        # eval
        class_probs = nn.functional.softmax(class_logits, dim=-1)
        class_pred = class_probs.argmax(dim=-1)
        acc = (class_pred == labels).float().mean().item()

        train_loss.append(ce_loss.item())
        train_acc.append(acc)

    # Evaluation MLP
    dd_mlp.eval()
    for batch_i, (img, labels) in enumerate(valid_loader_mlp):
        # compute latent variables
        with torch.no_grad():
            img = img.to(device)
            labels = labels.to(device)
            batch_size, _, img_h, img_w = img.shape
            img = img.view(batch_size, -1)

            _, _, concat_z = dd.eval_layer(img, len(dd.dictionary_layers) - 1, concat_prev=True)
            concat_z = concat_z.detach()

        # prediction and compute loss
        class_logits = dd_mlp(concat_z)  # class_logits : (batch size x num_classes)
        ce_loss = nn.functional.cross_entropy(class_logits, labels)

        # eval
        class_probs = nn.functional.softmax(class_logits, dim=-1)
        class_pred = class_probs.argmax(dim=-1)
        acc = (class_pred == labels).float().mean().item()

        valid_loss.append(ce_loss.item())
        valid_acc.append(acc)

    epoch_total_acc_train = sum(train_acc)/len(train_acc)
    epoch_total_loss_train = sum(train_loss)/len(train_loss)
    epoch_total_acc_valid = sum(valid_acc) / len(valid_acc)
    epoch_total_loss_valid = sum(valid_loss) / len(valid_loss)

    # the data of table graph
    plt_y_train.append(epoch_total_acc_train)
    plt_y_test.append(epoch_total_acc_valid)

    print(f'---------------------------- Epoch:{epoch} ----------------------------------------')
    print(f'[TRAIN] | Loss: {epoch_total_loss_train:.4f} | ACC: {epoch_total_acc_train:.4f}')
    print(f'[VALID] | Loss: {epoch_total_loss_valid:.4f} | ACC: {epoch_total_acc_valid:.4f}')

    # record the best metric
    if epoch_total_acc_valid > best_metric:
        best_model_state_dict = copy.deepcopy(dd_mlp.state_dict())
        best_metric = epoch_total_acc_valid
    opt_schd.step()

# Final Evaluation on Test Set
# Evaluation MLP
dd_mlp.load_state_dict(best_model_state_dict)
dd_mlp.eval()
test_loss, test_correct = 0, 0

for batch_i, (img, labels) in enumerate(test_loader_mlp):
    # compute latent variables
    with torch.no_grad():
        img = img.to(device)
        labels = labels.to(device)
        batch_size, _, img_h, img_w = img.shape
        img = img.view(batch_size, -1)

        _, _, concat_z = dd.eval_layer(img, len(dd.dictionary_layers) - 1, concat_prev=True)
        concat_z = concat_z.detach()

    # prediction and compute loss
    class_logits = dd_mlp(concat_z)  # class_logits : (batch size x num_classes)
    ce_loss = nn.functional.cross_entropy(class_logits, labels, reduction='sum')
    test_loss += ce_loss.item()

    # eval
    class_probs = nn.functional.softmax(class_logits, dim=-1)
    class_pred = class_probs.argmax(dim=-1)
    num_correct = (class_pred == labels).float().sum().item()
    test_correct += num_correct


test_loss_avg = test_loss/len(test_loader_mlp.dataset)
test_acc = test_correct/len(test_loader_mlp.dataset)

print(f'---------------------------- FINAL TEST RESULTS ----------------------------------------')
print(f'[TEST] | Loss: {test_loss_avg:.4f} | ACC: {test_acc:.4f}')

show_loss(max_x=epoch_mlp,max_y=1,train_data=plt_y_train,test_data=plt_y_test)






