import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import GraphConvolution, MyGraphConvolution, InnerProductGraphConvolution, LayerNormalization
import pdb

'''
class GAE(nn.Module):

    #remove the variational component 
    """Encoder using GCN layers"""

    def __init__(self, n_feat, n_hid, n_latent, dropout):
        super(GAE, self).__init__()

        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2_mu = GraphConvolution(n_hid, n_latent)
        self.gc2_var = GraphConvolution(n_hid, n_latent)
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7

    def encoder(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gc2_mu(x, adj)
        logvar = self.gc2_var(x, adj)
        std = torch.exp(logvar)
        return mu, std

    def reparameterize(self, mu, std):
        if self.training:
            #eps = torch.randn(std.size())
            #eps = _get_variable(True, eps)
            eps = Variable(std.data.new(std.size()).normal_())
            eps.cuda()
            return torch.mul(eps, std) + mu
        else:
            return mu

    def decoder(self, z):
        #pdb.set_trace()
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        #adj = self.sigmoid(torch.mm(z, z.t()))
        return adj

    def forward(self, x, adj, vae=False):
        if vae:
            mu, std = self.encoder(x, adj)
            z = self.reparameterize(mu, std)
            return self.decoder(z), mu, std
        else:
            mu, std = self.encoder(x, adj)
            return self.decoder(mu), mu, std
'''



class GAE(nn.Module):

    #remove the variational component 
    """Encoder using GCN layers"""

    def __init__(self, n_feat, n_hid, n_latent, adj, dropout):
        super(GAE, self).__init__()

        #pdb.set_trace()
        self.gc1 = MyGraphConvolution(n_feat, n_hid, adj)
        #self.gc1 = InnerProductGraphConvolution(n_feat, n_hid, adj)
        self.ln1 = LayerNormalization(n_hid)
        self.gc2_mu = MyGraphConvolution(n_hid, n_latent, adj)
        #self.gc2_mu = InnerProductGraphConvolution(n_hid, n_latent, adj)
        self.ln2 = LayerNormalization(n_latent)
        self.gc2_var = MyGraphConvolution(n_hid, n_latent, adj)
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7

    def encoder(self, x):
        x = F.relu(self.ln1(self.gc1(x)))
        #x = F.relu(self.gc1(x))
        #mu = F.relu(self.ln2(self.gc2_mu(x)))
        mu = F.relu(self.gc2_mu(x))
        logvar = self.gc2_var(x)
        std = torch.exp(logvar)
        return mu, std

    def reparameterize(self, mu, std):
        if self.training:
            #eps = torch.randn(std.size())
            #eps = _get_variable(True, eps)
            eps = Variable(std.data.new(std.size()).normal_())
            eps.cuda()
            return torch.mul(eps, std) + mu
        else:
            return mu

    def decoder(self, z):
        #pdb.set_trace()
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        #adj = self.sigmoid(torch.mm(z, z.t()))
        return adj

    def forward(self, x, vae=False):
        if vae:
            mu, std = self.encoder(x)
            z = self.reparameterize(mu, std)
            return self.decoder(z), mu, std
        else:
            mu, std = self.encoder(x)
            return self.decoder(mu), mu, std
            #return self.decoder(mu), x, std


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3_mu = nn.Linear(512, 128)
        self.fc3_var = nn.Linear(512, 128)

        self.fc4 = nn.Linear(128, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, 2048)

        self.fc = nn.Linear(2048, 128, bias=True)
        self.fc.weight.data.normal_(0.0, 0.001)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def encode(self, x):
        #pdb.set_trace()
        h1 = self.leaky_relu(self.fc1(x))
        h2 = self.leaky_relu(self.fc2(h1))
        #mu = self.fc3_mu(h2)
        mu = self.fc(x)
        logvar = self.fc3_var(h2)
        std = torch.exp(logvar)
        return mu, std

    def reparameterize(self, mu, std):
        if self.training:
            eps = Variable(std.data.new(std.size()).normal_())
            eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4 = self.leaky_relu(self.fc4(z))
        h5 = self.leaky_relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x, vae=False):
        if vae:
            mu, std = self.encode(x)
            z = self.reparameterize(mu, std)
            return self.decode(z), mu, std
        else:
            mu, std = self.encode(x)
            return self.decode(mu), mu, std


class CLS(nn.Module):
    def __init__(self, num_classes):
        super(CLS, self).__init__()

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.cls = nn.Linear(2048, num_classes, bias=False)

        #self.leaky_relu = nn.LeakyReLU(0.2, True)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        return h2

    def forward(self, x):
        #z = self.encode(x)
        return F.sigmoid(self.cls(x))
