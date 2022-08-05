import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from sklearn.preprocessing import LabelBinarizer

class CVAE(nn.Module):
    def __init__(self,input_dim=512,hidden_dim=400,class_num=60):
        super(CVAE, self).__init__()

        self.embedding = nn.Linear(input_dim+class_num, hidden_dim)
        self.embedding_mu = nn.Linear(hidden_dim, 20)
        self.embedding_var = nn.Linear(hidden_dim, 20)
        self.fc3 = nn.Linear(30, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim+class_num)

        self.lb = LabelBinarizer()

    # 将标签进行one-hot编码
    def to_categrical(self, y: torch.FloatTensor):
        y_n = y.numpy()
        self.lb.fit(list(range(0, 10)))
        y_one_hot = self.lb.transform(y_n)
        floatTensor = torch.FloatTensor(y_one_hot)
        return floatTensor

    def encode(self, x, y):
        y_c = self.to_categrical(y)
        # 输入样本和标签y的one-hot向量连接
        con = torch.cat((x, y_c), 1)
        h1 = F.relu(self.embedding(con))
        return self.embedding_mu(h1), self.embedding_var(h1)

    def reparameterize(self, mu, logvar):
        # 训练时使用重参数化技巧，测试时不用。（测试时应该可以用）
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, y):
        y_c = self.to_categrical(y)
        # 解码器的输入：将z和y的one-hot向量连接
        cat = torch.cat((z, y_c), 1)
        h3 = F.relu(self.fc3(cat))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
