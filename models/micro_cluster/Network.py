import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.cvae.CVAE_Network import CVAE

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.memory = torch.Tensor(self.args.num_classes, 2, 20)
        self.cvae = CVAE()

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input , label):
        if self.mode == 'cvae_generator':
            k = self.args.episode_way * self.args.episode_shot
            support, query = input[:k], input[k:]
            support_label, query_label = label[:k], label[k:]
            support = support.view(self.args.episode_shot, self.args.episode_way, support.shape[-1])
            query = query.view(self.args.episode_query, self.args.episode_way, query.shape[-1])
            recon_batch, con, mu, logvar = self.cvae_gen(support, query, support_label, query_label)
            proto_mu = mu.reshape(self.args.episode_shot,self.args.episode_way,-1).mean(0).squeeze()
            proto_logvar=logvar.reshape(self.args.episode_shot,self.args.episode_way,-1).mean(0).squeeze()
            label_id = support_label[:self.args.episode_way]
            for i, id in enumerate(label_id):
                self.memory[id,0]= proto_mu[i]
                self.memory[id, 1] = proto_logvar[i]
            return recon_batch, con, mu, logvar
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        elif self.mode == 'train_classifier':
            k = self.args.episode_way * self.args.episode_shot
            support, query = input[:k], input[k:]
            support_label, query_label = label[:k], label[k:]
            proto_label=support_label[:self.args.episode_way]
            mu=self.memory[proto_label,0,:]
            logvar=self.memory[proto_label,1,:]
            fake_support=self.cvae.generate(mu,logvar,proto_label,shot=self.args.episode_shot)[:,:self.num_features]
            all_support=torch.cat((fake_support,support),dim=0)
            all_support = all_support.view(self.args.episode_shot*2, self.args.episode_way, support.shape[-1])
            query = query.view(self.args.episode_query, self.args.episode_way, query.shape[-1])
            logits = self.metric_classify(all_support, query)
            return logits
        elif self.mode == 'test_classifier':
            k = self.args.episode_way * self.args.episode_shot
            support, query = input[:k], input[k:]
            support = support.view(self.args.episode_shot, self.args.episode_way, support.shape[-1])
            query = query.view(self.args.episode_query, self.args.episode_way, query.shape[-1])
            logits = self.metric_classify(support, query)
            return logits
        else:
            raise ValueError('Unknown mode')

    def metric_classify(self, support,query):
        # support: (num_sample, num_class, num_emb)
        # query: (num_sample, num_class, num_emb)

        emb_dim = support.size(-1)
        # get mean of the support

        num_sample = support.shape[0]
        num_class = support.shape[1]
        num_query = query.shape[0]*query.shape[1]#num of query*way


        proto = support.mean(dim=0)
        query = query.view(-1, emb_dim).unsqueeze(1)

        proto = proto.unsqueeze(0).expand( num_query, num_class, emb_dim).contiguous()

        logits=F.cosine_similarity(query,proto,dim=-1)
        logits=logits*self.args.temperature

        return logits

    def cvae_gen(self,support,query, support_label, query_label):
        recon_batch, mu, logvar = self.cvae(support, support_label)
        #         print(recon_batch.shape) #[64, 794]
        # ?????????????????????????????????????????????????????????one-hot??????
        flat_data = support.view( support.shape[0] * support.shape[1],-1)
        #         print(data.shape, flat_data.shape)
        y_condition = self.cvae.to_categrical(support_label)
        con = torch.cat((flat_data, y_condition), 1)

        return recon_batch, con, mu, logvar


    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            if self.args.use_gpu:
                data, label = [_.cuda() for _ in batch]
                new_fc = nn.Parameter(
                    torch.rand(len(class_list), self.num_features, device="cuda"),
                    requires_grad=True)
            else:
                data, label = [_ for _ in batch]
                new_fc = nn.Parameter(
                    torch.rand(len(class_list), self.num_features),requires_grad=True)
            data=self.encode(data).detach()


        nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

