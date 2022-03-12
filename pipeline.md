session 0:
    trainloader: 50
    train_set: 30000
    testloader: 60
    
base train:

    batch: 75*3*84*84
    
    support: 5*5*3*84*84 class*sample_num      query: 10*5*3*84*84
    
    label: 50=15*5

    after encoder

    support: 25*512    query: 50*512

replace_base_fc:
    
    trainloader: 235    trainset: 30000     batch: 128*3*84*84  class: 60
    replace model.module.fc.weight = proto_list(60,512) 每个类别样本的特征均值

validation (感觉有点问题，从session 1-8的数据中采样，数据泄露？)
    
    batch: 25(5*5)*3*84*84   dataloader: 1   label: 25

    update_fc_avg: self.fc.weight = new_fc(5,512)

test(将训练的类别和验证的类别合起来测试) 60+5 class: 
    
    batch: 65*100   testset: 6500   testloader: 65  data: 100*3*84*84
    proto: model.module.fc.weight[:test_class, :].detach() 前65维作为分类proto
