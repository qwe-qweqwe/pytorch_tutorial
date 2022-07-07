import torch

'''torch.utils.data.Dataset()（这个是用来处理单个训练样本的，包括特征、标签，怎么把它读取进来，然后可能会稍微做一点变形，做一些函数预处理，最终变成x和y这样的训练对，也就是它是对于单个样本而言的。）
torch.utils.data.Dataloader()（这个是对于多个样本而言的，也就是经过Dataset之后给它，把数据变成SGD需要的mini-batch的形式，它可能对多个样本组合成mini-batch，也可能多个周期之后对数据进行打乱，甚至有可能将数据固定地保存在gpu中，等等一系列操作都在Dataloader中实现。）
讲解4，3：21。
'''