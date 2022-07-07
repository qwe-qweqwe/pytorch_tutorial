import torch
b = torch.rand([3,2,])
c, d = torch.chunk(b,chunks=2)
print(c, d)
'''torch.chunk()（这个是用来分割张量的。第一个参数是被分割的张量，
    第二个参数chunks=多少就是分割成几个张量。第三个参数dim=多少，为1的话，就是画竖线进行分割，0的话就是画横线进行分割，默认是0.
    有余数的话，最后一个张量会小一点。）
    torch.gather()（沿着某一个维度取一些变量，）
    pytorch讲解2，6：57，我觉得这个距离应用太远，还是先去看数据读取。
'''