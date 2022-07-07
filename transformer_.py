import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
batch_size = 2
max_num_src_words = 8
max_num_tgt_words = 8
max_src_seq_words = 5
max_tgt_seq_words = 5
model_dim = 8
max_src_seq_words = 5
max_tgt_seq_words = 5
max_position_len = 5
src_len=torch.Tensor([2,4]).to(torch.int32)
tgt_len=torch.Tensor([4,3]).to(torch.int32)
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_words-L)),0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L,)),(0,max_tgt_seq_words-L)),0) for L in tgt_len])
src_embedding_table=nn.Embedding(max_num_src_words+1,model_dim)
tgt_embedding_table=nn.Embedding(max_num_tgt_words+1,model_dim)
src_embedding=src_embedding_table(src_seq)
tgt_embedding=tgt_embedding_table(tgt_seq)
pos_mat = torch.arange(max_position_len).reshape((-1,1))
i_mat = torch.pow(10000,torch.arange(0,8,2).reshape((1,-1))/model_dim)
pe_embedding_table = torch.zeros(max_position_len,model_dim)
pe_embedding_table[:,0::2] = torch.sin(pos_mat/i_mat)
pe_embedding_table[:,1::2] = torch.cos(pos_mat/i_mat)
pe_embedding = nn.Embedding(max_position_len,model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table,requires_grad=False)
src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)+1),0)for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)+1),0)for _ in tgt_len]).to(torch.int32)
src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0, max(src_len)-L)),0) for L in src_len]),2)
valid_encoder_pos_matric = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1,2))
invalid_encoder_pos_matrix = 1-valid_encoder_pos_matric
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)
score = torch.randn(batch_size, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention,-1e9)
prob = F.softmax(masked_score, -1)

# 这些是intra-attention mask。
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0, max(src_len)-L)),0) for L in src_len]),2)
valid_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0, max(tgt_len)-L)),0) for L in tgt_len]),2)
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos,valid_encoder_pos.transpose(1,2))
invalid_cross_pos_matrix = 1-valid_cross_pos_matrix
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)


valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones((L, L))),(0, max(tgt_len)-L, 0, max(tgt_len)-L)),0) for L in tgt_len], 0)
invalid_decoder_tri_matrix = 1-valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
# print(invalid_decoder_tri_matrix)
score=torch.randn(batch_size,max(tgt_len),max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix,-1e9)
prob = F.softmax(masked_score,-1)
print(tgt_len)
print(prob)

def scaled_dot_product_attention(Q, K, V, att_mask):
    score = torch.bmm(Q, K.transpose(-2,-1))/torch.sqrt(model_dim)
    masked_score = score.masked_fill(att_mask,-1e9)
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)
    return context
'''max_num_src_words源序列的单词总数（源序列的单词表里面最多有8个词。）
model_dim单词的维度（我的理解就是单个单词最多用多少维表示，原论文中是512，我也更确定了这一点。我觉得翻译为单词维度比特征维度更准确。）
max_src_seq_words（这个是序列的最大长度。这个是源序列最大单词数量。是一句话的最大单词数吗？我记得一个序列是可以不止一句话的。）
src_len（这个记录了源序列内每个句子内单词的个数。长这样：tensor([2, 4], dtype=torch.int32)。）
序列（一个序列中包含若干个句子。）
src_seq源序列（这个就是源序列，虽然曾经觉得准确的名字是src_seq_index，因为直接表明具体形式的是src_embedding，但我现在认为，处理的时候处理这个更简介的index格式更好，将最具体的特别地标出来在处理上更好。
        这个东西长这样：
        tensor([[6, 2, 0, 0, 0],
                [3, 6, 6, 7, 0]])
        就是源序列,内部是若干个句子，再内部是单词索引。
        这个tensor就是一个序列，该序列内部的每个列表都表示一个句子，列表内部的数字代表单词的索引。单词表要从1开始，0要留给padding。里面的每个句子都是先把有意义的放上去，0在后面，不会出现0和有意义的词交错的情况。）
src_embedding_table（这个东西出来是
        tensor([[ 0.5528,  1.8205,  0.4812,  1.3853,  2.0397,  0.9970,  2.8227, -0.5962],
        [-0.5067,  0.0149,  1.2804,  0.6966,  1.1971, -0.0777, -0.2806,  0.3657],
        [-0.6302, -1.0364, -0.1344, -0.4719,  1.8047,  2.1974, -1.0568,  1.3502],
        [ 0.7620,  0.1171,  0.1379, -0.2764, -0.7112,  1.9431,  0.8043, -0.6205],
        [-1.7865, -0.6310,  0.3947,  0.0340, -1.4315,  0.0161, -1.2777, -0.3831],
        [ 0.0972, -1.0677, -0.4474,  1.1807, -0.5114, -0.6390,  0.0924, -1.8908],
        [-0.3110,  0.2131, -1.8088,  1.4974, -1.4979,  1.2311,  0.3355,  0.4077],
        [-2.5418, -1.1898, -1.2080, -0.5993,  0.5280,  1.3093, -1.2794, -1.3107],
        [ 1.0498,  0.9842, -1.2831,  0.8252, -0.0717, -0.5996, -2.2093,  0.3967]],
       requires_grad=True)
        内部的每一行表示一个单词。我的理解就是源序列的单词表。另外就是源嵌入表和目标嵌入表是不同的，本来就是要翻译的，我觉得不一样也是可以理解的。
        我觉将里面替换为内部，更能清楚其本身和内部的元素不是一个东西。他说这是一个2维的矩阵。第0行是padding，第1行到第8行是分配给单词的。）
pos_mat（出来是
        tensor([[0],
        [1],
        [2],
        [3],
        [4]])
        就是每一行都是一样的。
.reshape((-1,1))是转化成一列的。）
pos_mat（pos_mat决定着行，i_mat决定着列。一个矩阵反映pos变化，一个矩阵反映i变化。要构建两个矩阵相乘的话，pos这个矩阵每一行都是一样的。i这个矩阵每一列都是一样的。）
i_mat(pos_mat除以i_mat就是括号里面的。再分别进行sin和cos就可以了。）
pe_embedding_table（先是随机化了一个二维矩阵。然后进行赋值。偶数列是一个sin函数，奇数列是一个cos函数。这里是用两个矩阵相乘来构建的。这是一个常量，所以不需要转到cuda上。这样就可以得到每个序列的位置嵌入了。）
pe_embedding_table[:,0::2]（挑出了偶数列。那么我就懂了，单个冒号的意思是不管，就是随便都可以。）
pe_embedding_table[:,1::2]（挑出了奇数列。）
to(torch.int32)（视频中说是转成torch.int32的格式，但具体是什么我不清楚。 待完成）
max_src_seq_words（源序列中单个句子的最大长度）
max_position_len（这是单个句子的最大长度，准确讲就是句子长度。代码中是认为句子长度统一，长度的不同体现在padding上。）
pos位置（位置指的是绝对位置、相对位置的那个位置，跟索引不是一个东西。）
pe_embedding（是一个以句子长度，单词维度为参数词表。也就是说，每个词的位置信息是5维的。输入序列位置而不是序列，输出该序列的位置嵌入。就是说，这个其实是单个句子的位置表。而且现在句子长度已经统一了，因此句子的位置嵌入表）
（pe_embedding和pe_embedding_table一样的话，为什么要分2个？？）
src_pos（这个和src_seq的区别在哪里呢？这个东西的最内部的数字，是位置嵌入的索引，因此每个句子都是[0,1,2,3,4]，src_seq最内部表示单词索引。src_pos似乎应当再+1，才能保证每个词都有一个位置嵌入。 待完成）
src_pe_embedding（我的理解就是源序列的位置嵌入，这个东西和单词编码复合一下就可以组成最终嵌入。但具体如何复合还不清楚。 待完成）
src_embedding（它是源序列的嵌入格式，缺少了seq总觉了缺了点什么。
        长这样
        tensor([[[ 0.4676,  0.7180,  1.4003,  0.6132, -0.5129, -1.6297,  0.3025,
          -0.2349],
         [ 0.9293, -0.9396,  0.1066,  0.5660, -1.4424, -0.6596,  1.6842,
          -1.2658],
         [-1.2636,  0.7444, -1.3059, -0.6882,  1.7836,  0.4800,  0.5177,
           1.1534],
         [-1.2636,  0.7444, -1.3059, -0.6882,  1.7836,  0.4800,  0.5177,
           1.1534],
         [-1.2636,  0.7444, -1.3059, -0.6882,  1.7836,  0.4800,  0.5177,
           1.1534]],

        [[ 0.4676,  0.7180,  1.4003,  0.6132, -0.5129, -1.6297,  0.3025,
          -0.2349],
         [ 0.9956, -0.3683,  1.7343, -0.4362,  0.6208,  0.1986, -0.9280,
           1.0556],
         [ 0.1577, -0.9092,  0.0124,  1.1689,  0.5860,  0.2313,  0.7875,
           0.2474],
         [ 0.3811,  1.3722, -0.5217,  0.1411,  0.1492, -1.5178, -0.9420,
          -1.6662],
         [-1.2636,  0.7444, -1.3059, -0.6882,  1.7836,  0.4800,  0.5177,
           1.1534]]], grad_fn=<EmbeddingBackward0>)
        结合src_seq可以知道，这个就是源序列的具体嵌入，一共两个句子，第一个句子里面有5个词（词有可能是padding。），第二个句子也是5个词。
        
score（这个是相似度的结果。目前认为这个是Q乘K的结果，也就是认为这是某个词和整个序列上所有词的相似度的结果。）
softmax_func()（以score为参数）
（掩模矩阵的规模是t*t，内部的值为1或-inf。）
valid_encoder_pos（这是有效编码器的位置矩阵。再用unsqueeze给它扩维，就得到了2*1*4的张量。是表示源序列的有效位置的。）
valid_decoder_pos（这个反映了目标序列哪些位置是有效的，第一个句子全是1，那就全都有效，第二个句子最后一个是0，那就说明第二个句子的最后一个单词是pad。）
valid_cross_pos_matrix（这个是反映了目标序列对源序列的有效性的一个关系。它内部的第一个列表的第二行就是目标序列的第一句的第二个单词对源序列第一句各个单词的关系。）
invalid_cross_pos_matrix（这个里面1是代表无效的，0是代表有效的。）
valid_encoder_pos_matric（这个内部的第一行就是第一个句子的第一个单词和第一个句子的其他单词的有效性，1是因为这个位置存在单词，0是因为这个位置是pad。
    这也意味着着这个有效的位置构成的肯定是方阵，因为行数、列数都是该句子中有效单词的数量。encoder不是三角矩阵，decoder才是三角矩阵。这个有效矩阵就是两个矩阵两两相乘得到有效性。）
invalid_encoder_pos_matrix（这个无效矩阵就是有效矩阵取一个反得到的。）
mask_encoder_self_attention（这个就是算出来的mask矩阵。就是一个关系矩阵。True就代表这个位置需要mask，False就代表不需要mask。一会就是用这个对score进行mask。这个就是无效矩阵变成布尔型得到的。）
masked_score（把mask矩阵跟score进行主元素相乘，就得到masked_score。）
prob概率（prob表示这个词和整个序列上所有词的相似度的结果，但经过了归一化。这个就是注意力的权重。内部的值越大，就表示这个词和整个序列内某个词的相似度越大。
    当然这里还没有带入序列，只是一个句子。prob的时候如果acore*0.1，那方差还比较小，乘上10，方差就大了。对masked_score求一个softmax，就得到attention的prob。我们也看到，被mask的概率为0，但如果全被mask掉，那么概率相等。）
valid_decoder_tri_matrix（这个里面有2个张量，第一个是4*4的下三角矩阵，第二个是3*3的上三角矩阵。因为第一个目标序列长度为4，第二个目标序列长度为3.当解码器要预测第二个字符的时候，给的是第一个字符和一个特殊字符。）
prob（这个是注意力权重的矩阵，上面是第一个样本，下面是第二个样本。出来是
    tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
         [0.7199, 0.2801, 0.0000, 0.0000],
         [0.9644, 0.0191, 0.0165, 0.0000],
         [0.0647, 0.3978, 0.3487, 0.1889]],

        [[1.0000, 0.0000, 0.0000, 0.0000],
         [0.2237, 0.7763, 0.0000, 0.0000],
         [0.1197, 0.7620, 0.1183, 0.0000],
         [0.2500, 0.2500, 0.2500, 0.2500]]])
    第一个样本第一次解码的时候只注意到第一个单词，第二次解码的时候注意到前两个单词，第三词解码的时候注意到前三个单词。
    第二个样本最后一行都是0.25是因为它是pad。）
接着是19也就是六大难点上的6：52.（我觉得模型的代码就是那么一两行就没了，原理讲解这些人讲的也不清楚，现在的重点是我不清楚其他关于数据的代码是什么含义，以及多gpu怎么训练。）
'''