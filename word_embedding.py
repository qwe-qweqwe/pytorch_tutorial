import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
batch_size = 2
# 单词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8
# 序列的最大长度
max_src_seq_words = 5
max_tgt_seq_words = 5
src_len=torch.Tensor([2,4]).to(torch.int32)
tgt_len=torch.Tensor([4,3]).to(torch.int32)
# 以单词索引构成的句子。
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_words-L)),0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L,)),(0,max_tgt_seq_words-L)),0) for L in tgt_len])
src_embedding_table=nn.Embedding(max_num_src_words+1,model_dim)
tgt_embedding_table=nn.Embedding(max_num_tgt_words+1,model_dim)
src_embedding=src_embedding_table(src_seq)
tgt_embedding=tgt_embedding_table(tgt_seq)
print(src_embedding)