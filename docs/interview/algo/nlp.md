# NLP

1.1 Transformer
BERT是一种预训练的的方案，其创新之处更多在于数据处理与训练方式上的一些trick，再模型的设计上还是用的经典的Transformer结构。而Transformer是谷歌于2017年在那篇沸沸扬扬的《Attention is All you Need》中提出的，当时是用在机器翻译上的，取得了STOA的效果，其结构如下


主要由左边的Encoder与右边的Decoder两部分组成：

Encoder：用于编码，输入为每个位置的词的Embedding，由N层的encoder层组成。
Decoder：用于解码，由N层的decoder层组成，输入为Encoder的输出与当前层前一个位置的decoder输出，Decoder最终输出为当前位置每个词的概率。
无论是encoder层还是decoder层，都是是由Multi-Head Attention单元加全连接层组成的，并且这两个单元的input与output之间都加了Residual connection与Layer Normalization。最大的创新之处在于Multi-Head Attention单元，其由多个Self Attention组成。

1.2 Self Attention

对input信息经过三个矩阵进行线性变换得到Q、K与V三个矩阵，然后将Q与K进行点乘与尺度变换，再经过一个softmax变换得到一组权重，再用这组权重对V进行加权求和得到了每个位置的Attention向量。


1.3 Multi-Head Attention
多个head同时处理，每个head进行不同线性变换的，不同的head可以学习到token之间不同的空间依赖关系。


1.4 Transformer中的三种Attention
在Transformer中的Encoder与Decoder用的Attention并不完全相同，其区别主要在于输入信息的来源

Encoder Self-Attention：QKV均来自输入序列的Embedding，全局Attention。
Masked Decoder Self-Attention：第一级decoder，QKV均来自前一层decoder的输出，但加入了Mask操作，即只能attend到前面已经翻译过的输出的词语。
Encoder-Decoder Attention：位于decoder中第二级，Query来自于之前一级的decoder层的输出，但Key和Value来自于encoder的输出，全局Attention。

1.5 Position-wise feed-forward networks
主要是为了提供非线性变换，之所以是position-wise是因为过线性层时每个位置i的变换参数是一样的。但不同层的FFN参数不一样，并且使用的激活函数是GELU而不是RELU，即


相当于做了两次1x1的卷积操作，计算代码如下

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN = max(0, xw_1 + b_1)w_2 + b_2 equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
1.6 Positional Encoding
如果只是将Word Embedding输入网络中，那么Transformer只能算是一个精致的词袋模型，因为它虽然能处理很复杂的空间依赖关系，但并不能捕捉到词之间的次序信息，这对语义的表达非常重要。因此提出了Positional Encoding的方法，用于对词的次序信息进行编码。



2.1 BERT
BERT的全称是Bidirectional Encoder Representations from Transformers，它其实只是使用了Transformer中的Encoder，因为Encoder中的Self Attention可以捕捉全局信息，所以它是Bidirectional的。


在此之前GPT用的是Transformer中的Decoder部分用于对语言模型进行建模，它只是单向的模型。ELMo是双向的模型，只是使用的是LSTM单元，BERT可谓是结合了两者的长处。

2.2 训练目标
选择大于努力，在有了海量的预训练数据的情况下，所以学习目标的设计，对模型最终的performance至关重要，BERT的学习目标主要有两部分。

Mask-LM(Mask-Language Model)
一般的语言模型的建模方式是根据某个词之前的词序列来预测当前词，Mask-Language Model是将句子中待预测的词用[MASK]这个token替换掉，然后再将句子送入Transformer的Encoder中，最后在MASK掉的词的位置接一个Sofmax层用于输出该位置的应该填的词的概率。


关于如何构建MASK-LM的训练样本，BERT采用了如下Trick：

选取语料中所有词的15%进行随机mask
选中的词在80%的概率下被真实mask
选中的词在10%的概率下不做mask，而被随机替换成其他一个词
选中的词在10%的概率下不做mask，仍然保留原来真实的词
Next Sentence Prediction
为了能让模型学习到句子级别的语义信息，在模型的学习目标中添加了Next Sentence Prediction任务。其实就是预测两个句子是否是前后相邻的两个句子。

具体做法是将两个句子拼接到一起当成一个序列，然后输入Encoder，这就成了一个二分类问题。在构建训练样本时，数据中有50%的句子对是先后关系，另外50%句子对是随机从语料中凑到一起的，即负样本。

最终两个任务联合学习的损失函数是


2.3 BERT统一输入

为了能让BERT的输入既可以是单一的一个句子也可以是句子对，实际的输入BERT的输入词向量是三个向量之和：

Token Embedding：WordPiece tokenization subword词向量。
Segment Embedding：表明这个词属于哪个句子（NSP需要两个句子）。
Position Embedding：学习出来的embedding向量。这与Transformer不同，Transformer中是预先设定好的值。

2.4 使用方法
当花了很大成本完了BERT的预训练后，如何利将其迁移到特定任务上呢？BERT的作者将常见的NLP任务分为四类并给出了相应的使用方案。

单个句子分类：在句子前面加上[CLS]这个token，然后将句子输入Encoder，将[CLS]所在位置的hidden output作为整句话的表征，输入Softmax层进行分类。
句子对分类：与单个句子的分类类似，只是把两个句子拼接成一个句子，句子之间加一个[SEP]的token。
问答类：这类任务是找出答案在文本中的区间，其实就是确定答案的开始位置与终止位置，建模的方式就是维护两个向量即起始符的向量和终止符的向量，然后用这两个向量与所有位置的hidden output做点乘，取分数最高的地方作为起始位置与终止位置
序列标注：其实就是对每个位置的hidden output接一个softmax层做分类，这点比HMM与CRF等需要做动态规划计算的方式简单多了。

3 面试考点
Attention结构有什么优点
一步到位捕捉全局与局部的联系：一步到位灵活地捕捉全局与局部的relevance信息，而且不存在信息的链式传递，可以很好的处理长距离依赖关系。Attention函数是将序列中的每个元素与其他元素的对比，每两个元素间（Query与Key）的距离都是1。而RNNs通过一步步递推得到长期依赖关系好的多，越长的序列RNN能捕捉到的长期依赖关系就越弱。
并行计算减少模型训练时间：Attention机制每一步计算不依赖于上一步的计算结果，可以像CNN一样并行处理。但CNN每次只能捕捉局部信息，再通过层叠来扩大视野获取全局的联系。
模型复杂度小，参数少：模型复杂度是与CNN和RNN同条件下相比较的。
position embeddig的作用是什么
对子中token次序信息进行编码

Residual Connection的作用是什么
减缓梯度衰减，加快收敛

加入LayerNorm层有什么好处
当使用梯度下降法做优化时，随着网络深度的增加，数据的分布会不断发生变化，加入Layer Normalization可以提高数据特征分布的稳定性，从而加速模型的收敛速度

BERT的缺点有哪些
MASK标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现;
每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）；
BERT对硬件资源的消耗巨大。
加入Next Sentence Prediction任务的目的是什么
获得句子级的语义表征，对问答、推理、句⼦关系类的NLP任务帮助非常大。

Mask-LM的样本中，选中的词在10%的概率不做Mask保持真实的词的原因是什么
给模型一定的bias，相当于是额外的奖励，将模型对于词的表征能够拉向词的真实表征

Mask-LM的样本中，选中的词在10%的概率下不做mask，而是被随机替换成为一个其他词的目的是什么
因为模型不知道哪些词是被mask的，哪些词是mask了之后又被替换成了一个其他的词，这会迫使模型尽量在每一个词上都学习到一个全局语境下的表征，因而也能够让BERT获得更好的语境相关的词向量，提升模型的鲁棒性。

为什么即便数量很小，基于BERT做微调也能取得很好的泛化效果
这个问题最直观的解释是BERT提供了以恶好的起点，模型的训练是站在巨人的肩膀上。但更恰当的解释是用BERT初始化模型，相当于提供了一种正则化，即便数据量很少也不容易过拟合。有了BERT做初始化，用少量的高质量数据可以训练出比大量劣质数据更好的模型。