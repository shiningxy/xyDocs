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

# 3 面试考点

## Attention结构有什么优点

一步到位捕捉全局与局部的联系：一步到位灵活地捕捉全局与局部的relevance信息，而且不存在信息的链式传递，可以很好的处理长距离依赖关系。Attention函数是将序列中的每个元素与其他元素的对比，每两个元素间（Query与Key）的距离都是1。而RNNs通过一步步递推得到长期依赖关系好的多，越长的序列RNN能捕捉到的长期依赖关系就越弱。
并行计算减少模型训练时间：Attention机制每一步计算不依赖于上一步的计算结果，可以像CNN一样并行处理。但CNN每次只能捕捉局部信息，再通过层叠来扩大视野获取全局的联系。
模型复杂度小，参数少：模型复杂度是与CNN和RNN同条件下相比较的。

## position embeddig的作用是什么

对子中token次序信息进行编码

## Residual Connection的作用是什么

减缓梯度衰减，加快收敛

## 加入LayerNorm层有什么好处

当使用梯度下降法做优化时，随着网络深度的增加，数据的分布会不断发生变化，加入Layer Normalization可以提高数据特征分布的稳定性，从而加速模型的收敛速度

## BERT的缺点有哪些

MASK标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现;
每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）；
BERT对硬件资源的消耗巨大。

## 加入Next Sentence Prediction任务的目的是什么

获得句子级的语义表征，对问答、推理、句⼦关系类的NLP任务帮助非常大。

## Mask-LM的样本中，选中的词在10%的概率不做Mask保持真实的词的原因是什么

给模型一定的bias，相当于是额外的奖励，将模型对于词的表征能够拉向词的真实表征

## Mask-LM的样本中，选中的词在10%的概率下不做mask，而是被随机替换成为一个其他词的目的是什么

因为模型不知道哪些词是被mask的，哪些词是mask了之后又被替换成了一个其他的词，这会迫使模型尽量在每一个词上都学习到一个全局语境下的表征，因而也能够让BERT获得更好的语境相关的词向量，提升模型的鲁棒性。

## 为什么即便数量很小，基于BERT做微调也能取得很好的泛化效果

这个问题最直观的解释是BERT提供了以恶好的起点，模型的训练是站在巨人的肩膀上。但更恰当的解释是用BERT初始化模型，相当于提供了一种正则化，即便数据量很少也不容易过拟合。有了BERT做初始化，用少量的高质量数据可以训练出比大量劣质数据更好的模型。



## Question: 如何使用langchain为语言大模型添加记忆？
Answer:
Langchain通过Memory类为语言大模型添加记忆。Langchain提供了两种形式的记忆组件。一种是聊天消息历史，大多数内存模块的核心实用类都是ChatMessageHistory类，该类中的会话缓存记忆包装器（ConversationBufferMemory）允许存储消息，并将消息提取到一个变量中。另一种是保存消息历史，允许将消息转换为普通的python字典，保存这些字典，然后加载它们来实现记忆。


## Question: langchain中具体实现了哪几种记忆函数？
Answer:
五种，分别是会话缓存记忆（ConversationBufferMemory）、会话缓存窗口记忆（ConversationBufferWindowMemory）、实体记忆（ConversationEntityMemory）、摘要记忆（ConversationSummaryMemory）、向量存储检索记忆（VectorStoreRetrieverMemory）

## Question: 解释langchian中摘要记忆（ConversationSummaryMemory）的原理
Answer:
摘要记忆ConversationSummaryMemory。这种记忆类型会随着时间的推移对对话进行摘要。这对于从对话中压缩信息非常有用。摘要记忆会即时总结对话并将当前摘要存储在记忆中。然后可以将这个记忆用于将到目前为止的对话摘要注入到提示/链中。将过去的消息历史原样保留在提示中会占用太多的标记。摘要记忆则对较长的对话有明显优势。
在ConversationSummaryMemory类继承了SummarizerMixin类。通过Mixin摘要器中的predict_new_summary方法，来实现获取历史人类消息和历史AI消息并进行摘要的功能。


## Question: Git中 fork、branch、clone的区别？
Answer: fork不属于git命令，是远程代码托管平台提供的一种操作。对于远程代码仓库，通过fork操作，可以得到一个该远程仓库的副本，基于该副本，可以实现新功能的开发、Code Review等，而不对原远程仓库产生任何影响。
branch意为分支，git branch是git的一种命令，命令结果是建立一个新分支。
git clone是git的一种命令，它的作用是将文件从远程代码仓库下载到本地，从而形成一个本地代码仓库。


## Question: pull和fetch的区别？
Answer: pull = fetch + merge。

## Question: 解释一下冒泡排序的原理。
Answer: 冒泡排序是一种简单的排序算法，它重复地遍历待排序的元素序列，比较相邻元素并交换它们的位置，直到整个序列有序。通过多次遍历，每次将最大的元素“冒泡”到序列的末尾。

## Question: 快速排序是如何工作的？
Answer:快速排序是一种高效的排序算法。它选择一个基准元素，将序列分割为两个子序列，其中一个子序列的元素都小于基准元素，另一个子序列的元素都大于基准元素。然后对子序列递归地应用相同的排序过程，直到每个子序列只包含一个元素或为空。

## Question: 解释一下堆排序（Heap Sort）算法。
Answer:堆排序是一种基于堆数据结构的排序算法。它首先将待排序的元素构建成一个最大堆（或最小堆），然后依次将堆顶元素（最大值或最小值）与堆的最后一个元素交换，并对剩余元素重新进行堆调整，直到所有元素有序。堆排序的时间复杂度为O(n log n)。

## Question: 解释一下图的广度优先搜索（BFS）算法。
Answer:广度优先搜索是一种用于图的遍历的算法。它从图中的某个节点开始，首先访问该节点，然后逐层访问与当前节点相邻的未访问节点，直到遍历完所有可达节点。广度优先搜索通常使用队列来辅助实现。

## Question: 解释一下图的深度优先搜索（DFS）算法。
Answer:深度优先搜索是一种用于图的遍历的算法。它从图中的某个节点开始，首先访问该节点，然后递归地访问与当前节点相邻的未访问节点，直到到达最深的节点，然后回溯到上一层继续遍历其他节点。

## Question: 什么是递归算法？它有什么特点和优缺点？
Answer:递归算法是一种通过调用自身来解决问题的算法。它的特点是问题可以被分解为规模较小的子问题，并且这些子问题与原问题具有相同的解决方法。递归算法的优点是思路清晰，代码简洁易懂；缺点是递归调用会增加额外的函数调用开销，可能导致栈溢出等问题。

## Question: 什么是贪心算法（Greedy Algorithm）？它有什么特点和应用场景？
Answer:贪心算法是一种通过每一步选择局部最优解来构造问题的解决方法。它不一定能得到全局最优解，但通常具有高效性和简单性。贪心算法适用于满足贪心选择性质和最优子结构性质的问题，例如最小生成树、背包问题和任务调度等。

## Question: 请解释图（Graph）的概念和常见的表示方式。
Answer: 图是由节点（顶点）和边组成的非线性数据结构。常见的图的表示方式有两种：

邻接矩阵：使用二维矩阵表示图的连接关系，其中矩阵的行和列分别代表节点，矩阵中的元素表示节点之间的边的存在与否。
邻接表：使用哈希表或数组的列表表示图的连接关系，其中列表中的每个元素表示一个节点，其对应的值是与该节点直接相连的节点列表。


## Question: 索引的优缺点？
Answer: 优点：
提高数据的检索速度，降低数据库IO成本：使用索引的意义就是通过缩小表中需要查询的记录的数目从而加快搜索的速度。
降低数据排序的成本，降低CPU消耗：索引之所以查的快，是因为先将数据排好序，若该字段正好需要排序，则正好降低了排序的成本。
缺点：
占用存储空间：索引实际上也是一张表，记录了主键与索引字段，一般以索引文件的形式存储在磁盘上。
降低更新表的速度：表的数据发生了变化，对应的索引也需要一起变更，从而减低的更新速度。否则索引指向的物理数据可能不对，这也是索引失效的原因之一。


## Question: 索引的类型？
Answer:索引，都是实现在存储引擎层的。主要有六种类型：

普通索引：最基本的索引，没有任何约束。
唯一索引：与普通索引类似，但具有唯一性约束。
主键索引：特殊的唯一索引，不允许有空值。
复合索引：将多个列组合在一起创建索引，可以覆盖多个列。
外键索引：只有InnoDB类型的表才可以使用外键索引，保证数据的一致性、完整性和实现级联操作。
全文索引：MySQL 自带的全文索引只能用于 InnoDB、MyISAM ，并且只能对英文进行全文检索，一般使用全文索引引擎。

## Question: 索引创建的原则
Answer: 最适合索引的列是出现在 WHERE 子句中的列，或连接子句中的列，而不是出现在 SELECT 关键字后的列。
索引列的基数越大，索引效果越好。
根据情况创建复合索引，复合索引可以提高查询效率。
避免创建过多的索引，索引会额外占用磁盘空间，降低写操作效率。
主键尽可能选择较短的数据类型，可以有效减少索引的磁盘占用提高查询效率。
对字符串进行索引，应该定制一个前缀长度，可以节省大量的索引空间。

## Question: 相比较于llama而言，llama2有哪些改进，对于llama2是应该如何finetune？
Answer: llama和llama2都是一种大型语言模型（Large Language Model，LLM），它们可以用于多种自然语言处理的任务，如文本生成、文本摘要、机器翻译、问答等。llama是一种基于Transformer的seq2seq模型，它使用了两种预训练任务，一种是无监督的Span级别的mask，另一种是有监督的多任务学习。llama将所有的下游任务都视为文本到文本的转换问题，即给定一个输入文本，生成一个输出文本。llama使用了一个干净的大规模英文预料C4，包含了约750GB的文本数据。llama的最大规模达到了11B个参数。llama2是llama的改进版本，它在以下几个方面有所提升：

数据量和质量：llama2使用了比llama1多40%的数据进行预训练，其中包括更多的高质量和多样性的数据，例如来自Surge和Scale等数据标注公司的数据。
上下文长度：llama2的上下文长度是llama1的两倍，达到了4k个标记，这有助于模型理解更长的文本和更复杂的逻辑。
模型架构：llama2在训练34B和70B参数的模型时使用了分组查询注意力（Grouped-Query Attention，GQA）技术，可以提高模型的推理速度和质量。
微调方法：llama2使用了监督微调（Supervised Fine-Tuning，SFT）和人类反馈强化学习（Reinforcement Learning from Human Feedback，RLHF）两种方法来微调对话模型（llama2-chat），使模型在有用性和安全性方面都有显著提升。
对llama2进行微调有以下步骤：

准备训练脚本：你可以使用Meta开源的llama-recipes项目，它提供了一些快速开始的示例和配置文件，以及一些自定义数据集和策略的方法。
准备数据集：你可以选择一个符合你目标任务和领域的数据集，例如GuanacoDataset，它是一个多语言的对话数据集，支持alpaca格式。你也可以使用自己的数据集，只要按照alpaca格式进行组织即可。
准备模型：你可以从Hugging Face Hub下载llama2模型的权重，并转换为Hugging Face格式。
启动训练：你可以使用单GPU或多GPU来进行训练，并选择是否使用参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）或量化等技术来加速训练过程。具体命令可以参考这里。

## Question: transformer 为什么使用 layer normalization，而不是batch normalization？
Answer: 首先:如果在一个维度内进行normalization，那么在这个维度内，相对大小有意义的，是可以比较的；但是在normalization后的不同的维度之间，相对大小这是没有意义的。

BN(batch normalization)广泛应用于CV，针对同一特征，以跨样本的方式开展归一化，也就是对不同样本的同一channel间的所有像素值进行归一化，因此不会破坏不同样本同一特征之间的关系，
NLP中对不同样本同一特征的信息进行归一化没有意义：
举例三个样本（为中华之崛起而读书；我爱中国；母爱最伟大）中，“为”、“我”、“母”归一到同一分布没有意义。
舍弃不了BN中舍弃的其他维度的信息，也就是同一个样本的不同维度的信息：
“为”、“我”、“母”归一到同一分布后，第一句话中的“为”和“中”就没有可比性了，何谈同一句子之间的注意力机制？

最根本的不同即BatchNorm和LayerNorm的作用对象不同——BatchNorm认为相同维的特征具有相同分布，因此在特征维度上开展归一化操作，归一化的结果保持样本之间的可比较性。而LayerNorm认为每个样本内的特征具有相同分布，因此针对每一个样本进行归一化处理，保持相同样本内部不同对象的可比较性。由于上述根本差异的存在，引出了一系列使用方法的不同：BatchNorm在批次中执行跨样本的归一化操作，这就意味着批次的构成和规模会直接影响BatchNorm的效果。BatchNorm需要平衡小批次统计量和整体样本统计量之间的关系，还需要考虑利用批次统计量更新全局统计量的方法，这也涉及训练和测试阶段使用的统计量有“批次版”和“全局版”的问题…等等。而这些问题到了LayerNorm就都不再是问题——LayerNorm的归一化操作只在样本内部独立开展，因此实际可以完全忽略批次的存在。因此也不用考虑保存和更新的问题且训练和测试应用模式完全一致，均值和标准差随算随用。

## Question: 为什么Bert中要用WordPiece/BPE这样的subword Token？

Answer:避免OOV（Out Of Vocabulary），也就是词汇表外的词。在NLP中，通常会预先构建一个词汇表，包含所有模型能够识别的词。然而，总会有一些词没有出现在预先构建的词汇表中，这些词就是 OOV。
传统的处理方式往往是将这些 OOV 映射到一个特殊的符号，如 <UNK>，但这种方式无法充分利用 OOV 中的信息。例如，对于词汇表中没有的词 "unhappiness"，如果直接映射为 <UNK>，则模型就无法理解它的含义。
WordPiece/Byte Pair Encoding (BPE) 等基于子词的分词方法提供了一种解决 OOV 问题的方式。现在更多的语言大模型选择基于BPE的方式，只不过BERT时代更多还是WordPiece。BPE 通过将词分解为更小的单元（子词或字符），可以有效地处理词汇表外的词。对于上面的 "unhappiness" 例子，即使 "unhappiness" 本身不在词汇表中，但是它可以被分解为 "un"、"happiness" 等子词，而这些子词可能在词汇表中。这样，模型就可以通过这些子词来理解 "unhappiness" 的含义。
另一方面就是，BPE本身的语义粒度也很合适，一个token不会太大，也不会小到损失连接信息（如一个字母）。

## Question: Bert中有哪些地方用到了mask?
Answer: 预训练任务Masked Language Model (MLM)
self-attention的计算
下游任务的decoder