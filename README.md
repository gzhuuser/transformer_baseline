# transformer_baseline
Datawhale夏令营机器翻译挑战赛基于transformer实现的代码



# transformer基础
在2017年，一篇划时代的论文《Attention Is All You Need》发布，极大地推动了自然语言处理（NLP）领域的进展。自此以后，Transformer模型不仅在文本生成中得到了广泛应用，还在扩散模型等多个领域显示出了其强大的能力。接下来，我们将详细介绍Transformer的整体框架。
![image](https://github.com/user-attachments/assets/ade4eba0-c70a-4d9e-8b9f-0a25e0a144d1#pic_center)

如上图:我们可以看到Transformer架构分为四个部分,分别是输入部分,编码器部分,解码器部分,输出部分。
- 输入部分：包括编码器输入和解码器输入。对于编码器输入，我们首先将文本通过词嵌入（Word Embedding）和位置嵌入（Position Embedding）转换为向量，然后送入编码器以提取特征。解码器输入则在编码器的基础上增加了一个掩码机制，该机制用于屏蔽未来的信息，防止信息提前泄露。

- 编码器部分：根据Transformer的原始论文，编码器由六个结构相同的编码层堆叠而成。这里我们重点分析单个编码层的结构。从结构图中可以看出，每个编码器层包括两个子模块：多头自注意力（Multi-Head Self-Attention）和前馈全连接层（Feed-Forward Neural Network）。每个子模块都是通过一个残差连接和随后的层归一化（Layer Normalization）来实现的。

- 解码器部分：解码器同样由六个结构相同的解码层堆叠而成，构成整个解码器。在解码器的每个层中，有三个主要的子模块：多头自注意力层、编码器-解码器注意力层（用于从编码器层抽取信息），以及前馈全连接层。与编码器类似，这些子模块也采用残差连接和层归一化。

- 输出部分：最后，输出通过一个全连接层进行特征提取，并通过Softmax函数生成最终的预测结果


我们以机器翻译为例子理解一下Transformer的训练全过程:
> 假设我们的任务是将我爱北京翻译成I love Beijing
> 1. 预处理和词嵌入
输入处理：首先，中文句子“我爱北京”会被分词为单独的词或字符。假设我们使用字符级的分割，得到“我”、“爱”、“北”、“京”。这一部分是由tokenizer完成的
词嵌入：这些字符通过词嵌入层转换成向量。此外，由于Transformer不具备处理序列顺序的能力，我们还需为每个字符添加位置嵌入，以表示其在句子中的位置。对应结构中的Embedding
>2. 编码器操作
多头自注意力机制(Multi-Head Self-Attention)：在编码器中，多头自注意力层会评估句子中每个字符与其他字符的关系，这有助于捕获例如“我爱”（我和爱之间的直接关系）这样的局部依赖关系。
前馈全连接层机制(Feed-Forward Neural Network): 经过大量的实验表面,全连接层的特征提取能力是很强的,而且结构简单,为了防止多头注意力机制特征提取不够充分,所有加入了这一层,让模型进一步学习到词语词之间的依赖关系
层次结构处理：编码器的每一层都将之前层的输出作为输入，逐层提取更抽象的特征。每个层的输出都是一个加强了输入句子每个部分上下文信息的表示。
>3. 解码器操作
屏蔽未来信息：解码器在生成翻译时使用屏蔽技巧来避免“看到”未来的输出。例如，在预测单词“love”时，模型只能访问到“<start> I”，而不能访问到“Beijing”。
注意力机制：解码器的编码器-解码器注意力层使得每一步的生成都可以关注到输入句子的不同部分。例如，当生成“Beijing”时，模型可能会特别关注“北京”。
>4. 生成预测和训练
输出：每次解码步骤，模型都会输出一个词的概率分布，选择概率最高的词作为这一位置的翻译。例如，首先生成“I”，然后是“love”，最后是“Beijing”。
>5. 训练过程：在训练阶段，我们使用实际的目标句子“<start> I love Beijing <end>”作为训练目标。模型通过比较预测输出与实际输出的差异,计算出损失值，并通过反向传播优化其参数。

# 数据预处理
在所有深度学习的数据预处理部分,我们都可以用一句话概括。那就是将数据处理成x和y的形式。x是可以让模型识别到的输入结果, y是目标结果。如下图:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/348f2e84cad6471d9d156ceaf59f5ab5.png#pic_center)
对于我们这个数据集来说,首先要明确的是哪个是x,哪个是y。

根据官方的赛事说明,我们这个是一个英译中的任务。所以我们很容易的可以知道,x是英语数据,y是中文数据。

然后就开始了我们的数据清洗之路了,我大概分为了以下几个步骤:
1. 读取并划分数据
2. 对数据进行清洗
3. 针对术语字典做一些特殊处理
4. 构建分词器
5. 保存处理好后的数据集

## 数据读取和划分
这一部分,我们需要去读取我们训练的数据集。我们的训练集是==英文\t中文==的形式, 所以可以直接按行读取,然后按制表符划分。代码如下:
```python
# 读取并处理数据
with open("./data/train.txt", 'r', encoding='utf-8') as f:
    data = f.readlines()
    en_data = [preprocess_en(line.strip().split('\t')[0]) for line in data]
    zh_data = [preprocess_zh(line.strip().split('\t')[1]) for line in data]
```
这里我们读取完后还会对每一行的数据做清洗,清洗函数看下面一部分
## 数据清洗
拿到数据后,我们首先要做的就是先观察数据。我们打开train.txt可以看到
> There’s a tight and surprising link between the ocean’s health and ours, says marine biologist Stephen Palumbi. He shows how toxins at the bottom of the ocean food chain find their way into our bodies, with a shocking story of toxic contamination from a Japanese fish market. His work points a way forward for saving the oceans’ health – and humanity’s.
生物学家史蒂芬·帕伦认为，海洋的健康和我们的健康之间有着紧密而神奇的联系。他通过日本一个渔场发生的让人震惊的有毒污染的事件，展示了位于海洋食物链底部的有毒物质是如何进入我们的身体的。他的工作主要是未来拯救海洋健康的方法——同时也包括人类的。

There’s这些,如果我们直接构建词表的话,有可能出现分词为’的情况。所以我们要将这些There’s变成There is。
除此之外,我们要删除一些特殊字符,只保留一些标点符号和数字等。代码如下:

```python
import contractions

def unicodeToAscii(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def preprocess_en(text):
    text = unicodeToAscii(text.strip())
    text = contractions.fix(text)
    text = re.sub(r'\（[^）]*\）', '', text)
    text = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", text)  # 保留数字
    return text
```
处理完后的效果:
>There is a tight and surprising link between the ocean s health and ours says marine biologist Stephen Palumbi . He shows how toxins at the bottom of the ocean food chain find their way into our bodies with a shocking story of toxic contamination from a Japanese fish market . His work points a way forward for saving the oceans health and humanity s .

可以看出There’s 已经变成了There is了

接着对中文数据进行处理。在中文数据中,经过探查,竟然发现有(掌声)这种不该出现在翻译文本中的脏数据。比如:

> 他指着我碗底的三粒米， 然后说"吃干净。" (笑声）
他说，“如果你要回你的车，那么我就要tase(用高压眩晕枪射击）你
Okay. Good. 好，很好!(笑）
But many people see the same thing and think things differently, and one of them is here, Ratan Tata. 看到的是同样的东西， 但很多人的想法却不一样， 其中一个就是，Ratan Tata (Tata集团的现任主席）。

这些脏数据可以使用正则表达式剔除,代码如下:

```python
def preprocess_zh(text):
    # 去除(掌声)这些脏数据
    text = re.sub(r'\（[^）]*\）', '', text)
    text = re.sub(r"[^\u4e00-\u9fa5，。！？0-9]", "", text)  # 保留数字
    return text
```
这一步操作虽然会删除一些可能真的需要()翻译的内容,但是也是小部分,比如:

> Kary Mullis: They might have done it for the teddy bear, yeah. (Kary Mullis回答：）那他们可能也会吧。


然后这是对数据内容处理的部分。


接下来，我们要做的是找到一个合适的截断长度。为了在后续的训练过程中能够方便地进行批量训练，我们需要将传入训练的文本统一在一个固定的长度上。那么，我们该选择多长作为我们的截断长度呢？

有些小伙伴可能会说，可以创建一个DataFrame格式的表格，统计文本长度，然后画个图进行分析。这确实是一种方法，但在这里，我将介绍一个更便捷的小技巧，让大家可以跳过分析长度这一步，直接得到一个合理的截断长度。

这个技巧的原理其实很简单，使用的是正态分布的3σ原则。我们可以对整体数据集的长度进行正态分布分析，取2σ的值，这样可以确保95%的数据不会因为截断而造成内容的缺失，还能规避一些异常值。从而得到一个合理的截断长度。

通过这种方法，我们不仅可以快速确定截断长度，还能确保大多数数据都能被有效地利用，提高模型的训练效果和效率。

代码如下:
```python
import numpy as np
# 计算长度
en_lengths = [len(d_e) for d_e in en_data]
zh_lengths = [len(d_z) for d_z in zh_data]

# 计算平均值和标准差
en_mean = np.mean(en_lengths)
en_std = np.std(en_lengths)
zh_mean = np.mean(zh_lengths)
zh_std = np.std(zh_lengths)

# 使用正态分布的 1σ 原则计算最大长度
SRC_MAX_LEN = int(en_mean + 2 * en_std)
TGT_MAX_LEN = int(zh_mean + 2 * zh_std)

print(f"SRC_MAX_LEN: {SRC_MAX_LEN}")
print(f"TGT_MAX_LEN: {TGT_MAX_LEN}")
```
得到结果为:
> SRC_MAX_LEN: 227
TGT_MAX_LEN: 70

我们就可以将这个作为我们输入数据的最大截断长度,和输出数据的最大截断长度


执行代码:
```python
with open("./data/train.txt", 'r', encoding='utf-8') as f:
    data = f.readlines()
    en_data = [preprocess_en(line.strip().split('\t')[0]) for line in data]
    zh_data = [preprocess_zh(line.strip().split('\t')[1]) for line in data]
```

## 针对术语字典做一些特殊处理
和群友讨论过后在这里我做了两种尝试:
1. 对特殊词典加入特殊符号如<|sword|>special_token<|eword|>试图让大模型认识这种格式,看到这个以后就知道这个是一个特殊的单词,需要重点翻译
2. 第二种,直接将特殊词表扔进训练集中,进行训练。让大模型看到这个单词就知道这个单词的中文意思是什么

但是效果都有点不理想的样子,还不如不做处理正常翻译(很难不让人怀疑这个词典是一个坑),可能还有其他更好的处理方法还没想出来。

这里给出我添加特殊字符的代码,在分词的过程中顺便添加了特殊字符上去,让那个token变成<|sword|>token<|eword|>的形式。
```python
# 添加特殊符号
def add_split_symbols(tokens, special_dict):
    return ['<|sword|>' + token + '<|eword|>' if token in special_dict else token for token in tokens]

```

测试效果:
>  测试自定义的分词和添加特殊符号功能
test_sentence_en = "Oxford philosopher and transhumanist Nick Bostrom examines the future of humankind and asks whether we might alter the fundamental nature of humanity to solve our most intrinsic problems."
token_transform[SRC_LANGUAGE](test_sentence_en)

> ['Oxford',
 'philosopher',
 'and',
 'transhumanist',
 '<|sword|>Nick<|eword|>',
 'Bostrom',
 'examines',
 'the',
 '<|sword|>future<|eword|>',
 'of',
 'humankind',
 'and',
 'asks',
 'whether',
 'we',
 'might',
 'alter',
 'the',
 'fundamental',
 'nature',
 'of',
 '<|sword|>humanity<|eword|>',
 'to',
 'solve',
 'our',
 'most',
 'intrinsic',
 'problems.']


直接将特殊词表引入进去原来的词典
```python
en_data[len(en_data)-len(dic)]
en_data_ = []
for i in range(len(en_data)):
    if i < len(en_data)-len(dic):
        en_data_.append(en_data[i])
    else:
        en_data_.append('<|sword|>'+en_data[i]+'<|eword|>')

en_data = en_data_
```
这里的en_data就会加入我们的特殊词了。


这一部分比较开放,欢迎大家不断地尝试。
## 构建分词器
构建分词器这一部分,我直接从简了,英语按空格划分,中文用jieba分词。代码和上面的一样,这里就不做重复粘贴了。
```python
# 加载训练和验证数据
train_src_file = './data/train.en'  
train_tgt_file = './data/train.zh'  

valid_src_file = './data/dev_en.txt'  
valid_tgt_file = './data/dev_zh.txt'  

train_src_data = read_data(train_src_file)
train_tgt_data = read_data(train_tgt_file)

valid_src_data = read_data(valid_src_file)
valid_tgt_data = read_data(valid_tgt_file)

train_data = data_iterator(train_src_data, train_tgt_data)
valid_data = data_iterator(valid_src_data, valid_tgt_data)




# 定义词表
token_transform = {}
vocab_transform = {}

# 定义分词器
token_transform[SRC_LANGUAGE] = lambda x: add_split_symbols(x.split(' '), dic_en_zh)
token_transform[TGT_LANGUAGE] = lambda x: list(jieba.cut(x))

# 定义特殊字符以及它们在词汇表中的索引
# UNK_IDX：未知词的索引
# PAD_IDX：填充词的索引
# BOS_IDX：句子开始符的索引
# EOS_IDX：句子结束符的索引
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SWORD_IDX, EWORD_IDX = 0, 1, 2, 3, 4, 5
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', '<|sword|>', "<|eword|>"]

# 构建 vocab_transform
# vocab_transform 是一个字典，用于存储源语言和目标语言的词汇表
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train_data, ln),  # 从数据迭代器中生成分词结果
        min_freq=1,  # 词汇表中的词必须至少出现1次
        specials=special_symbols,  # 特殊符号列表
        special_first=True  # 将特殊符号放在词汇表的前面
    )
print(vocab_transform)

# 将unk设置为默认字符
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)
```

## 保存数据集
对这些数据保存下来,方便下次复用。因为每次分词都要大概花不少时间,所以记录下来下次就可以直接加载了。

代码如下:
```python
with open('./data/train.en', 'w', encoding='utf-8') as f:
    for line in en_data:
        f.write(line+"\n")
        
with open('./data/train.zh', 'w', encoding='utf-8') as f:
    for line in zh_data:
        f.write(line+"\n")
```

# 模型构建
transformer可以看作四个模块
1. Embedding词嵌入模块
2. Encoder模块
3. Decoder模块
4. 输出模块

接下来我们来定义一下:
## 位置编码
由于我们RNN这些模型是一个一个输入进去的,本身自带位置顺序。但是transformer为了实现并行运算,他是一次性输入进去的。这时候就会损失掉位置信息,所以我们要定义一个位置编码来引入这个位置信息。
```python
# 定义位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        # 填充
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # 变成三维, 方便后期计算
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # 将token_embedding和位置编码相融合
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        # 调用nn中的预定义层Embedding, 获取一个词嵌入对象self.embedding
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # 让 embeddings vector 在增加 之后的 position encoding 之前相对大一些的操作，
        # 主要是为了让position encoding 相对的小，这样会让原来的 embedding vector 中的信息在和 position encoding 的信息相加时不至于丢失掉
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


```

## encoder-decoder模块和输出模块
这一部分就是transformer经典架构了
```python
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        # 创建Transformer对象
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        # 创建全连接线性层
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # 创建源语言的embedding层
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # 创建目标语言的embedding层
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # 创建位置编码器层对象
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

```


## 定义一些辅助函数
```python
# 生成一个上三角矩阵掩码，用于目标序列
def generate_square_subsequent_mask(sz):
    # 生成一个sz x sz的上三角矩阵，值全为1
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    # 将上三角矩阵中的0位置的值替换为负无穷大，将1位置的值替换为0,因为在transform库中的掩码是对0为非遮掩部分
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    # 生成目标序列的掩码
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # 源序列的掩码，全为0
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    # 源序列和目标序列的填充掩码，标记出填充位置
    # 这里转置的原因是:
    # src和tgt的shape是(seq_len, batch_siez), 通过转置后,我们的src_padding_mask为(batch_size, seq_len)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
```

## 定义回调函数
```python
# 数据批处理函数，用于DataLoader
def collate_fn(batch):
    """python
    [('Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
  'Two young, White males are outside near many bushes.'),.....]
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        # 对源语言和目标语言的句子进行转换处理
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    # 对源语言和目标语言的批次进行填充
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
```


# 模型训练

## 定义训练函数和验证函数
```python
def train_epoch(model, optimizer, dataloader):
    model.train()
    losses = 0
    for src, tgt in tqdm(dataloader, desc="Training", leave=False):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # 这一步将目标序列的最后一个时间步去掉，得到 tgt_input。这是因为在训练过程中，我们使用目标序列的前 T个时间步。
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:].to(torch.long)
        logits = logits.to(torch.float32)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    losses = 0
    for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(dataloader)
```


定义配置参
数,然后开始训练!
```python
BATCH_SIZE = 16
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
# 设置种子用于生成随机数，以使得结果是确定的
torch.manual_seed(0)

# 设置调用时候使用的参数
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# 实例化Transformer对象
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
# 为了保证每层的输入和输出的方差相同, 防止梯度消失问题
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# 如果有GPU则将模型移动到GPU上
transformer = transformer.to(DEVICE)
# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# 定义优化器  betas: 用于计算梯度及其平方的运行平均值的系数  eps:添加到分母以提高数值稳定性
"""
betas 是 Adam 优化器中两个超参数的元组，用于计算一阶和二阶矩估计的指数衰减率。
第一个值 0.9 是用于计算梯度的一阶矩（即动量）的衰减率。较高的值表示动量更大，历史梯度的影响更长久。
第二个值 0.98 是用于计算梯度的二阶矩（即平方梯度）的衰减率。较高的值表示对最近梯度变化的敏感度更低。
"""
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 创建数据加载器
NUM_EPOCHS = 3

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, train_dataloader)
    end_time = timer()
    val_loss = evaluate(transformer, valid_dataloader)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")



```

## 保存模型
```python

# 模型保存和加载
path = './model/transformer_translation_5.pth'
torch.save(transformer.state_dict(), path)

# 加载模型
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
transformer.load_state_dict(torch.load(path))
```





# 模型推理
## 贪心解码
```python
# 贪婪解码函数，用于从模型中生成翻译结果
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 将输入数据和掩码移动到设备上
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    
    # 编码器对源序列进行编码
    memory = model.encode(src, src_mask)
    
    # 初始化目标序列，以开始符号开始
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    
    # 逐步生成目标序列
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        
        # 生成目标序列掩码
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        # 解码器对目标序列进行解码
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        
        # 生成下一个词的概率分布
        prob = model.generator(out[:, -1])
        
        # 选择概率最高的词作为下一个词
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # 将下一个词添加到目标序列中
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        
        # 如果生成结束符，则停止生成
        if next_word == EOS_IDX:
            break
    
    # 返回生成的目标序列
    return ys

```

## 翻译函数
```python

# 翻译函数，将源语言句子翻译成目标语言句子
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()  # 设置模型为评估模式
    
    # 将源语言句子进行分词、数值化和tensor转换
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    src = src.to(DEVICE)
    # 获取源序列的长度
    num_tokens = src.shape[0]
    
    # 创建源序列掩码，全为0
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    # 使用贪婪解码生成目标语言句子
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    
    # 将生成的目标语言句子tensor转换为字符串，并去掉开始和结束符
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

```

开始推理:
```python
with open("./data/test_en.txt", 'r', encoding='utf-8') as f:
    test_data = f.readlines()

with open("sumbit.txt", 'w', encoding='utf-8') as f:
    for line in test_data:
        transformer.to(DEVICE)
        res = translate(transformer, line)
        f.write(''.join(res.split(' '))+'\n')
        
```

之后去提交即可。目前最高拿了13.9分
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2ccb5e2f64bf48b09935448bb6a06576.png)

 [完整代码](https://www.csdn.net/)
