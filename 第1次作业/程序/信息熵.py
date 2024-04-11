import jieba
import math
from collections import Counter

# 假设停词.txt包含所有停用词
with open('..\停词.txt','r', encoding='GB2312', errors='ignore') as f:
    stopwords = set([line.strip() for line in f])


def calculate_entropy(text, unit='word'):
    # 分词
    if unit == 'word':
        tokens = jieba.lcut(text)
    else:  # unit == 'char'
        tokens = list(text)

    # 删除停用词
    tokens = [token for token in tokens if token not in stopwords]

    # 计算频率
    count = Counter(tokens)
    total = sum(count.values())

    # 计算信息熵
    entropy = -sum(freq / total * math.log(freq / total, 2) for freq in count.values())
    return entropy


# 读取多个文本文件并计算信息熵
file_names = ['..\jyxstxtqj_downcc.com\白马啸西风.txt',
              '..\jyxstxtqj_downcc.com\碧血剑.txt',
              '..\jyxstxtqj_downcc.com\飞狐外传.txt',
              '..\jyxstxtqj_downcc.com\连城诀.txt',
              '..\jyxstxtqj_downcc.com\鹿鼎记.txt',
              '..\jyxstxtqj_downcc.com\三十三剑客图.txt',
              '..\jyxstxtqj_downcc.com\射雕英雄传.txt',
              '..\jyxstxtqj_downcc.com\神雕侠侣.txt',
              '..\jyxstxtqj_downcc.com\书剑恩仇录.txt',
              '..\jyxstxtqj_downcc.com\天龙八部.txt',
              '..\jyxstxtqj_downcc.com\侠客行.txt',
              '..\jyxstxtqj_downcc.com\笑傲江湖.txt',
              '..\jyxstxtqj_downcc.com\雪山飞狐.txt',
              '..\jyxstxtqj_downcc.com\倚天屠龙记.txt',
              '..\jyxstxtqj_downcc.com\鸳鸯刀.txt',
              '..\jyxstxtqj_downcc.com\越女剑.txt']  # 示例文件名列表
entropies = {'word': [], 'char': []}

for file_name in file_names:
    with open(file_name, 'r',  encoding='GB2312', errors='ignore') as f:
        text = f.read()
        entropies['word'].append(calculate_entropy(text, 'word'))
        entropies['char'].append(calculate_entropy(text, 'char'))

# 输出每个文件的平均信息熵
for unit in ['word', 'char']:
    average_entropy = sum(entropies[unit]) / len(entropies[unit])
    print(f'平均{unit}信息熵: {average_entropy}')

for file_name in file_names:
    with open(file_name, 'r',  encoding='GB2312', errors='ignore') as f:
        text = f.read()
        word_entropy = calculate_entropy(text, 'word')
        char_entropy = calculate_entropy(text, 'char')
        print(f'{file_name} - 字信息熵: {char_entropy}, 词信息熵: {word_entropy}')