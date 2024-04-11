import jieba
from collections import Counter
import matplotlib.pyplot as plt
import os

# 创建停用词列表
def create_stopwords_list(filepath):
    with open('..\停词.txt','r', encoding='GB2312', errors='ignore') as file:
        stopwords = [line.strip() for line in file.readlines()]
    return stopwords

# 读取多个文件并统计词频
def process_files(file_list, stopwords):
    word_counts_list = []
    for file_path in file_list:
        with open(file_path, 'r', encoding='GB2312', errors='ignore') as file:
            text = file.read()
            words = jieba.lcut(text)
            filtered_words = [word for word in words if word not in stopwords]
            word_counts = Counter(filtered_words)
            word_counts_list.append(word_counts)
    return word_counts_list

# 绘制Zipf图
def plot_zipf(word_counts_list, file_list):
    plt.figure(figsize=(10, 6))
    for i, word_counts in enumerate(word_counts_list):
        ranks = list(range(1, len(word_counts)+1))
        frequencies = sorted(word_counts.values(), reverse=True)
        plt.loglog(ranks, frequencies, label=f'File {i+1}')
    plt.title('Zipf\'s Law')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# 获取当前工作目录中所有的.txt文件
file_list = [os.path.join('..\jyxstxtqj_downcc.com', f) for f in os.listdir('..\jyxstxtqj_downcc.com') if os.path.isfile(os.path.join('..\jyxstxtqj_downcc.com', f)) and f.endswith('.txt')]


# 加载停用词表
stopwords = create_stopwords_list('..\停词.txt')  # 假设您的停用词表文件名为'stopwords.txt'

# 处理文件并绘图
word_counts_list = process_files(file_list, stopwords)
plot_zipf(word_counts_list, file_list)
