from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from nltk.tokenize import word_tokenize
import numpy as np
import random
import os
import nltk
import matplotlib.pyplot as plt



def extract_paragraphs(corpus_dir, num_paragraphs, max_tokens):
    paragraphs = []
    labels = []

    # 遍历语料库中的每个txt文件
    for novel_file in os.listdir(corpus_dir):
        if novel_file.endswith('.txt'):
            novel_path = os.path.join(corpus_dir, novel_file)

            # 读取小说内容
            with open(novel_path, 'r', encoding='ANSI', errors='ignore') as file:  # 使用gbk编码
                novel_text = file.read()

            # 根据换行符分割成段落
            novel_paragraphs = novel_text.split('\n')

            # 随机抽取一定数量的段落
            random.shuffle(novel_paragraphs)
            for paragraph in novel_paragraphs:
                # 如果段落长度不超过max_tokens，则添加到数据集中
                if len(paragraph.split()) <= max_tokens:
                    paragraphs.append(paragraph)
                    labels.append(novel_file[:-4])  # 小说文件名作为标签
                    if len(paragraphs) == num_paragraphs:
                        return paragraphs, labels
    return paragraphs, labels

# 语料库路径
corpus_dir = r'./data'

# 参数设置
num_paragraphs = 1000
max_tokens = [20, 100, 500, 1000, 3000]

# 提取段落
paragraphs_all = {}
labels_all = {}
for k in max_tokens:
    paragraphs, labels = extract_paragraphs(corpus_dir, num_paragraphs, k)
    paragraphs_all[k] = paragraphs
    labels_all[k] = labels

# 定义LDA模型
def train_lda_model(paragraphs, num_topics, choose_tokenization):
    # 创建字典和语料库
    if choose_tokenization == 'word':
        texts = [word_tokenize(paragraph) for paragraph in paragraphs]
    elif choose_tokenization == 'char':
        texts = [list(paragraph) for paragraph in paragraphs]
    else:
        raise ValueError("Invalid choose_tokenization parameter.")

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 训练LDA模型
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    return lda_model



# 使用朴素贝叶斯分类器
classifier = MultinomialNB()

# 使用Pipeline将向量化和分类器组合起来
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', classifier)
])

# 参数设置
num_topics_list = [10, 20,50,60,80,100]  # 主题数量列表
max_tokens_list = [20, 100, 500, 1000, 3000]  # 最大 token 数量列表
tokenization_list = ['word', 'char']  # 分词方式列表


# 存储结果的字典
results = {}

# 训练和评估模型
for num_topics in num_topics_list:
    for max_tokens in max_tokens_list:
        for tokenization in tokenization_list:
            # 提取段落
            paragraphs, labels = extract_paragraphs(corpus_dir, num_paragraphs, max_tokens)

            # 训练LDA模型
            lda_model = train_lda_model(paragraphs, num_topics=num_topics, choose_tokenization=tokenization)

            # 将段落表示为主题分布
            topics_distribution = np.zeros((len(paragraphs), num_topics))
            for i, paragraph in enumerate(paragraphs):
                if tokenization == 'word':
                    bow_vector = lda_model.id2word.doc2bow(word_tokenize(paragraph))
                elif tokenization == 'char':
                    bow_vector = lda_model.id2word.doc2bow(list(paragraph))
                else:
                    raise ValueError("Invalid choose_tokenization parameter.")
                topics = lda_model[bow_vector]
                for topic in topics:
                    topics_distribution[i, topic[0]] = topic[1]

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(topics_distribution, labels, test_size=0.2,
                                                                random_state=42)

            # 使用10次交叉验证进行评估
            cv_scores = cross_val_score(classifier, X_train, y_train, cv=10)

            # 计算平均交叉验证分数
            mean_cv_score = np.mean(cv_scores)

            # 输出结果
            print(f"Num Topics: {num_topics}, Max Tokens: {max_tokens}, Tokenization: {tokenization}, Mean Cross-validation Score: {mean_cv_score:.4f}")

