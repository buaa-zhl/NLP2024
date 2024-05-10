import os
import jieba
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import LatentDirichletAllocation

# 从目录中的文件加载语料库
corpus_directory = "./"  # 将此处改为包含txt文件的目录路径
corpus_files = [f for f in os.listdir(corpus_directory) if f.endswith('.txt')]

corpus = []
for file in corpus_files:
    with open(os.path.join(corpus_directory, file), 'r', encoding='ANSI') as f:
        corpus.extend(f.readlines())


# 预处理语料库的函数
def preprocess_corpus(corpus):
    preprocessed_corpus = []
    labels = []
    for line in corpus:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            label, text = parts
            preprocessed_corpus.append(text)
            labels.append(label)
    return preprocessed_corpus, labels


# 分词和向量化文本的函数
def tokenize_text(text, max_tokens):
    words = jieba.lcut(text)  # 使用jieba进行中文分词
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return ' '.join(words)


# 参数
num_paragraphs = min(1000, len(corpus))  # 调整为确保段落数不超过语料库大小
max_tokens_values = [3000]
num_topics = 50

# 预处理语料库
preprocessed_corpus, labels = preprocess_corpus(corpus)

# 随机抽样1000个段落
indices = np.random.choice(len(preprocessed_corpus), num_paragraphs, replace=True)  # 将replace改为True
sampled_corpus = [preprocessed_corpus[i] for i in indices]
sampled_labels = [labels[i] for i in indices]

# 分词和向量化文本
X = {}
for max_tokens in max_tokens_values:
    X[max_tokens] = [tokenize_text(text, max_tokens) for text in sampled_corpus]

# 编码标签
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(sampled_labels)

# 定义朴素贝叶斯分类器
classifier = MultinomialNB()

# 执行交叉验证
for max_tokens in max_tokens_values:
    print(f"最大词数: {max_tokens}")
    X_train, X_test, y_train, y_test = train_test_split(X[max_tokens], encoded_labels, test_size=0.1, random_state=30)

    # 构建管道
    pipeline = make_pipeline(CountVectorizer(), classifier)

    # 交叉验证
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10)
    print("交叉验证得分:", cv_scores)
    print("平均CV准确率:", np.mean(cv_scores))

    # 训练和测试最终模型
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("测试准确率:", test_accuracy)
    print()
