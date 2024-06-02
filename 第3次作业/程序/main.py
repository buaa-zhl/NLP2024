import os
import re
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np


def load_data(corpus_dir, stopwords_path):
    # 读取小说文本
    novels = []
    for file_name in os.listdir(corpus_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(corpus_dir, file_name), 'r', encoding='ANSI') as file:
                novels.append(file.read())

    # 读取停词表
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())

    return novels, stopwords


def preprocess(text, stopwords):
    # 移除非汉字字符
    text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    # 分词
    tokens = jieba.lcut(text)
    # 移除停词
    tokens = [word for word in tokens if word not in stopwords]
    return tokens


def train_word2vec(processed_novels):
    # 训练Word2Vec模型
    model = Word2Vec(sentences=processed_novels, vector_size=100, window=5, min_count=5, workers=4)
    return model


def calculate_similarity(model, word1, word2):
    similarity = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")


def cluster_words(model, n_clusters=5):
    # 获取词向量
    words = list(model.wv.index_to_key)
    word_vectors = [model.wv[word] for word in words]

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(word_vectors)

    # 输出聚类结果
    for i, word in enumerate(words):
        print(f"Word: {word}, Cluster: {clusters[i]}")


def get_paragraph_vector(paragraph, model, stopwords):
    tokens = preprocess(paragraph, stopwords)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def calculate_paragraph_similarity(paragraph1, paragraph2, model, stopwords):
    paragraph1_vector = get_paragraph_vector(paragraph1, model, stopwords)
    paragraph2_vector = get_paragraph_vector(paragraph2, model, stopwords)

    similarity = np.dot(paragraph1_vector, paragraph2_vector) / (
                np.linalg.norm(paragraph1_vector) * np.linalg.norm(paragraph2_vector))
    print(f"Similarity between paragraph1 and paragraph2: {similarity}")


def main():
    # 定义路径
    corpus_dir = './novels'  # 小说文件夹路径
    stopwords_path = '../stopwords.txt'  # 停词表路径

    # 加载数据
    novels, stopwords = load_data(corpus_dir, stopwords_path)

    # 文本预处理
    processed_novels = [preprocess(novel, stopwords) for novel in novels]

    # 训练Word2Vec模型
    model = train_word2vec(processed_novels)

    # 验证词向量的有效性
    calculate_similarity(model, '一阵', '太虚')  # 替换成要比较的词
    cluster_words(model, n_clusters=5)  # 聚类分析

    # 计算段落之间的语义关联
    # paragraph1 = "第一段文本。"  # 替换成实际段落
    # paragraph2 = "第二段文本。"  # 替换成实际段落
    # calculate_paragraph_similarity(paragraph1, paragraph2, model, stopwords)


if __name__ == '__main__':
    main()
