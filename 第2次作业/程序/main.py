import os
import random
import re
import jieba
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# 获取当前脚本所在的文件夹路径
current_directory = os.path.dirname(os.path.abspath(__file__))


# 加载停用词
def load_stop_words(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        stop_words = set([line.strip() for line in file])
    return stop_words


# 获取停用词列表
stop_words_path = os.path.join(current_directory, '..', '停词.txt')
stop_words = load_stop_words(stop_words_path)


# 中文分词函数，包括停用词过滤和删除无意义符号
def chinese_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text)  # 删除标点符号
    text = re.sub(r'\s+', ' ', text)  # 将一个或多个空白符号替换为单个空格
    words = jieba.lcut(text)
    return ' '.join(words).strip()  # 返回处理后的字符串，去除首尾空白

# 读取文件并提取段落
def extract_paragraphs(file_path, token_limit, sample_size):
    all_paragraphs = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='ANSI', errors='ignore') as file:
            text = file.read()
        paragraphs = [para for para in text.split('\n') if len(para) >= 20]  # 确保段落长度至少为20
        all_paragraphs.extend(paragraphs)

    if len(all_paragraphs) < total_paragraphs_needed:
        raise ValueError(f"所有文件中符合长度要求的段落数量不足 {total_paragraphs_needed}")

    selected_paragraphs = random.sample(all_paragraphs, total_paragraphs_needed)
    return [' '.join(chinese_tokenizer(para)[:token_limit]) for para in selected_paragraphs]


# 计算每个文件的段落数量
def calculate_paragraphs_count(file_paths):
    word_counts = {}
    total_word_count = 0
    for file_path in file_paths:
        with open(file_path, 'r', encoding='ANSI') as file:
            text = file.read()
            words = chinese_tokenizer(text)
            word_counts[file_path] = len(words)
            total_word_count += len(words)

    paragraphs_count = {}
    for file_path, count in word_counts.items():
        proportion = count / total_word_count
        paragraphs_count[file_path] = round(proportion * total_paragraphs_needed)

    return paragraphs_count


# 加载数据集
data = []
labels = []
file_paths = [os.path.join(current_directory, f) for f in os.listdir(current_directory) if f.endswith('.txt')]
total_paragraphs_needed = 1000

# 调用calculate_paragraphs_count函数并打印结果
paragraphs_count = calculate_paragraphs_count(file_paths)
for file_path, count in paragraphs_count.items():
    print(f'{os.path.basename(file_path)}: {count}段落')

# 循环不同的K值
token_limits = [3000]  # K值确定为20
accuracy_results = {}

for token_limit in token_limits:
    data.clear()
    labels.clear()
    for file_path in file_paths:
        sample_size = paragraphs_count[file_path]  # 使用calculate_paragraphs_count函数的结果
        paragraphs = extract_paragraphs(file_path, token_limit, sample_size)
        data.extend(paragraphs)
        labels.extend([os.path.basename(file_path)] * len(paragraphs))

    # 文本向量化
    vectorizer = CountVectorizer(
        tokenizer=chinese_tokenizer,
        max_df=0.9,  # 在超过90%的段落中出现的词汇将被忽略
        min_df=10,  # 只考虑在至少10个段落中出现的词汇
        max_features=1000  # 最多保留1000个最重要的词汇
    )
    X = vectorizer.fit_transform(data)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # 应用LDA模型
    topic_count = 50  # 主题数量T
    lda = LatentDirichletAllocation(n_components=topic_count, random_state=0, max_iter=10)
    X_topics = lda.fit_transform(X)

    # 分类器选择
    classifier = MultinomialNB()

    # 创建管道
    pipeline = make_pipeline(vectorizer, lda, classifier)

    # 执行交叉验证
    cv_scores = cross_val_score(pipeline, X, y, cv=10)
    print("交叉验证分数:", cv_scores)
    print("平均交叉验证准确率:", cv_scores.mean())

    # 训练最终模型并测试准确性
    X_train, X_test, y_train, y_test = train_test_split(X_topics, y, test_size=0.1, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("测试准确率:", test_accuracy)