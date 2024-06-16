import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 超参数
BATCH_SIZE = 15
SEQ_LENGTH = 40
EMBEDDING_DIM = 64
HIDDEN_UNITS = 128
EPOCHS = 10
# 数据处理
def load_text(file_path):
    with open(file_path, "r", encoding="ANSI") as file:
        text = file.read()
    return text

def prepare_data(text):
    chars = list(text)
    char2idx = {char: idx for idx, char in enumerate(set(chars))}
    idx2char = {idx: char for char, idx in char2idx.items()}
    input_seq = [char2idx[char] for char in chars]
    return input_seq, char2idx, idx2char

def split_data(input_seq):
    train_size = int(len(input_seq) * 0.8)
    val_size = int(len(input_seq) * 0.1)
    test_size = len(input_seq) - train_size - val_size
    train_seq = input_seq[:train_size]
    val_seq = input_seq[train_size:train_size+val_size]
    test_seq = input_seq[train_size+val_size:]
    return train_seq, val_seq, test_seq

# 数据生成器
# 数据生成器
def data_generator(seqs, batch_size, seq_length):
    X_enc, X_dec, y = [], [], []
    while True:
        for i in range(0, len(seqs) - seq_length, 1):
            X_enc.append(seqs[i:i+seq_length])
            X_dec.append(seqs[i:i+seq_length])
            y.append(seqs[i+1:i+seq_length+1])
            if len(X_enc) == batch_size:
                yield (tf.convert_to_tensor(X_enc), tf.convert_to_tensor(X_dec)), tf.convert_to_tensor(y)
                X_enc, X_dec, y = [], [], []


# Seq2Seq模型
def build_seq2seq_model(input_vocab_size, output_vocab_size, embedding_dim, hidden_units):
    encoder_input = Input(shape=(None,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_input)
    encoder_lstm = LSTM(hidden_units, return_state=True)
    encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(None,))
    decoder_embedding = Embedding(output_vocab_size, embedding_dim)(decoder_input)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab_size, activation="softmax")
    decoder_output = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)
    return model

# 文本生成
def generate_text(model, start_string, seq_length, num_chars, char2idx, idx2char):
    input_chars = [char2idx[char] for char in start_string]
    input_chars = np.array(input_chars).reshape(1, -1)

    text_generated = []

    for i in range(num_chars):
        predictions = model.predict([input_chars, np.zeros((1, 1))])
        predicted_id = np.argmax(predictions[0, -1, :])
        input_chars = np.append(input_chars, predicted_id).reshape(1, -1)

        text_generated.append(idx2char[predicted_id])

    return start_string + "".join(text_generated)

# 主函数
def main():
    # 加载数据
    file_path = "神雕侠侣.txt"
    text = load_text(file_path)
    input_seq, char2idx, idx2char = prepare_data(text)
    train_seq, val_seq, test_seq = split_data(input_seq)

    # 构建模型
    model = build_seq2seq_model(len(char2idx), len(char2idx), EMBEDDING_DIM, HIDDEN_UNITS)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy")

    # 训练模型
    train_gen = data_generator(train_seq, BATCH_SIZE, SEQ_LENGTH)
    val_gen = data_generator(val_seq, BATCH_SIZE, SEQ_LENGTH)
    model.fit(train_gen, steps_per_epoch=len(train_seq) // BATCH_SIZE,
              validation_data=val_gen, validation_steps=len(val_seq) // BATCH_SIZE,
              epochs=EPOCHS)

    # 保存模型
    model.save("seq2seq_generator.h5")

    # 生成文本
    start_string = "小龙女道：“杨"
    num_chars = 400
    generated_text = generate_text(model, start_string, SEQ_LENGTH, num_chars, char2idx, idx2char)
    print(generated_text)

if __name__ == "__main__":
    main()
