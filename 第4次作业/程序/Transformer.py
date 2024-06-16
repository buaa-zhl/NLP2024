import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 加载预训练的GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 读取小说文本
file_path = "神雕侠侣.txt"  # 替换为你的小说文件名
with open(file_path, "r", encoding="ANSI") as file:
    novel_text = file.read()

# 将文本转换为GPT-2模型所需的格式
encoded_input = tokenizer.encode(novel_text, return_tensors="tf")

# 创建数据集
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=file_path,
    block_size=128
)

# 创建数据加载器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 微调模型
trainer.train()

# 生成后续片段
input_text = "小龙女道：“杨"
input_ids = tokenizer.encode(input_text, return_tensors="tf")

# 使用微调后的模型生成文本
generated_text = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
)

# 解码生成的文本
generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(generated_text)
