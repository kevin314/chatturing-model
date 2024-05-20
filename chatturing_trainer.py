from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer 
from datasets import load_dataset, Dataset
import pandas as pd
from torch.utils.data import TensorDataset
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!

# url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
# dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

#model_name = "unsloth/llama-3-8b-bnb-4bit"
model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)


###############################################################

chat_template = open('./llama-3-chat.jinja').read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer.chat_template = chat_template

print('wow')


df = pd.read_csv('conversations.csv')

conversations = []
current_conversation = []

lastRole = 'human_imposter'
for i, row in df.iterrows():
    if i == 0 or df.loc[i - 1, 'answer'] == row['question']:  # Simplified boundary condition
        if lastRole == 'human_imposter':
            role = 'user'
            lastRole = 'user'
            oppRole = 'human_imposter'
        else:
            role = 'human_imposter'
            lastRole = 'human_imposter'
            oppRole = 'user'

        current_conversation.append({'role': role, 'content': row['question']})
    else:
        current_conversation.append({'role': oppRole, 'content': df.loc[i-1,'answer']})
        conversations.append(current_conversation)
        lastRole = 'user'
        current_conversation = [{'role': 'user', 'content': row['question']}]
        

if current_conversation:
    conversations.append(current_conversation)


# messages = [
#     {'role': 'system', 'content': 'This is a system prompt.'},
#     {'role': 'user', 'content': 'This is the first user input.'},
#     {'role': 'human_imposter', 'content': 'This is the first human_imposter response.'},
#     {'role': 'user', 'content': 'This is the second user input.'},
# ]


print('###### Default (yet Correct) Chat Template ######')
#formatted_conversations = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

for i in range(0, 2):
    print(conversations[i])
    print('----')

dataset = Dataset.from_dict({"chat": conversations})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])

# Tokenize each formatted conversation
#tokenized_conversations = [tokenizer(conv, truncation=True, padding='longest', max_length=512, return_tensors='pt') for conv in formatted_conversations]
# tokenized_conversations = [tokenizer.encode(conv, return_tensors='pt') for conv in formatted_conversations]
# # input_ids = torch.cat(tokenized_conversations, dim=1).squeeze()
# # attention_mask = torch.ones_like(input_ids)

# # dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)

# # Calculate total number of conversations
# total_conversations = len(tokenized_conversations)

# # Calculate average sequence length
# average_sequence_length = sum(input_ids_tensor.shape[1] for conv in tokenized_conversations) / total_conversations

# print(f"Total conversations: {total_conversations}")
# print(f"Average sequence length: {average_sequence_length}")

###############################################################



# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "formatted_chat",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

trainer.train()

model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)