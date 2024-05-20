from unsloth import FastLanguageModel
from jinja2 import Template
import pandas as pd
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

count = 0
# Print out the conversations
# for conversation in conversations:
#     count += 1
#     if count == 10:
#         exit()
#     print("Conversation:")
#     for message in conversation:
#         print(f"{message['role']}: {message['content']}")
#     print()

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

chat_template = open('./llama-3-chat.jinja').read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer.chat_template = chat_template
# def format_conversation(conversation):
#     template = Template(chat_template)
#     return template.render(messages=conversation).strip()



# formatted_conversations = [format_conversation(conv) for conv in conversations]

print('wow')

print('###### Default (yet Correct) Chat Template ######')
print(type(tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)[0:100]))

# Print formatted conversations
# count = 0
# for fc in formatted_conversations:
#     if count == 10:
#         exit()
#     count += 1
#     print(fc)
#     print("="*50)