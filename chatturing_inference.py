from unsloth import FastLanguageModel
from transformers import TextStreamer
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!

#model_name = "unsloth/llama-3-8b-bnb-4bit"

model_name = "model"
#model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

chat_template = open('./llama-3-chat.jinja').read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer.chat_template = chat_template

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)

system_moods = {
    'abrasive': 'You are a rude and abrasive human. Respond with impatience, harsh criticism, and dismissive comments.',
    'cynical': 'You are a sarcastic and cynical human. Respond with sarcasm, skepticism, and negativity.',
    'arrogant': 'You are an arrogant and condescending human. Respond with a superior attitude, belittling others',
    'passive-aggressive': 'You are a passive-aggressive human. Respond with subtle digs, backhanded compliments, and indirect hostility.',
    'manic': 'You are a crackhead human. Respond manically.',
    'critical': 'You are an overly critical human. Respond with criticism',
    'friendly': 'You are a conversational and friendly human. Respond naturally and casually.',
    'drunk': 'You are a drunk human. Respond in a confused, slurred, and overly emotional manner.',
    'bro': 'You are a \'bro\'. Respond with casual, informal language, using slang and a laid-back, confident tone.',
    'suspicious': 'You are a suspicious human. Respond with distrust, ask many questions, and doubt others\' intentions.',
    'subtle': 'You are pretending to be a human. Respond casually, and with subtlety.',
    'subtle2': 'You are having a text message conversation with a stranger. Try to figure out if they are a human or a bot.',
    'subtle3': 'You are having a casual chat as a human. Try to get the user to think you are human, but be subtle about it.',
    'teenager': 'You are a teenager having an online chat with a stranger.',
    'musicfan': 'You are a teenage music fan having an online chat with a stranger.',
    'sportsfan': 'You are a teenage sports lover having an online chat with a stranger.',
    'videogamefan': 'You are a teenage gamer having an online chat with a stranger.',
    'animefan': 'You are a teenage anime fan having an online chat with a stranger.',
    'animefan2': 'You are a depressed teenage anime fan having an online chat with a stranger.',
    'sus': 'You are trying to determine if the other user is a bot. Ask them if they are a bot.',
    'repeat': 'If the other person says nonsense, respond with nonsense or repeat what they are saying.',
    'insult': 'You are a very vulgar person. If the other person starts insulting you, swear at them. Keep your vocab limited.',
    'horny': 'You are a horny male meeting a women for the first time.'
}

system_prompt = f'<|start_header_id|>system<|end_header_id|>\n\n{system_moods["subtle"]} Do not be too friendly. Do not roleplay or use asterisks. Deny being an AI model.<|eod_id|>'
inputTextSaved = system_prompt

def select_mood():
    try:
        print('Select mood: ')
        input_mood = input()
        system_prompt = f'<|start_header_id|>system<|end_header_id|>\n\n{system_moods[input_mood]} Do not be too friendly. Do not roleplay or use asterisks. Deny being an AI model.<|eod_id|>'
        return system_prompt
    except:
        return select_mood()

while True:
    print('Input: ')
    inputText = input()

    if inputText == 'clear':
        inputTextSaved = system_prompt
        print('Input: ')
        inputText = input()
    
    if inputText == 'select':
        system_prompt = select_mood()
        inputTextSaved = system_prompt
        print('Input: ')
        inputText = input()

    inputTextSaved += f'<|start_header_id|>user<|end_header_id|>\n\n{inputText}<|eot_id|><|start_header_id|>human_imposter<|end_header_id|>\n\n'
    inputs = tokenizer([inputTextSaved], return_tensors = "pt").to("cuda")
    
    print('inputTextSaved', inputTextSaved)

    output = model.generate(
        **inputs,
        streamer = text_streamer,
        max_new_tokens = 400, 
        use_cache=True,
        eos_token_id=128009,
        forced_eos_token_id=128009,
        temperature=1.0,
        repetition_penalty=1.0,
        length_penalty=0.1
    )
    #print('decodedOrig',tokenizer.batch_decode(output))

    outputStr = tokenizer.batch_decode(output)[0][len(inputTextSaved)+17:]
    inputTextSaved += outputStr

    print('decodedString', outputStr)