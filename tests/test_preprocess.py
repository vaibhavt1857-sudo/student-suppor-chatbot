import os
import shutil
import subprocess
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# 1. Ensure directory link / structure
target_ds_dir = os.path.abspath('chatbot_project/data/huggingface_dataset/microsoft_ms_marco')
src_ds_dir = os.path.abspath('data/huggingface_dataset/microsoft___ms_marco/v1.1/0.0.0/a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a')

os.makedirs('chatbot_project/data/huggingface_dataset', exist_ok=True)
os.makedirs('chatbot_project/models', exist_ok=True)
os.makedirs('models', exist_ok=True)

if not os.path.exists(target_ds_dir):
    try:
        cmd = f'cmd /c mklink /J "{target_ds_dir}" "{src_ds_dir}"'
        subprocess.run(cmd, shell=True, check=True)
        print("Created junction successfully")
    except Exception as e:
        print("Junction failed, copying files:", e)
        shutil.copytree(src_ds_dir, target_ds_dir)

# 2. Load dataset
dataset_path = 'chatbot_project/data/huggingface_dataset/microsoft_ms_marco'
data_files = {
    'train': os.path.join(dataset_path, 'ms_marco-train.arrow'),
    'validation': os.path.join(dataset_path, 'ms_marco-validation.arrow'),
    'test': os.path.join(dataset_path, 'ms_marco-test.arrow')
}

raw_datasets = load_dataset('arrow', data_files=data_files)
print("Loaded raw datasets:", raw_datasets)

# 3. Test Tokenizer & Preprocessing
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def preprocess_qa_batch(examples):
    queries = [q.strip() for q in examples['query']]
    contexts = []
    start_positions = []
    end_positions = []

    for i in range(len(queries)):
        passages = examples['passages'][i]
        answers = examples['answers'][i]
        
        # Find passage text marked as selected, or first passage
        passage_text = ""
        is_selected = passages.get('is_selected', [])
        passage_list = passages.get('passage_text', [])
        
        if 1 in is_selected:
            idx = is_selected.index(1)
            passage_text = passage_list[idx]
        elif len(passage_list) > 0:
            passage_text = passage_list[0]
            
        contexts.append(passage_text)
        
        # Target answer text
        answer_text = answers[0] if len(answers) > 0 else ""
        
        start_char = -1
        if answer_text and passage_text:
            start_char = passage_text.lower().find(answer_text.lower())
            
        if start_char != -1:
            end_char = start_char + len(answer_text)
        else:
            start_char = 0
            end_char = 0
            
        start_positions.append(start_char)
        end_positions.append(end_char)

    # Tokenize query and context
    tokenized = tokenizer(
        queries,
        contexts,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=False,
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = tokenized.pop("offset_mapping")
    final_start = []
    final_end = []

    for i, offsets in enumerate(offset_mapping):
        s_char = start_positions[i]
        e_char = end_positions[i]
        sequence_ids = tokenized.sequence_ids(i)

        if s_char == 0 and e_char == 0:
            final_start.append(0)
            final_end.append(0)
            continue

        # Find token indices corresponding to character indices in context (sequence_id == 1)
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if context_start > context_end:
            final_start.append(0)
            final_end.append(0)
            continue

        # Check if answer is within context bounds
        if not (offsets[context_start][0] <= s_char and offsets[context_end][1] >= e_char):
            final_start.append(0)
            final_end.append(0)
        else:
            token_start_idx = context_start
            while token_start_idx <= context_end and offsets[token_start_idx][0] <= s_char:
                token_start_idx += 1
            token_start_idx = max(context_start, token_start_idx - 1)

            token_end_idx = context_end
            while token_end_idx >= context_start and offsets[token_end_idx][1] >= e_char:
                token_end_idx -= 1
            token_end_idx = min(context_end, token_end_idx + 1)

            final_start.append(token_start_idx)
            final_end.append(token_end_idx)

    tokenized["start_positions"] = final_start
    tokenized["end_positions"] = final_end
    return tokenized

sample_batch = raw_datasets['train'].select(range(10))
processed = preprocess_qa_batch(sample_batch)
print("Keys in processed batch:", list(processed.keys()))
print("Sample start_positions:", processed['start_positions'][:5])
print("Sample end_positions:", processed['end_positions'][:5])
