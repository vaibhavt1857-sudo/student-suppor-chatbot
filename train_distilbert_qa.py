import os
import sys
import time
import json
import glob
import shutil
import subprocess
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertPreTrainedModel,
    DistilBertModel,
    Trainer,
    TrainingArguments,
    set_seed
)

# -------------------------------------------------------------
# Multi-Task DistilBERT Architecture for QA & Intent Classification
# -------------------------------------------------------------
class DistilBertForQAAndIntent(DistilBertPreTrainedModel):
    def __init__(self, config, num_intent_labels=13):
        super().__init__(config)
        self.num_intent_labels = num_intent_labels
        self.distilbert = DistilBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.seq_classif_dropout if hasattr(config, 'seq_classif_dropout') else 0.2)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        intent_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = distilbert_output[0]  # (batch_size, seq_len, hidden_size)

        # QA Logits
        qa_logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 2)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # Intent Logits using [CLS] token representation (index 0)
        cls_output = self.dropout(sequence_output[:, 0, :])
        intent_logits = self.intent_classifier(cls_output)  # (batch_size, num_intent_labels)

        total_loss = None
        qa_loss = None
        intent_loss = None

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # Mask out unaligned/invalid start & end positions when computing QA loss
            valid_qa = (start_positions >= 0) & (end_positions >= 0)
            if valid_qa.any():
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                s_loss = loss_fct(start_logits[valid_qa], start_positions[valid_qa])
                e_loss = loss_fct(end_logits[valid_qa], end_positions[valid_qa])
                qa_loss = (s_loss + e_loss) / 2.0
                total_loss = qa_loss

        if intent_labels is not None:
            valid_intents = (intent_labels != -100)
            if valid_intents.any():
                loss_fct_intent = nn.CrossEntropyLoss(ignore_index=-100)
                intent_loss = loss_fct_intent(intent_logits[valid_intents], intent_labels[valid_intents])
                if total_loss is not None:
                    total_loss = total_loss + intent_loss
                else:
                    total_loss = intent_loss

        if total_loss is None:
            total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        if not return_dict:
            output = (start_logits, end_logits, intent_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "intent_logits": intent_logits,
        }

def main():
    set_seed(42)
    torch.set_num_threads(os.cpu_count())
    start_time = time.time()

    print("=" * 70)
    print("Multi-Task DistilBERT Fine-Tuning: Question Answering & Intent Classification")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Load All Datasets (TriviaQA RC, HuggingFace MS MARCO, Intents)
    # -------------------------------------------------------------
    print("[Step 1] Loading Datasets...")

    # A. Load Intents from data/intents.json
    intents_file = 'data/intents.json'
    if not os.path.exists(intents_file):
        intents_file = 'student-support-chatbot/data/intents.json'

    if not os.path.exists(intents_file):
        print(f"[ERROR] {intents_file} not found!")
        sys.exit(1)

    with open(intents_file, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)

    intents_list = intents_data.get('intents', [])
    tags = sorted(list({item['tag'] for item in intents_list}))
    intent2id = {tag: i for i, tag in enumerate(tags)}
    id2intent = {i: tag for i, tag in enumerate(tags)}

    print(f"Loaded {len(tags)} unique intent classes: {tags}")

    intent_samples = []
    for item in intents_list:
        tag = item['tag']
        label = intent2id[tag]
        for pattern in item.get('patterns', []):
            intent_samples.append({
                'question': pattern,
                'context': '',
                'answer_text': '',
                'intent_label': label,
                'source': 'intents.json'
            })

    print(f"Total intent patterns: {len(intent_samples)}")

    # B. Load TriviaQA RC Dataset (iterating across ALL 26 train-*.parquet files with error skipping)
    trivia_samples = []
    trivia_dirs = ['data/triviaqa_rc', 'student-support-chatbot/data/triviaqa_rc']
    trivia_files = []
    for td in trivia_dirs:
        tf = sorted(glob.glob(os.path.join(td, 'train-*.parquet')))
        if tf:
            trivia_files = tf
            break

    print(f"Found {len(trivia_files)} TriviaQA RC train parquet files. Loading samples from ALL files...")
    
    for pf in trivia_files:
        try:
            pf_obj = pq.ParquetFile(pf)
            top_cols = pf_obj.schema_arrow.names
            available_cols = [c for c in ['question', 'context', 'answer', 'search_results', 'entity_pages'] if c in top_cols]
            for batch in pf_obj.iter_batches(batch_size=20, columns=available_cols):
                d = batch.to_pydict()
                q_list = d.get('question', [])
                ctx_list = d.get('context', [None] * len(q_list))
                ans_list = d.get('answer', [None] * len(q_list))
                sr_list = d.get('search_results', [None] * len(q_list))
                ep_list = d.get('entity_pages', [None] * len(q_list))

                count = 0
                for i in range(len(q_list)):
                    q = q_list[i] if i < len(q_list) else ""
                    ans_obj = ans_list[i] if i < len(ans_list) else None
                    ans_text = ''
                    if isinstance(ans_obj, dict):
                        ans_text = ans_obj.get('value', '')
                        if not ans_text and 'aliases' in ans_obj and len(ans_obj['aliases']) > 0:
                            ans_text = ans_obj['aliases'][0]
                    elif isinstance(ans_obj, str):
                        ans_text = ans_obj

                    ctx_text = ''
                    if ctx_list and i < len(ctx_list) and isinstance(ctx_list[i], str) and ctx_list[i].strip():
                        ctx_text = ctx_list[i]
                    else:
                        sr = sr_list[i] if sr_list and i < len(sr_list) else None
                        if isinstance(sr, dict) and 'search_context' in sr:
                            sc = sr['search_context']
                            if isinstance(sc, (list, tuple, np.ndarray)) and len(sc) > 0:
                                ctx_text = str(sc[0])
                            elif isinstance(sc, str):
                                ctx_text = sc
                        
                        if not ctx_text:
                            ep = ep_list[i] if ep_list and i < len(ep_list) else None
                            if isinstance(ep, dict) and 'wiki_context' in ep:
                                wc = ep['wiki_context']
                                if isinstance(wc, (list, tuple, np.ndarray)) and len(wc) > 0:
                                    ctx_text = str(wc[0])
                                elif isinstance(wc, str):
                                    ctx_text = wc

                    if q and ctx_text:
                        trivia_samples.append({
                            'question': q,
                            'context': ctx_text,
                            'answer_text': ans_text,
                            'intent_label': -100,
                            'source': 'triviaqa_rc'
                        })
                        count += 1
                        if count >= 15:
                            break
                break # processed first batch from this parquet file
        except Exception as e:
            print(f"Warning skipping unparseable TriviaQA file {pf}: {e}")

    print(f"Loaded {len(trivia_samples)} TriviaQA RC QA samples across all {len(trivia_files)} parquet files.")

    # C. Load HuggingFace MS MARCO Dataset
    ms_marco_samples = []
    hf_ds_path = 'chatbot_project/data/huggingface_dataset/microsoft_ms_marco'
    src_ds_dir = 'data/huggingface_dataset/microsoft___ms_marco/v1.1/0.0.0/a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a'
    if not os.path.exists(src_ds_dir):
        src_ds_dir = 'student-support-chatbot/data/huggingface_dataset/microsoft___ms_marco/v1.1/0.0.0/a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a'

    load_path = hf_ds_path if os.path.exists(hf_ds_path) else src_ds_dir
    if os.path.exists(load_path):
        try:
            print(f"Loading MS MARCO dataset from HuggingFace dataset folder ({load_path})...")
            data_files = {}
            if os.path.exists(os.path.join(load_path, 'ms_marco-train.arrow')):
                data_files['train'] = os.path.join(load_path, 'ms_marco-train.arrow')
            if os.path.exists(os.path.join(load_path, 'ms_marco-validation.arrow')):
                data_files['validation'] = os.path.join(load_path, 'ms_marco-validation.arrow')
            
            if data_files:
                hf_ds = load_dataset('arrow', data_files=data_files)
                train_raw = hf_ds['train'].select(range(min(200, len(hf_ds['train']))))
                for ex in train_raw:
                    q = ex.get('query', '')
                    passages = ex.get('passages', {})
                    answers = ex.get('answers', [])
                    p_list = passages.get('passage_text', [])
                    is_sel = passages.get('is_selected', [])
                    ctx = ""
                    if 1 in is_sel:
                        ctx = p_list[is_sel.index(1)]
                    elif len(p_list) > 0:
                        ctx = p_list[0]
                    ans = answers[0] if len(answers) > 0 else ""
                    if q and ctx:
                        ms_marco_samples.append({
                            'question': q,
                            'context': ctx,
                            'answer_text': ans,
                            'intent_label': -100,
                            'source': 'ms_marco'
                        })
        except Exception as e:
            print(f"Warning skipping unparseable MS MARCO dataset file: {e}")

    print(f"Loaded {len(ms_marco_samples)} MS MARCO QA samples.")

    # Combine datasets & balance representation
    intent_samples_boosted = intent_samples * 3
    all_training_data = trivia_samples + ms_marco_samples + intent_samples_boosted
    
    # Shuffle and split train and validation (85% / 15%)
    import random
    random.seed(42)
    random.shuffle(all_training_data)
    
    val_size = max(50, int(len(all_training_data) * 0.15))
    train_data = all_training_data[val_size:]
    val_data = all_training_data[:val_size]

    print(f"Total Combined Training Samples   : {len(train_data)}")
    print(f"Total Combined Validation Samples : {len(val_data)}")

    # Convert to HuggingFace Dataset objects
    train_raw_ds = Dataset.from_pandas(pd.DataFrame(train_data))
    val_raw_ds = Dataset.from_pandas(pd.DataFrame(val_data))

    # -------------------------------------------------------------
    # Step 2: Preprocess & Align Token Spans
    # -------------------------------------------------------------
    print("[Step 2] Tokenizing questions, contexts, and aligning answer spans...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def preprocess_function(examples):
        queries = [str(q).strip() for q in examples['question']]
        contexts = [str(c).strip() for c in examples['context']]
        answers = examples['answer_text']
        intent_labels = examples['intent_label']

        tokenized = tokenizer(
            queries,
            contexts,
            max_length=192,
            truncation="longest_first",
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = tokenized.pop("offset_mapping")
        final_start = []
        final_end = []

        for i, offsets in enumerate(offset_mapping):
            ans_text = str(answers[i]) if answers[i] is not None else ""
            ctx_text = str(contexts[i]) if contexts[i] is not None else ""
            sequence_ids = tokenized.sequence_ids(i)

            if not ans_text or not ctx_text:
                final_start.append(0)
                final_end.append(0)
                continue

            s_char = ctx_text.lower().find(ans_text.lower())
            if s_char == -1:
                final_start.append(0)
                final_end.append(0)
                continue

            e_char = s_char + len(ans_text)

            context_indices = [idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1]
            if not context_indices:
                final_start.append(0)
                final_end.append(0)
                continue

            context_start = context_indices[0]
            context_end = context_indices[-1]

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
        tokenized["intent_labels"] = intent_labels
        return tokenized

    train_dataset = train_raw_ds.map(preprocess_function, batched=True, remove_columns=train_raw_ds.column_names)
    val_dataset = val_raw_ds.map(preprocess_function, batched=True, remove_columns=val_raw_ds.column_names)

    print(f"Preprocessed train dataset size: {len(train_dataset)}")
    print(f"Preprocessed val dataset size  : {len(val_dataset)}")

    # -------------------------------------------------------------
    # Step 3: Initialize Model & Trainer
    # -------------------------------------------------------------
    print("[Step 3] Initializing DistilBertForQAAndIntent Model...")
    model = DistilBertForQAAndIntent.from_pretrained('distilbert-base-uncased', num_intent_labels=len(tags))

    num_epochs = 3
    batch_size = 16
    learning_rate = 2e-5

    output_checkpoint_dir = 'chatbot_project/models/distilbert_qa_checkpoints'

    training_args = TrainingArguments(
        output_dir=output_checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        use_cpu=True
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        start_logits, end_logits, intent_logits = predictions

        if isinstance(labels, tuple):
            start_positions, end_positions, intent_labels = labels[:3]
        elif isinstance(labels, np.ndarray) and labels.ndim == 2:
            if labels.shape[1] >= 3:
                start_positions, end_positions, intent_labels = labels[:, 0], labels[:, 1], labels[:, 2]
            elif labels.shape[0] >= 3:
                start_positions, end_positions, intent_labels = labels[0], labels[1], labels[2]
            else:
                start_positions = labels[:, 0]
                end_positions = labels[:, 0]
                intent_labels = labels[:, 0]
        else:
            start_positions, end_positions, intent_labels = labels[0], labels[1], labels[2]

        pred_start = start_logits.argmax(axis=-1)
        pred_end = end_logits.argmax(axis=-1)
        pred_intent = intent_logits.argmax(axis=-1)

        # QA Accuracy
        start_acc = (pred_start == start_positions).mean()
        end_acc = (pred_end == end_positions).mean()
        exact_match = ((pred_start == start_positions) & (pred_end == end_positions)).mean()

        # Intent Accuracy (only for samples with valid intent_labels != -100)
        valid_mask = (intent_labels != -100)
        if valid_mask.any():
            intent_acc = (pred_intent[valid_mask] == intent_labels[valid_mask]).mean()
        else:
            intent_acc = 0.0

        return {
            "start_accuracy": float(start_acc),
            "end_accuracy": float(end_acc),
            "exact_match_accuracy": float(exact_match),
            "intent_accuracy": float(intent_acc)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # -------------------------------------------------------------
    # Step 4: Fine-Tune Model (3 Epochs, batch size 16, lr 2e-5)
    # -------------------------------------------------------------
    print("[Step 4] Starting Multi-Task Training...")
    train_result = trainer.train()
    print("Training finished successfully!")

    print("Running evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")

    # -------------------------------------------------------------
    # Step 5: Save Trained Model Artifacts into student-support-chatbot/models/distilbert_qa/
    # -------------------------------------------------------------
    save_dirs = [
        'student-support-chatbot/models/distilbert_qa',
        'models/distilbert_qa',
        'chatbot_project/models/distilbert_qa'
    ]

    for sd in save_dirs:
        abs_sd = os.path.abspath(sd)
        print(f"[Step 5] Saving model and tokenizer to {abs_sd}...")
        os.makedirs(abs_sd, exist_ok=True)
        trainer.save_model(abs_sd)
        tokenizer.save_pretrained(abs_sd)

        # Save intent mapping file
        intent_map_path = os.path.join(abs_sd, 'intent_map.json')
        with open(intent_map_path, 'w', encoding='utf-8') as f:
            json.dump({'intent2id': intent2id, 'id2intent': id2intent}, f, indent=2)

    # -------------------------------------------------------------
    # Step 6: Export Training Logs and Evaluation Metrics into student-support-chatbot/models/training_report.txt
    # -------------------------------------------------------------
    total_time = time.time() - start_time
    report_dirs = [
        'student-support-chatbot/models/training_report.txt',
        'models/training_report.txt',
        'chatbot_project/models/training_report.txt'
    ]

    report_content = f"""====================================================================
Multi-Task DistilBERT Fine-Tuning Training Report
====================================================================
Datasets Used      : TriviaQA RC (26 parquets), HuggingFace MS MARCO, intents.json
Base Model         : distilbert-base-uncased
Task Type          : Question Answering (Extractive) & Intent Classification
Device             : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}

HYPERPARAMETERS:
--------------------------------------------------------------------
Epochs             : {num_epochs}
Batch Size         : {batch_size}
Learning Rate      : {learning_rate}
Optimizer          : AdamW (weight_decay=0.01)
Max Sequence Len   : 192
Doc Stride         : 64

DATASET STATISTICS:
--------------------------------------------------------------------
TriviaQA RC Samples: {len(trivia_samples)} (loaded across all {len(trivia_files)} parquet files)
MS MARCO Samples   : {len(ms_marco_samples)}
Intent Patterns    : {len(intent_samples)} ({len(tags)} intent classes)
Training Samples   : {len(train_data)}
Validation Samples : {len(val_data)}

TRAINING PERFORMANCE:
--------------------------------------------------------------------
Total Training Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)
Global Steps       : {train_result.global_step}
Training Loss      : {train_result.training_loss:.4f}

EVALUATION METRICS (Validation Set):
--------------------------------------------------------------------
Validation Loss    : {eval_metrics.get('eval_loss', 'N/A'):.4f}
Start Pos Accuracy : {eval_metrics.get('eval_start_accuracy', 0.0) * 100:.2f}%
End Pos Accuracy   : {eval_metrics.get('eval_end_accuracy', 0.0) * 100:.2f}%
Exact Match Acc    : {eval_metrics.get('eval_exact_match_accuracy', 0.0) * 100:.2f}%
Intent Accuracy    : {eval_metrics.get('eval_intent_accuracy', 0.0) * 100:.2f}%

INTENT CLASSES ({len(tags)}):
--------------------------------------------------------------------
{json.dumps(id2intent, indent=2)}

TRAINING LOG HISTORY:
--------------------------------------------------------------------
"""
    for log in trainer.state.log_history:
        report_content += json.dumps(log) + "\n"

    report_content += f"""====================================================================
Model Artifacts Saved To:
- student-support-chatbot/models/distilbert_qa/
- models/distilbert_qa/
- chatbot_project/models/distilbert_qa/
Status: COMPLETED SUCCESSFULLY
====================================================================
"""

    for rp in report_dirs:
        abs_rp = os.path.abspath(rp)
        os.makedirs(os.path.dirname(abs_rp), exist_ok=True)
        with open(abs_rp, 'w', encoding='utf-8') as f:
            f.write(report_content)

    print("[Step 6] Training report exported successfully!")
    print("Multi-task DistilBERT fine-tuning pipeline completed cleanly.")

if __name__ == '__main__':
    main()
