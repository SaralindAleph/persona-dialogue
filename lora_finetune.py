from datasets import load_dataset
import transformers
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', '-d', default='processed_dataset.json', type=str)
parser.add_argument('--model_name', '-m', default='pygmalion-6b', type=str)
parser.add_argument('--lora_name', '-l', default='lora_test', type=str)
parser.add_argument('--batch_size', '-b', default=4, type=int)
parser.add_argument('--epochs', '-e', default=10, type=int)
parser.add_argument('--cutoff_len', '-c', default=256, type=int)
parser.add_argument('--lora_rank', '-r', default=16, type=int)
args = parser.parse_args()

model_name_or_path = f"""models/{args.model_name}"""
device_map = "auto"

# an error will be raised while using AutoModel.from_pretrained, maybe someone could help fix this
if "gpt-j" or "pygmalion" or "opt" or "galactica" in model_name_or_path.lower():
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
elif "pythia" in model_name_or_path.lower():
    model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
else:
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map=device_map)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

model = prepare_model_for_int8_training(model)
config = LoraConfig(r=args.lora_rank, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                    bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)
train_data = load_dataset("json", data_files=args.dataset_name, split='train')


def tokenize_dialog(dataset):
    context = tokenizer.encode(dataset["input"], add_special_tokens=False)
    target = tokenizer.encode(dataset["output"], add_special_tokens=False)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    input_ids = context + target
    labels = [-100] * len(context) + target + [eos_token_id]

    input_ids = input_ids[-(args.cutoff_len - 1):]
    labels = labels[-args.cutoff_len:]

    input_ids = [bos_token_id] + input_ids

    input_ids = input_ids + [tokenizer.pad_token_id] * (args.cutoff_len - len(input_ids))
    labels = labels + [-100] * (args.cutoff_len - len(labels))

    attention_mask = [1] * (len(input_ids) - (args.cutoff_len - len(input_ids))) + [0] * (
            args.cutoff_len - len(input_ids))
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


train_data = train_data.shuffle().map(tokenize_dialog)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=60,
        num_train_epochs=args.epochs,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        output_dir="loras",
        save_total_limit=3,
        load_best_model_at_end=False,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

trainer.train()

model.save_pretrained(f"""loras/{args.lora_name}""")
