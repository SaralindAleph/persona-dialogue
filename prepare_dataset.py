import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', help="character's name", required=True, type=str)
parser.add_argument('--rounds', '-r', help='context rounds', default=5, type=int)
parser.add_argument('--txtfilename', '-t', help='rare dataset', default='dataset.txt', type=str)
parser.add_argument('--jsonfilename', '-j', help='processed dataset', default='processed_dataset.json', type=str)
args = parser.parse_args()

all_dialog_samples = []
character_name = args.name + ":"
context_length = args.rounds
a = context_length + 1
with open(args.txtfilename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    section_lines = []
    for line in lines:
        if line.strip():
            section_lines.append(line)
            if line.startswith(character_name):
                single_dialog = {}
                single_dialog['output'] = line.lstrip(character_name).strip()
                if len(section_lines) > a:
                    single_dialog['input'] = ''.join(section_lines[-a:-1])
                else:
                    single_dialog['input'] = ''.join(section_lines[:-1])
                    single_dialog['input'] += character_name
                single_dialog_r = {k: v for k, v in reversed(single_dialog.items())}
                all_dialog_samples.append(single_dialog_r)

with open(args.jsonfilename, 'w', encoding='utf-8') as f:
    json.dump(all_dialog_samples, f, ensure_ascii=False)
