import json
import os
import argparse

from tqdm import tqdm

def get_acc(examples):
    res = {
        "direct":0,
        "analytical":0,
        "parallel":0,
        "sequential":0,
        "total":0
    }
    numbers = {
    'direct': 0,
    'analytical': 0,
    'parallel': 0,
    'sequential': 0,
    'total':0
    }

    for item in examples:
        claim_type = item['claim_type']
        label = item['label']
        numbers[claim_type] += 1
        numbers['total'] += 1
        if item["response"]:
            if label==True and 'yes' in item["response"].lower():
                res[claim_type] += 1
                res["total"] += 1
            elif label==False and 'yes' not in item["response"].lower():
                res[claim_type] += 1
                res["total"] += 1
    for k,v in res.items():
        res[k] /= numbers[k]
    print(len(examples))
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs/test_cot")
    parser.add_argument('--eval_dir', type=str, default="processed_outputs")
    args = parser.parse_args()

    subdir = os.path.basename(args.output_dir)
    os.makedirs(os.path.join(args.eval_dir, subdir), exist_ok=True)

    for output_file in os.listdir(args.output_dir):
        if not output_file.endswith(".json"):
            continue

        examples_path = os.path.join(args.output_dir, output_file)
        examples = json.load(open(examples_path))
        
        eval_file = os.path.join(args.eval_dir, subdir, output_file)
        
        outputs = get_acc(examples)
        json.dump(outputs, open(eval_file, "w"), indent=4, ensure_ascii=False)
        

if __name__ == "__main__":
    main()