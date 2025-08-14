import json
import os
import sys
import re
import torch
import concurrent.futures
from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Initialize T-LLaMA
# model_path = "/ssd11/other/meiyy02/code_files/mt-corpus/T-LLaMA/T-LLaMA"
# model_path = snapshot_download(
#     repo_id="Pagewood/T-LLaMA",
#     local_dir="./T-LLaMA-downloaded",
#     resume_download=True
# )
model_path = "facebook/m2m100_418M"

print("Loading tokenizer...")
tokenizer = M2M100Tokenizer.from_pretrained(model_path, src_lang="ti", tgt_lang="zh")
print("Loading model...")
model = M2M100ForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
print("Loaded successfully!")

def translate(text, src_lang="ti", tgt_lang="zh"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def backtranslate_score(source, translation):
    """Score via back-translation similarity (Tibetan -> Chinese -> Tibetan)."""
    backtranslated_ti = translate(translation_zh, src_lang="zh", tgt_lang="ti")
    # Simple similarity: Jaccard index of word overlap
    source_words = set(source_ti.split())
    backtranslated_words = set(backtranslated_ti.split())
    overlap = len(source_words & backtranslated_words) / len(source_words | backtranslated_words)
    return round(overlap * 5)  # Scale to 1-5

def score(source, translations):
    """Score translations using:
        1. backtrack scoring
        2. perplexity scoring 
    """
    scores = []
    for trans in translations:
        # 1. BACKTRACK
        bt_score = backtranslate_score(source, trans)

        # 2. PERPLEXITY
        inputs = tokenizer(trans, return_tensors="pt", truncations=True).to(model.device)
        with torch.no_grad():
            loss = model(**inputs, lables=inputs["input_ids"]).loss
        perplexity = torch.exp(loss).item()
        perplexity_score = max(1, min(5, 5-(perplexity/1)))

        # weighted avg 
        combined_score = round(0.7 * bt_score + 0.3 * perplexity_score)
        scores.append(combined_score)


def main(src_file, *tgt_files):
    """
python3 M2M100/m2m100_ranking_comparative.py tbt-cn-200/test-mt-hyps/src.txt tbt-cn-200/test-mt-hyps/modelA.txt tbt-cn-200/test-mt-hyps/modelB.txt tbt-cn-200/test-mt-hyps/modelC.txt
    """
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    
    tgt_lines = []
    for f in tgt_files:
        with open(f, 'r', encoding='utf-8') as fin:
            tgt_lines.append([line.strip() for line in fin])
    
    results = []
    for i in tqdm(range(len(src_lines))):
        source = src_lines[i]
        translations = [tgt_lines[j][i] for j in range(len(tgt_files))]
        scores = score_translations(source, translations)
        results.append({
            "id": i,
            "source": source,
            "translations": translations,
            "scores": scores
        })
    
    # Save results
    with open("m2m100_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python m2m100_scoring.py <source_file> <translation1> [translation2...]")
        sys.exit(1)
    main(*sys.argv[1:])


# def main(src_file, *tgt_files):
#     """
#     usage:
# HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 T-LLaMA/t-llama_ranking_comparative.py tbt-cn-200/src_clean.txt tbt-cn-200/mt-hyps/hyp_deepseek-v3 tbt-cn-200/mt-hyps/hyp_google-translate tbt-cn-200/mt-hyps/hyp_qwen2.5_72b
# HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 T-LLaMA/t-llama_ranking_comparative.py tbt-cn-200/test-mt-hyps/src.txt tbt-cn-200/test-mt-hyps/modelA.txt tbt-cn-200/test-mt-hyps/modelB.txt tbt-cn-200/test-mt-hyps/modelC.txt
#     """
#     # Load data
#     with open(src_file, 'r', encoding='utf-8') as f:
#         src_lines = [line.strip() for line in f]
    
#     model_names = [os.path.basename(f).replace('hyp_', '') for f in tgt_files]
#     tgt_lines = []
    
#     for f in tgt_files:
#         with open(f, 'r', encoding='utf-8') as fin:
#             tgt_lines.append([line.strip() for line in fin])
    
#     # Verify lengths
#     num_lines = len(src_lines)
#     for lines in tgt_lines:
#         if len(lines) != num_lines:
#             print("Error: All files must have same number of lines")
#             sys.exit(1)
    
#     # Prepare jobs
#     jobs = []
#     for i in range(num_lines):
#         jobs.append({
#             'id': i,
#             'source': src_lines[i],
#             'translations': [tgt_lines[j][i] for j in range(len(tgt_files))],
#             'model_names': model_names
#         })
    
#     # Process with threading
#     results = []
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         futures = [executor.submit(score_translations, job) for job in jobs]
#         for future in tqdm(concurrent.futures.as_completed(futures), 
#                           total=len(futures), 
#                           desc="Scoring translations"):
#             results.append(future.result())
    
#     # Sort and save results
#     results.sort(key=lambda x: x['id'])
    
#     # output_dir = "tbt-cn-200/T-LLaMA_mev_scores"
#     output_dir = "tbt-cn-200/test-mt-hyps/M2M100"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save detailed results
#     with open(f"{output_dir}/detailed_results.jsonl", 'w') as fout:
#         for res in results:
#             fout.write(json.dumps(res, ensure_ascii=False) + '\n')
    
#     # Save individual score files
#     for i, name in enumerate(model_names):
#         with open(f"{output_dir}/scores_{name}.txt", 'w') as fout:
#             for res in results:
#                 fout.write(f"{res['scores'][i]}\n")
    
#     print(f"\nResults saved to {output_dir}/")
#     print(f"Average scores:")
#     for i, name in enumerate(model_names):
#         avg = sum(r['scores'][i] for r in results) / num_lines
#         print(f"{name}: {avg:.2f}")
