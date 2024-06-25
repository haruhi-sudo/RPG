import re
import math
import spacy
import json
import torch
import argparse
from dataset_utils import PROMPT_DICT, TASK_INST
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagReranker


def bge_rerank_result(reranker, query_text: str, docs: List[str], top_n=3):
    # 如果只有一个文档，直接返回
    if len(docs) == 1:
        return docs
    scores = reranker.compute_score([[query_text, passage] for passage in docs])
    
    score_doc_pairs = zip(scores, docs)
    sorted_pairs = sorted(score_doc_pairs, key=lambda x: x[0], reverse=True)
    sorted_docs = [doc for _, doc in sorted_pairs]

    return sorted_docs[:top_n]

@torch.no_grad()
def generate_text_iter(model, extractor, tokenizer, ext_tokenizer, prompt, retriever_content, max_new_tokens_plan=50, max_new_tokens_context=100, threshold=0.3, max_turns=3):
    retriever_content = [retriever_content[i]["text"] for i in range(len(retriever_content))]
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    input_ids = inputs["input_ids"]

    plan_task_ids = torch.Tensor([0]).long().cuda()
    context_task_ids = torch.Tensor([1]).long().cuda()

    # 目前只支持 batch_size=1
    turn = 0
    have_generated_pointer = input_ids.shape[1] # 指针，指向上一轮已经成功生成文本的末尾
    while(turn <= max_turns):
        attention_mask_plan = torch.ones_like(input_ids)
        # Plan时需要mask掉paragraph的部分
        start_indices = (input_ids == tokenizer.convert_tokens_to_ids("<fparagraph>")).nonzero(as_tuple=True)[-1]
        end_indices = (input_ids == tokenizer.convert_tokens_to_ids("</fparagraph>")).nonzero(as_tuple=True)[-1]
        for start, end in zip(start_indices, end_indices):
            if start < end:
                attention_mask_plan[:, start:end+1] = 0

        # 生成 plan 文本
        plan_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask_plan,
            max_new_tokens=max_new_tokens_plan, 
            task_ids=plan_task_ids, 
            eos_token_id=tokenizer.convert_tokens_to_ids("</plan>"),
            return_dict_in_generate=True, output_scores=True
        )
        score_dict = plan_outputs.scores[0][0]
        plan_outputs = plan_outputs.sequences

        plan_text = tokenizer.batch_decode(plan_outputs[:,have_generated_pointer:], skip_special_tokens=False)[0]

        plan_id = tokenizer.convert_tokens_to_ids("<plan>")
        no_plan_id = tokenizer.convert_tokens_to_ids("</plan>")
        # 如果生成的plan文本是</plan>，证明规划终止了
        if plan_outputs[:,have_generated_pointer] == no_plan_id: 
            if turn == 0:
                do_retrieve = score_dict[plan_id] / ( score_dict[no_plan_id] + score_dict[no_plan_id]) > threshold
                if do_retrieve == False: # 第一轮就生成了</plan>，说明是不需要检索的特殊情况
                    context_inputs = torch.cat([input_ids, plan_outputs[:,:input_ids.shape[1]+1]], dim=-1)
                    context_outputs = model.generate(
                        input_ids=context_inputs, 
                        max_new_tokens=max_new_tokens_context, 
                        task_ids=context_task_ids, 
                        eos_token_id=tokenizer.convert_tokens_to_ids("</answer>")
                    )
                    return tokenizer.batch_decode(context_outputs, skip_special_tokens=False)[0]
                
                else: # 第一轮就生成了</plan>，但仍然需要检索
                    input_ids = torch.cat([input_ids, plan_outputs[:,input_ids.shape[1]:input_ids.shape[1]+1]], dim=-1)
                    input_ids[:,-1] = plan_id
                    turn += 1
                    have_generated_pointer = input_ids.shape[1] - 1
                    continue 

        match = re.search('<plan>(.*?)</plan>', plan_text)
        if not match:
            break  # 如果没有找到匹配项，则结束循环。一般训练充分的情况下，很少出现这种情况

        # 提取匹配的计划文本
        plan_text = match.group(1)

        # 检索，找出最相关的文章
        passage = retriever_content[0]
        # 分段落
        doc = ext_tokenizer(passage)
        fine_passage = [sent.text.strip() for sent in doc.sents]
        # 合并fine_passage，防止太短
        step = math.ceil(len(fine_passage) / 5)
        fine_passage_merged = [" ".join(fine_passage[i:i+step]) for i in range(0, len(fine_passage), step)]
        # 重排序，找出最相关的段落
        fparagraph = bge_rerank_result(extractor, plan_text, fine_passage_merged, top_n=1)[0]
        fparagraph = passage

        fparagraph = "<fparagraph>" + fparagraph + "</fparagraph>"
        retrieve_tokens = tokenizer(fparagraph,return_tensors="pt")["input_ids"][:,1:].cuda()
        context_inputs = torch.cat([plan_outputs, retrieve_tokens], dim=1)

        context_outputs = model.generate(
            input_ids=context_inputs, 
            max_new_tokens=max_new_tokens_context, 
            task_ids=context_task_ids, 
            eos_token_id=tokenizer.convert_tokens_to_ids("</answer>")
        )

        input_ids = context_outputs
        have_generated_pointer = input_ids.shape[1]
        turn += 1

    return tokenizer.batch_decode(context_outputs, skip_special_tokens=False)[0]


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="path")
    parser.add_argument('--input_file', type=str, default="path")
    parser.add_argument('--output_file', type=str, default="output/popqa.json")
    parser.add_argument('--task', type=str, default="tqa")

    args = parser.parse_args()

    model_path = args.model_path
    input_path = args.input_file
    output_path = args.output_file
    
    input_data = []
    if input_path.endswith(".jsonl"):
        with open(input_path, 'r') as f:
            for line in f:
                input_data.append(json.loads(line))
    else:
        input_data = json.load(open(input_path))

    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)  
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, model_path).to(torch.bfloat16).cuda()

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    extractor = FlagReranker('data/bge-reranker-base', device='cuda')
    ext_tokenizer = spacy.load('data/en_core_web_sm')

    results = []
    for instance_idx, item in tqdm(enumerate(input_data), total=len(input_data)):
        try:
            instructions = TASK_INST[args.task]
            if args.task == "arc_c":
                prompt = item["question"]
                ctxs = item["ctxs"][:10]
                prompt = instructions + "## Input:\n\n" + prompt
                for choice_label, choice_text in zip(item["choices"]["label"], item["choices"]["text"]):
                    prompt += "\n" + choice_label + ": " + choice_text

            elif args.task == "fever":
                prompt = item["question"]
                ctxs = item["ctxs"][:10]
                prompt = instructions + "## Input:\n\n" + prompt

            elif args.task == "tqa":
                prompt = item["question"]
                ctxs = item["ctxs"][:10]
                prompt = instructions + "## Input:\n\n" + prompt

            prompt = prompt_no_input.format_map({"instruction": prompt})
            generated_text = generate_text_iter(model, extractor, tokenizer, ext_tokenizer, prompt, ctxs, max_turns=1)

            if args.task == "arc_c":
                result = {
                    "id": item["id"],
                    "question": item["question"],
                    "docs": item["ctxs"][:10],
                    "choices": item["choices"],
                    "answer": generated_text,
                    "gold": item["answerKey"],
                }

            elif args.task == "fever":
                result = {
                    "question": item["question"],
                    "docs": item["ctxs"][:10],
                    "answer": generated_text,
                    "gold": item["answers"],
                }

            elif args.task == "tqa":
                result = {
                    "question": item["question"],
                    "docs": item["ctxs"][:10],
                    "answer": generated_text,
                    "gold": item["answers"],
                }
            
            results.append(result)
            
        except Exception as e:
            if "id" in item.keys(): 
                print(f"Error processing item {item['id']}: {e}")
        
        if instance_idx % 10 == 0:
            with open(args.output_file + "_tmp", 'w') as writer:
                json.dump(results, writer)
        
        
    with open(args.output_file, 'w') as writer:
        json.dump(results, writer)
