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
def generate_text_iter(model, extractor, tokenizer, ext_tokenizer, prompt, retriever_content, max_new_tokens_plan=50, max_new_tokens_context=100, similarity_threshold=0.8, max_turns=3):
    retriever_content = [retriever_content[i]["text"] for i in range(len(retriever_content))]
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    input_ids = inputs["input_ids"]

    plan_task_ids = torch.Tensor([0]).long().cuda()
    context_task_ids = torch.Tensor([1]).long().cuda()

    used_content = {}
    used_plan = []

    # 目前只支持 batch_size=1
    turn = 0
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
        )

        plan_text = tokenizer.batch_decode(plan_outputs[:,input_ids.shape[1]:], skip_special_tokens=False)[0]

        # 如果生成的plan文本是</plan>，证明规划终止了
        if plan_outputs[:,input_ids.shape[1]] == tokenizer.convert_tokens_to_ids("</plan>"):
            if turn == 0: # 第一轮就生成了</plan>，说明是不需要检索的特殊情况
                context_inputs = torch.cat([input_ids, plan_outputs[:,input_ids.shape[1]:input_ids.shape[1]+1]], dim=-1)
                context_outputs = model.generate(
                    input_ids=context_inputs, 
                    max_new_tokens=max_new_tokens_context, 
                    task_ids=context_task_ids, 
                    eos_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
                )
                        
            return tokenizer.batch_decode(context_outputs, skip_special_tokens=False)[0]

        match = re.search('<plan>(.*?)</plan>', plan_text)
        if not match:
            break  # 如果没有找到匹配项，则结束循环。一般训练充分的情况下，很少出现这种情况

        # 提取匹配的计划文本
        plan_text = match.group(1)

        #如果新生成的plan_text和之前的plan_text很相似，设置flag=1
        # 以下这些步骤是为了考虑不同检索文档的情况，如果检索文档相似度很高，可以考虑不使用以下步骤
        flag = 0
        if len(used_plan) != 0:
            vectorizer = TfidfVectorizer().fit([plan_text] + used_plan)
            plan_text_vec = vectorizer.transform([plan_text])
            used_plan_vec = vectorizer.transform(used_plan)

            for used_vec in used_plan_vec:
                similarity = cosine_similarity(plan_text_vec, used_vec)
                if similarity[0][0] > similarity_threshold:  # 这里的0.8是阈值，可以根据需要调整
                    flag = 1
                    break 
        
        if flag == 1:
            idx = next((i for i, _ in enumerate(retriever_content) if i not in used_content), 0)
        else:
            idx = 0
  
        used_content[idx] = 1
        used_plan.append(plan_text)

        passage = retriever_content[idx]
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
            eos_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
        )

        input_ids = context_outputs
        turn += 1

    return tokenizer.batch_decode(context_outputs, skip_special_tokens=False)[0]


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="path")
    parser.add_argument('--input_file', type=str, default="path")
    parser.add_argument('--output_file', type=str, default="output/popqa_eval.json")
    parser.add_argument('--task', type=str, default="popqa")

    args = parser.parse_args()

    model_path = args.model_path
    input_path = args.input_file
    output_path = args.output_file

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
            if args.task != "hotpopqa":
                prompt = item["question"]
                ctxs = item["docs"][:5]
                instructions = TASK_INST[args.task]
                prompt = instructions + "## Input:\n\n" + prompt
                prompt = prompt_no_input.format_map({"instruction": prompt})

                generated_text = generate_text_iter(model, extractor, tokenizer, ext_tokenizer, prompt, ctxs)

                result = {
                    "id": item["sample_id"],
                    "question": item["question"],
                    "docs": item["docs"][:5],
                    "answer": generated_text,
                    "gold": item["answer"],
                }
                results.append(result)
            else:
                prompt = item["question"]
                ctxs = item["contexts"][:5]
                instructions = TASK_INST[args.task]
                prompt = instructions + "## Input:\n\n" + prompt
                prompt = prompt_no_input.format_map({"instruction": prompt})

                generated_text = generate_text_iter(model, extractor, tokenizer, ext_tokenizer, prompt, ctxs)

                result = {
                    "id": instance_idx,
                    "question": item["question"],
                    "docs": item["contexts"][:5],
                    "answer": generated_text,
                    "gold": item["answers"],
                }
                results.append(result)
        except Exception as e:
            print(f"Error processing item {instance_idx}: {e}")
        
        if instance_idx % 10 == 0:
            with open(args.output_file + "_tmp", 'w') as writer:
                json.dump(results, writer)
        
        
    with open(args.output_file, 'w') as writer:
        json.dump(results, writer)
