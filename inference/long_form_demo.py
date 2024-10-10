import re
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

def bge_rerank_result(reranker, query_text: str, docs: List[str], top_n=1):
    scores = reranker.compute_score([[query_text, passage] for passage in docs])
    if len(docs) == 1:
        if scores > 0.3:
            return docs
        else:
            return [" "] # No relevant paragraph found, answer it by yourself.
    if max(scores) <= 0.3:
        return [" "] # No relevant paragraph found, answer it by yourself.
    score_doc_pairs = zip(scores, docs)
    sorted_pairs = sorted(score_doc_pairs, key=lambda x: x[0], reverse=True)
    sorted_docs = [doc for _, doc in sorted_pairs]

    return sorted_docs[:top_n]

@torch.no_grad()
def generate_text_iter(
    model, extractor, tokenizer, ext_tokenizer, prompt, 
    retriever_content, max_new_tokens_plan=50, max_new_tokens_answer=100, 
    max_turns=5
):
    model.eval()
    retriever_content = [retriever_content[i]["text"] for i in range(len(retriever_content))] 
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    input_ids = inputs["input_ids"]

    plan_task_ids = torch.Tensor([0]).long().cuda()
    answer_task_ids = torch.Tensor([1]).long().cuda()

    used_content = {}
    used_plan = []

    # Only batch_size=1
    turn = 0
    while(turn <= max_turns):
        attention_mask_plan = torch.ones_like(input_ids)
        # Masking paragraph when planning
        start_indices = (input_ids == tokenizer.convert_tokens_to_ids("<fparagraph>")).nonzero(as_tuple=True)[-1]
        end_indices = (input_ids == tokenizer.convert_tokens_to_ids("</fparagraph>")).nonzero(as_tuple=True)[-1]
        for start, end in zip(start_indices, end_indices):
            if start < end:
                attention_mask_plan[:, start:end+1] = 0

        # Planning
        plan_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask_plan,
            max_new_tokens=max_new_tokens_plan, 
            task_ids=plan_task_ids, 
            eos_token_id=tokenizer.convert_tokens_to_ids("</plan>"),
        )
        plan_text = tokenizer.batch_decode(plan_outputs[:,input_ids.shape[1]:], skip_special_tokens=False)[0]

        # </plan> means generation is done
        if plan_outputs[:,input_ids.shape[1]] == tokenizer.convert_tokens_to_ids("</plan>"):
            if turn == 0: # </plan> in the first turn means no retrieval needed
                answer_inputs = torch.cat([input_ids, plan_outputs[:,input_ids.shape[1]:input_ids.shape[1]+1]], dim=-1)
                answer_outputs = model.generate(
                    input_ids=answer_inputs, 
                    min_length=answer_inputs.shape[1] + 30,
                    max_new_tokens=max_new_tokens_answer, 
                    task_ids=answer_task_ids, 
                    eos_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
                )
            
            return tokenizer.batch_decode(answer_outputs, skip_special_tokens=False)[0]

        match = re.search('<plan>(.*?)</plan>', plan_text, re.DOTALL)
        if not match:
            break

        plan_text = match.group(1)

        # If the newly generated plan_text is very similar to the previous plan_text, set flag=1
        # The following steps are to consider different retrieval documents.
        flag = 0
        if len(used_plan) != 0:
            vectorizer = TfidfVectorizer().fit([plan_text] + used_plan)
            plan_text_vec = vectorizer.transform([plan_text])
            used_plan_vec = vectorizer.transform(used_plan)
            flag = any(cosine_similarity(plan_text_vec, used_vec)[0][0] > 0.9 for used_vec in used_plan_vec)

        idx = next((i for i, _ in enumerate(retriever_content) if i not in used_content), 0) if flag == 1 else 0
        used_content[idx] = 1
        used_plan.append(plan_text)

        passage = retriever_content[idx]
        doc = ext_tokenizer(passage)
        fine_passage = [sent.text.strip() for sent in doc.sents]
        
        # merge fine_passage
        fine_passage_merged = []
        temp_sentence = ""
        for sentence in fine_passage:
            if len(sentence.split()) < 15:
                temp_sentence += " " + sentence
            else:
                if temp_sentence:
                    fine_passage_merged.append(temp_sentence.strip())
                    temp_sentence = ""
                fine_passage_merged.append(sentence)
        if temp_sentence:
            fine_passage_merged.append(temp_sentence.strip())

        # Find the most relevant paragraph
        fparagraph = bge_rerank_result(extractor, plan_text, fine_passage_merged, top_n=1)[0]
        fparagraph = "<fparagraph>" + fparagraph + "</fparagraph>"
        retrieve_tokens = tokenizer(fparagraph,return_tensors="pt")["input_ids"][:,1:].cuda()
        answer_inputs = torch.cat([plan_outputs, retrieve_tokens], dim=1)

        answer_outputs = model.generate(
            input_ids=answer_inputs, 
            min_length=answer_inputs.shape[1] + 20,
            max_new_tokens=max_new_tokens_answer, 
            task_ids=answer_task_ids, 
            eos_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
        )

        input_ids = answer_outputs
        turn += 1

    return tokenizer.batch_decode(answer_outputs, skip_special_tokens=False)[0]


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="path")
    parser.add_argument('--extract_model_path', type=str, default="path")
    parser.add_argument('--extract_tokenizer', type=str, default="path")
    parser.add_argument('--input_file', type=str, default="data/test_long_form.json")
    parser.add_argument('--output_file', type=str, default="output/eval.json")
    parser.add_argument('--task', type=str, default="asqa")

    args = parser.parse_args()

    model_path = args.model_path
    extract_model_path = args.extract_model_path
    extract_tokenizer_path = args.extract_tokenizer
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

    extractor = FlagReranker(extract_model_path, device='cuda')
    ext_tokenizer = spacy.load(extract_tokenizer_path)

    results = []
    for instance_idx, item in tqdm(enumerate(input_data), total=len(input_data)):
        try:
            prompt = item["question"]
            ctxs = item["docs"][:5]
            instructions = TASK_INST[args.task]
            prompt = f"{instructions}## Input:\n\n{prompt}"
            prompt = prompt_no_input.format_map({"instruction": prompt})

            generated_text = generate_text_iter(model, extractor, tokenizer, ext_tokenizer, prompt, ctxs)

            result = {
                "id": item.get("sample_id", instance_idx),
                "question": item["question"],
                "generated_answer": generated_text,
                "gold": item.get("answer", item.get("answers")),
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing item {instance_idx}: {e}")
        
        if instance_idx % 10 == 0:
            with open(args.output_file + "_tmp", 'w') as writer:
                json.dump(results, writer)
    
    with open(args.output_file, 'w') as writer:
        json.dump(results, writer)
