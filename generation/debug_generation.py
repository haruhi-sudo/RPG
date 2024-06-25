import re
import json
import torch
from dataset_utils import PROMPT_DICT
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertForQuestionAnswering, BertTokenizer

@torch.no_grad()
def generate_text_once(model, tokenizer, prompt, retriever_content):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    input_ids = inputs["input_ids"]

    context_task_ids = torch.Tensor([1]).long().cuda()

    context_outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=100, 
        task_ids=context_task_ids, 
        eos_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
        # do_sample=True,  # Enable sampling
        # temperature=1.0  # Control the randomness
    )
        
    return tokenizer.batch_decode(context_outputs, skip_special_tokens=False)[0]

@torch.no_grad()
def generate_text_iter(model, tokenizer, prompt, retriever_content):
    if len(retriever_content) == 0:
        return "End"
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    input_ids = inputs["input_ids"]

    plan_task_ids = torch.Tensor([0]).long().cuda()
    context_task_ids = torch.Tensor([1]).long().cuda()

    # 目前只支持 batch_size=1
    turn = 0
    while(turn < len(retriever_content)):
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
            max_new_tokens=50, 
            task_ids=plan_task_ids, 
            eos_token_id=tokenizer.convert_tokens_to_ids("</plan>"),
            # do_sample=True,  # Enable sampling
            # temperature=1.0  # Control the randomness
        )

        plan_text = tokenizer.batch_decode(plan_outputs[:,input_ids.shape[1]:], skip_special_tokens=False)[0]

        # 如果生成的plan文本是</plan>，证明规划终止了
        if plan_outputs[:,input_ids.shape[1]] == tokenizer.convert_tokens_to_ids("</plan>"):
            if turn == 0: # 第一轮就生成了</plan>，说明是不需要检索的特殊情况
                context_inputs = torch.cat([input_ids, plan_outputs[:,input_ids.shape[1]:input_ids.shape[1]+1]], dim=-1)
                context_outputs = model.generate(
                    input_ids=context_inputs, 
                    max_new_tokens=100, 
                    task_ids=context_task_ids, 
                    eos_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
                    # do_sample=True,  # Enable sampling
                    # temperature=1.0  # Control the randomness
                )
                        
            return tokenizer.batch_decode(context_outputs, skip_special_tokens=False)[0]

        match = re.search('<plan>(.*?)</plan>', plan_text)
        if match:
            plan_text = match.group(1)
        else:
            plan_text = "End" 

        if plan_text == "End":
            break

        passage = retriever_content[turn]
        fparagraph = "<fparagraph>" + passage + "</fparagraph>"
        retrieve_tokens = tokenizer(fparagraph,return_tensors="pt")["input_ids"][:,1:].cuda()
        context_inputs = torch.cat([plan_outputs, retrieve_tokens], dim=1)

        context_outputs = model.generate(
            input_ids=context_inputs, 
            max_new_tokens=100, 
            task_ids=context_task_ids, 
            eos_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
            # do_sample=True,  # Enable sampling
            # temperature=1.0  # Control the randomness
        )

        input_ids = context_outputs
        turn += 1
    
    if 'context_outputs' not in locals():
        return "End"

    return tokenizer.batch_decode(context_outputs, skip_special_tokens=False)[0]


if __name__ == "__main__":
    model_path = "output/rag_rag_50_virual_rank_1-tmp/-best-1"
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)  
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, model_path).to(torch.bfloat16).cuda()

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    # 读取data/val.jsonl文件，取出第一个作为example
    with open('data/split_train/asqa.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[5:]:
        example = json.loads(line)
        
        content = re.findall('<fparagraph>(.*?)</fparagraph>', example["output"], re.DOTALL)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        source_text = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)

        print(generate_text_iter(model, tokenizer, source_text, content))

