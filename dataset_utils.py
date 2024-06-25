from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy
)
from torch.utils.data import DataLoader
import numpy as np
import re
import copy
import torch
import transformers
from datasets import load_dataset
from typing import Dict
from functools import partial
from transformers import AutoTokenizer


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "tqa": "", 
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}


def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int) -> Dict:
    """Tokenize a list of strings."""
    input_ids = labels = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_seq_length,
            truncation=True,
    ).input_ids
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, context_markups=None):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    '''
    example["output"] = re.sub(r'\[Plan: (.*?)\]', r'<plan>\1</plan>', example["output"])
    # 不需要所有的粗粒度证据
    example["output"] = re.sub(r'<paragraph>.*?</paragraph>', '', example["output"], flags=re.DOTALL)
    # if prompt doesn't end with space and completion doesn't start with space, add space
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source_text = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    target_text = example['output'] + tokenizer.eos_token

    examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
    sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

    # set labels to -100 for the source tokens
    input_ids = examples_tokenized["input_ids"].flatten()
    source_len = sources_tokenized["input_ids_lens"]
    labels = copy.deepcopy(input_ids)
    labels[ :source_len] = -100

    # 只需计算<answer>和</answer>，<plan>和</plan>之间的内容，其他位置变为-100
    vaild_labels = {}
    if context_markups is not None:
        new_labels = torch.full_like(labels, -100)

        if source_len >= len(labels):
            print("source_len >= len(labels)")
        elif labels[source_len] == context_markups["plan_id"][1]:
            new_labels[source_len] = labels[source_len]
        else:
            # 寻找<plan>和</plan>的位置，只需计算<plan>和</plan>之间的内容
            start_indices = (labels == context_markups["plan_id"][0]).nonzero(as_tuple=True)[0]
            end_indices = (labels == context_markups["plan_id"][1]).nonzero(as_tuple=True)[0]
            if len(start_indices) != len(end_indices):
                print(example["output"])
            # assert len(start_indices) == len(end_indices)
            for start, end in zip(start_indices, end_indices):
                if start < end:
                    new_labels[start:end+1] = labels[start:end+1]
        
        vaild_labels["plan"] = new_labels

        # 寻找<answer>和</answer>的位置，只需计算<answer>和</answer>之间的内容
        new_labels = torch.full_like(labels, -100)
        start_indices = (labels == context_markups["answer_id"][0]).nonzero(as_tuple=True)[0]
        end_indices = (labels == context_markups["answer_id"][1]).nonzero(as_tuple=True)[0]
        for start, end in zip(start_indices, end_indices):
            if start < end:
                new_labels[start:end+1] = labels[start:end+1]
        vaild_labels["answer"] = new_labels

    # 把终止符替换为</plan>
    vaild_labels["plan"][-1] = tokenizer.convert_tokens_to_ids("</plan>")

    attention_mask_answer = torch.ones_like(input_ids)
    attention_mask_plan = torch.ones_like(input_ids)
    
    # Plan时需要mask掉paragraph的部分
    start_indices = (input_ids == context_markups["paragraph_id"][0]).nonzero(as_tuple=True)[0]
    end_indices = (input_ids == context_markups["paragraph_id"][1]).nonzero(as_tuple=True)[0]
    for start, end in zip(start_indices, end_indices):
        if start < end:
            attention_mask_plan[start:end+1] = 0

    # 将一条样本拆成两条，一条计算plan，一条计算answer，分别使用不同的prompt
    return {
        'input_ids': [input_ids, input_ids],
        'labels': [vaild_labels["plan"], vaild_labels["answer"]],
        'attention_mask': [attention_mask_plan, attention_mask_answer],
        'task_ids': torch.tensor([0, 1]), # 0 for plan, 1 for answer
    }

@dataclass
class MyDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        input_ids =  [item for feature in features for item in feature['input_ids']]
        labels = [item for feature in features for item in feature['labels']] if "labels" in features[0].keys() else None
        attention_mask = [item for feature in features for item in feature['attention_mask']]
        task_ids = [item for feature in features for item in feature['task_ids']]

        features = [{
            'input_ids': input_ids[i],
            'labels': labels[i],
            'attention_mask': attention_mask[i],
            'task_ids': task_ids[i].item(),
        } for i in range(len(input_ids))]
       
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

if __name__ == "__main__":
    data_files = {}
    dataset_args = {}
    data_files["train"] = "data/train.jsonl"

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    tokenizer = AutoTokenizer.from_pretrained("/data1/LLM_Model/Llama-2-7b-hf/")

    special_token_dict = {"additional_special_tokens": ["<fparagraph>", "</fparagraph>", "<plan>", "</plan>", "<answer>", "</answer>"]}
    special_token_dict["bos_token"] = "<s>"
    special_token_dict["eos_token"] = "</s>"
    special_token_dict["unk_token"] = "<unk>"
    special_token_dict["pad_token"] = "<pad>"
    num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
    
    plan_id = (tokenizer.convert_tokens_to_ids("<plan>"), tokenizer.convert_tokens_to_ids("</plan>"))
    answer_id = (tokenizer.convert_tokens_to_ids("<answer>"), tokenizer.convert_tokens_to_ids("</answer>"))
    paragraph_id = (tokenizer.convert_tokens_to_ids("<fparagraph>"), tokenizer.convert_tokens_to_ids("</fparagraph>"))

    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=1024,
        context_markups={"plan_id": plan_id, "answer_id": answer_id, "paragraph_id": paragraph_id},
    )

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=1,
        remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=MyDataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, padding="longest"),
        batch_size=2
    )
    for step, batch in enumerate(train_dataloader):
        print(batch)
