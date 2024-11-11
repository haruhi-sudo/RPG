# Retrieve-Plan-Generation: An Iterative Planning and Answering Framework for Knowledge-Intensive LLM Generation
 

This includes the original implementation of [Retrieve-Plan-Generation: An Iterative Planning and Answering Framework for Knowledge-Intensive LLM Generation(RPG)](https://arxiv.org/abs/2406.14979v1)


# Install environment with Conda
Create a conda env and follow setup.sh to install dependencies.

# Quick start
### 1. Train RPG
```bash
bash finetuning.sh # Modify the configuration as needed
```

### 2. Run RPG in the demo dataset

#### Download our demo models
1. Download our demo [models](https://drive.google.com/drive/folders/1HXh1LQmWL0XmHVInW4X4Tyg7hEeo0pvW?usp=drive_link)
2. Download bge-reranker-base in https://huggingface.co/BAAI/bge-reranker-base, and Download en_core_web_sm by
```bash
python -m spacy download en_core_web_sm
```
, put them into extractor/

#### Run RPG in the demo dataset
```
python inference/long_form_demo.py \
  --model_path "output/demo" \
  --extract_model_path "extractor/bge-reranker-base" \
  --extract_tokenizer "extractor/en_core_web_sm" \
  --input_file "data/test_long_form.json" \
  --output_file "output/asqa.json" \
  --task "asqa"
```

#### Or run RPG with retriever
Refer to [self-rag](https://github.com/AkariAsai/self-rag) to setup retriever(Retriever Setup Section). 


```python
import spacy
from inference.long_form_demo import generate_text_iter
from dataset_utils import PROMPT_DICT, TASK_INST
from peft import PeftConfig, PeftModel

def generation_with_retriever_demo(query, model, extractor, tokenizer, spacy_tokenizer, retriever=None, use_retriever=False, retrieve_content=None):
    if use_retriever:
        if retriever is None:
            raise ValueError("Retriever is required when use_retriever is True.")
        retrieve_content = [doc for doc in retriever.search_document_demo(query, n_docs=5)]
    elif retrieve_content is None:
        raise ValueError("retrieve_content cannot be None when use_retriever is False.")

    instructions = TASK_INST["asqa"]
    prompt = f"{instructions}## Input:\n\n{query}"
    prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": prompt})
    return generate_text_iter(model, extractor, tokenizer, spacy_tokenizer, prompt, retrieve_content)

model_path = "output/demo"
extract_model_path = "extractor/bge-reranker-base"
extract_tokenizer_path = "extractor/en_core_web_sm"

# Load model and retriever, you can refer to inference/
...

generation_with_retriever_demo("Tell me more about Hello Kitty", model, extractor, 
                                tokenizer, extract_tokenizer,
                                retriever=retriever, use_retriever=True)
```



# Citation
```
@article{lyu2024retrieve,
  title={Retrieve-Plan-Generation: An Iterative Planning and Answering Framework for Knowledge-Intensive LLM Generation},
  author={Lyu, Yuanjie and Niu, Zihan and Xie, Zheyong and Zhang, Chao and Xu, Tong and Wang, Yang and Chen, Enhong},
  journal={arXiv preprint arXiv:2406.14979},
  year={2024}
}
```
