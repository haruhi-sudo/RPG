import torch
import spacy
import argparse
from peft import PeftConfig, PeftModel
from retriever.passage_retrieval import Retriever
from inference.long_form_demo import generate_text_iter
from dataset_utils import PROMPT_DICT, TASK_INST
from transformers import AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import FlagReranker

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--extract_model_path', type=str, default="path")
    parser.add_argument('--extract_tokenizer', type=str, default="path")
    parser.add_argument('--retriever_model', type=str, required=True)
    parser.add_argument('--passages', type=str, required=True)
    parser.add_argument('--passages_embeddings', type=str, required=True)

    args = parser.parse_args()

    model_path = args.model_path
    extract_model_path = args.extract_model_path
    extract_tokenizer_path = args.extract_tokenizer
    retriever_model = args.retriever_model
    passages = args.passages
    passages_embeddings = args.passages_embeddings

    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)  
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, model_path).to(torch.bfloat16).cuda()

    extractor = FlagReranker(extract_model_path, device='cuda')
    spacy_tokenizer = spacy.load(extract_tokenizer_path)

    retriever = Retriever(None)
    retriever.setup_retriever_demo(retriever_model, passages, passages_embeddings, save_or_load_index=True)
    
    print(generation_with_retriever_demo("Tell me more about Hello Kitty", model, extractor, 
                                         tokenizer, spacy_tokenizer,
                                         retriever=retriever, use_retriever=True))
    