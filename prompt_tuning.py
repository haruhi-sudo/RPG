import os
import argparse
import logging
import math
import random
import datasets
import torch

from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset_utils import (
    encode_with_prompt_completion_format, 
    MyDataCollatorForSeq2Seq,
)

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from torch.nn import CrossEntropyLoss

from peft import (
    TaskType, 
    get_peft_model,
    PeftModel,
    PeftConfig,
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit
)

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--val_file", type=str, default=None, help="A csv or a json file containing the val data.", nargs='+'
    )
    parser.add_argument('--num_virtual_tokens', type=int, default=50, help='Number of virtual tokens')
    parser.add_argument('--modules_to_save', type=str, nargs='+', default=None, help='Modules to save')

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--already_trained_path",
        type=str,
        help="Path to already trained model",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
        help="Log the eval loss every eval_steps steps.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--use_special_tokens",
        action="store_true",
        help=(
            "Use special tokens."
        ),
    )
    
    args = parser.parse_args()

    return args

def loss_fn(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="sum")
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def main():
    args = parse_args()

    # A hacky way to make llama work with flash attention
    if args.use_flash_attn:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    data_files = {}
    dataset_args = {}
    data_files["train"] = args.train_file
    
    key_vals = [val.split('/')[-1][:-6] for val in args.val_file]
    for key_val, val in zip(key_vals, args.val_file):
        data_files[key_val] = val

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    # Load pretrained model and tokenizer
    if args.already_trained_path:
        config = PeftConfig.from_pretrained(args.already_trained_path)
    elif args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    
    if args.already_trained_path:
        tokenizer = AutoTokenizer.from_pretrained(args.already_trained_path, use_fast=not args.use_slow_tokenizer)
    elif args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    
    if args.already_trained_path is None:
        if args.use_special_tokens is True:
            special_token_dict = {"additional_special_tokens": ["<fparagraph>", "</fparagraph>", "<plan>", "</plan>", "<answer>", "</answer>"]}
        special_token_dict["bos_token"] = "<s>"
        special_token_dict["eos_token"] = "</s>"
        special_token_dict["unk_token"] = "<unk>"
        special_token_dict["pad_token"] = "<pad>"
        num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
        
    plan_id = (tokenizer.convert_tokens_to_ids("<plan>"), tokenizer.convert_tokens_to_ids("</plan>"))
    answer_id = (tokenizer.convert_tokens_to_ids("<answer>"), tokenizer.convert_tokens_to_ids("</answer>"))
    paragraph_id = (tokenizer.convert_tokens_to_ids("<fparagraph>"), tokenizer.convert_tokens_to_ids("</fparagraph>"))
    context_markups={"plan_id": plan_id, "answer_id": answer_id, "paragraph_id": paragraph_id}

    # 将文本数据转换为模型输入格式
    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        context_markups=context_markups if args.use_special_tokens is True else None
    )
    
    # tokenizing
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]
    val_datasets = {name: lm_datasets[name] for name in key_vals}

    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
   
    if args.already_trained_path is None:
        peft_config = MultitaskPromptTuningConfig(
            num_tasks=2,
            num_ranks=64,
            num_virtual_tokens=args.num_virtual_tokens,
            inference_mode=False, 
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=MultitaskPromptTuningInit.TEXT,
            tokenizer_name_or_path=args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path,
            prompt_tuning_init_text="Based on the input question, please plan the steps to answer the question. If planning has been completed, answer the question directly based on planning and retrieval of external knowledge.",
            # modules_to_save=["model.layers.31"],
            modules_to_save=args.modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        model = PeftModel.from_pretrained(model, args.already_trained_path)

    # 因为扩充了词表，所以需要重新训练模型的embedding层和lm_head层
    for name, param in model.named_parameters():
        if 'embed_tokens' in name or "lm_head" in name:
            param.requires_grad=True
    
    model.print_trainable_parameters()

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=MyDataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )

    eval_dataloaders = {
        name: DataLoader(
            dataset, 
            collate_fn=MyDataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.per_device_train_batch_size
        )
        for name, dataset in val_datasets.items()
    }

    # Optimizer, split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    eval_dataloaders = {name: accelerator.prepare(dataloader) for name, dataloader in eval_dataloaders.items()}

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        args.val_file = " ".join(key_vals)
        args.modules_to_save = None
        accelerator.init_trackers("open_instruct", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                labels = batch.pop("labels")
                prefix_labels = torch.full((labels.size(0), args.num_virtual_tokens), -100).to(labels.device)
                labels = torch.cat((prefix_labels, labels), dim=1)

                outputs = model(**batch)
                logits = outputs[0]
                loss = loss_fn(logits, labels)

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.eval_steps and completed_steps % args.eval_steps == 0:
                    model.eval()
                    
                    total_eval_loss = 0  # 初始化总 eval loss 为 0
                    for name, eval_dataloader in eval_dataloaders.items():
                        eval_loss = 0
                        for eval_step, eval_batch in enumerate(eval_dataloader):
                            with torch.no_grad():
                                eval_labels = eval_batch.pop("labels")
                                prefix_labels = torch.full((eval_labels.size(0), args.num_virtual_tokens), -100).to(eval_labels.device)
                                eval_labels = torch.cat((prefix_labels, eval_labels), dim=1)

                                eval_outputs = model(**eval_batch)
                                eval_logits = eval_outputs[0]
                                eval_loss += loss_fn(eval_logits, eval_labels).detach().float()
                        
                        eval_loss /= (eval_step + 1)
                        total_eval_loss += eval_loss  # 累加每个 eval_dataloader 的 eval loss
                        logger.info(f"  Step: {completed_steps}, Eval Loss for {name}: {eval_loss}")
                    
                    logger.info(f"  Step: {completed_steps}, Eval Loss: {total_eval_loss}")

                    if args.output_dir is not None:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(f"{args.output_dir}current")
                        unwrapped_model = accelerator.unwrap_model(model)
                        state_dict = accelerator.get_state_dict(model)
                        if accelerator.is_main_process:
                            unwrapped_model.save_pretrained(f"{args.output_dir}current", state_dict=state_dict, save_embedding_layers=True)

                    model.train()

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    unwrapped_model = accelerator.unwrap_model(model)
    # # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # # Otherwise, sometimes the model will be saved with only part of the parameters.
    # # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    # # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
    # # and has its own save_pretrained function for only saving lora modules.
    # # We have to mannually specify the is_main_process outside the save_pretrained function.
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(f"{args.output_dir}current", state_dict=state_dict, save_embedding_layers=True)
        tokenizer.save_pretrained(f"{args.output_dir}current")

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()

