#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import mpu
import copy
import math
import json
import logging
import numpy as np
from pathlib import Path
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping, Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score
import transformers
from transformers.modeling_utils import load_sharded_checkpoint
from transformers import Trainer, is_torch_tpu_available, MODEL_FOR_CAUSAL_LM_MAPPING, set_seed
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.utils.versions import require_version
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import nested_numpify
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox import GPTNeoXForCausalLM
from transformers.trainer import TRAINER_STATE_NAME

import data_utils


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class PretrainModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    max_sequence_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "model max sequence length"
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class PretrainDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: str = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class PretrainArguments(transformers.TrainingArguments):
    debug_mode: Optional[bool] = field(default=False)
    peft_path: Optional[str] = field(default=None)

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    flash_attention: Optional[bool] = field(default=False)
    # tensor parallel configs start
    model_parallel_size: Optional[int] = field(default=1, metadata={"help": "model parallel size"})
    parallel_output: Optional[bool] = field(default=False)
    # tensor parallel configs end
    read_cached: Optional[bool] = field(default=False,
        metadata={"help": "Read cached data without tokenizing text"}
    )
    # deepspeed scheduler config start
    decay_min_lr: float = field(default=None, metadata={"help": "deepspeed WarmupDecayLR final decay lr"})
    # deepspeed scheduler config end
    resume_new_data: Optional[bool] = field(default=False, metadata={"help": "resume from  a new data part"})


class ExitCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global_step = state.global_step
        # The period of breaking training is set as 5000
        if global_step % 5000 == 0:
            import sys
            sys.exit(0)


class EpochEndCallback(TrainerCallback):
    """
    save checkpoint on epoch end
    """
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True

        return control


def change_trainer_state_file(train_args, resume_from_checkpoint):
    from transformers.trainer_callback import TrainerState
    from transformers.trainer import TRAINER_STATE_NAME

    if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
        trainer_state_path = os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        print(f"[change_trainer_state_file] trainer_state_path {trainer_state_path}")
        state = TrainerState.load_from_json(trainer_state_path)

        if train_args.resume_new_data:
            state.global_step = 0  # to clear epochs_trained, steps_trained_in_current_epoch, train with new part data
            print("[change_trainer_state_file] reset global_step=0")

        state.eval_steps = train_args.eval_steps
        state.save_steps = train_args.save_steps
        state.logging_steps = train_args.logging_steps
        state.save_to_json(trainer_state_path)
        print(f"[change_trainer_state_file] set state.eval_steps={state.eval_steps}, set state.save_steps={state.save_steps}")


def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(
            accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
        )
    }


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


def init_mpu(train_args):
    mpu.initialize_model_parallel(
        train_args.model_parallel_size)


def get_pythia_config(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)

    pythia_config = GPTNeoXConfig()
    for k, v in data.items():
        if hasattr(pythia_config, k):
            setattr(pythia_config, k, v)
    return pythia_config


def build_pythia_model(model_args, training_args, new_vocab_size=None):
    config_path = os.path.join(model_args.model_name_or_path, 'config.json')
    print(model_args.model_name_or_path)
    pythia_config = get_pythia_config(config_path)
    pythia_config.parallel_output = training_args.parallel_output
    pythia_config.flash_attention = training_args.flash_attention
    pythia_config.rope_scaling = None
    if new_vocab_size is not None:
        pythia_config.vocab_size = new_vocab_size
        logger.info(f"build model, new_vocab_size {pythia_config.vocab_size}")
    if pythia_config.vocab_size % training_args.model_parallel_size:
        pythia_config.vocab_size = \
            int((pythia_config.vocab_size // training_args.model_parallel_size + 1) * training_args.model_parallel_size)

    logger.info(f"build model, final vocab_size {pythia_config.vocab_size}")
    model = GPTNeoXForCausalLM(pythia_config)
    return model


def set_pythia_scheduler_config(trainer):
    ## these configs must be unsetted by transformers. otherwise, the setting doedn't work
    decay_min_lr = trainer.args.decay_min_lr
    if decay_min_lr is None:
        decay_min_lr = trainer.args.learning_rate * 0.1
    decay_type = trainer.args.lr_scheduler_type.lower()
    trainer.accelerator.state.deepspeed_plugin.hf_ds_config.fill_match("scheduler.params.decay_min_lr", decay_min_lr,
                                                                       "decay_min_lr")
    trainer.accelerator.state.deepspeed_plugin.hf_ds_config.fill_match("scheduler.params.decay_type", decay_type,
                                                                       "decay_type")


def train():
    #### show env
    # show_env_var()
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM

    parser = transformers.HfArgumentParser((PretrainModelArguments, PretrainDataArguments, PretrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Set seed before initializing model.
    set_seed(training_args.seed)

    if training_args.model_parallel_size > 1:
        # Disable CuDNN.
        torch.backends.cudnn.enabled = False
        init_mpu(training_args)

    tokenizer = AutoTokenizer.from_pretrained("model")
    if training_args.resume_from_checkpoint is None:
        model = build_pythia_model(model_args, training_args)
    else:
        model = AutoModelForCausalLM.from_pretrained(training_args.resume_from_checkpoint)

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    block_size = model_args.max_sequence_length
    print(f'Chunk block_size {block_size}')

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = []
        if training_args.read_cached:
            cached_dir_list = data_utils.traverse_cache_root(data_args.data_cache_dir)
            files = cached_dir_list
        else:
            path = Path(data_args.dataset_dir)
            files = [file.name for file in path.glob("*.json")]

        for idx, file in enumerate(files):
            if training_args.read_cached:
                processed_dataset = datasets.load_from_disk(file, keep_in_memory=False)
                print("load", file)
                logger.info(f'training datasets-{file} has been loaded from disk')
            else:
                data_file = os.path.join(path, file)
                filename = os.path.splitext(file)[0]
                cache_path = os.path.join(data_args.data_cache_dir, filename)
                os.makedirs(cache_path, exist_ok=True)
                try:
                    processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                    logger.info(f'training datasets-{filename} has been loaded from disk')
                except Exception:
                    cache_dir = os.path.join(data_args.data_cache_dir, filename + "_text")
                    os.makedirs(cache_dir, exist_ok=True)
                    raw_dataset = load_dataset("json", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                    logger.info(f"{file} has been loaded")
                    tokenized_dataset = raw_dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns="text",
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names={k: os.path.join(cache_dir, f'tokenized.arrow') for k in raw_dataset},
                        desc="Running tokenizer on dataset",
                    )
                    grouped_datasets = tokenized_dataset.map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names={k: os.path.join(cache_dir, f'grouped.arrow') for k in tokenized_dataset},
                        desc=f"Grouping texts in chunks of {block_size}",
                    )
                    processed_dataset = grouped_datasets
                    processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset['train']
            else:
                assert lm_datasets.features.type == processed_dataset["train"].features.type
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

    if training_args.do_train:
        train_dataset = lm_datasets
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
        print(f"Num train_samples  {len(train_dataset)}")

    print(training_args)
    # Initialize  Trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            tokenizer=tokenizer,
            data_collator=fault_tolerance_data_collator,
            callbacks=[ExitCallback, EpochEndCallback]
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        ## modify trainer_state.json
        change_trainer_state_file(training_args, checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    train()
