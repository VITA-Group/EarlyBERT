# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Execute the ticket-drawing and efficient training stage of EarlyBERT on the
finetuning task for text classification tasks on GLUE benchmark.

This code is based on the implementation by The HuggingFace Inc. team.
Modified by:
    - Xiaohan Chen
    - xiaohan.chen@utexas.edu
    - Last modified on: Aug 25, 2021
"""


import dataclasses
import logging
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

import ipdb
import copy
import math

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from trainer import Trainer
from training_args import TrainingArguments

from transformers.modeling_bert import BertSelfAttention, BertAttention, BertLayer

from utils import get_pruning_mask

logger = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up handler for logging
    from utils import set_logging_config
    set_logging_config(training_args.output_dir)
    global logger
    logger = logging.getLogger(__name__)
    
    # make sure that we will not perform the pruning twice
    assert training_args.prune_before_train and (not training_args.prune_before_eval)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if training_args.load_from_pruned:
        new_config = copy.deepcopy(config)
        new_config.self_pruning_ratio = training_args.self_pruning_ratio
        new_config.inter_pruning_ratio = training_args.inter_pruning_ratio
    else:
        new_config = config

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=new_config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer)
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev") if training_args.do_eval else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test") if training_args.do_predict else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Calculate at which step we draw the ticket. We provide several ways to
    # determine this depending on the value of the argument `args.slimming_coef_step`
    # Let t = args.slimming_coef_step (float type)
    #
    #                       /-- int(t), if t >= 1
    # step to draw ticket =  -- round(t * num_steps_per_epoch), if 0 < t < 1
    #                       \-- floor(t), if t <= 0, used as random seed for random pruning exp
    #
    # If t >= 1, we will just use `t` as the step index and draw the ticket at that step.
    # If 0 < t < 1, `t` represents that we draw the ticket after (t*100)% of the
    # total number of steps in the first epoch.
    # If t <= 0, we apply random pruning and use `-floor(t)` as the random seed for pruning.
    if training_args.slimming_coef_step >= 1.0:
        training_args.slimming_coef_step = int(training_args.slimming_coef_step)
    elif training_args.slimming_coef_step > 0.0:
        num_steps_per_epoch = len(train_dataset) / training_args.train_batch_size
        training_args.slimming_coef_step = round(training_args.slimming_coef_step * num_steps_per_epoch)
    else:
        training_args.slimming_coef_step = math.floo(training_args.slimming_coef_step)

    # Training
    if training_args.do_train:
        # Prune intermediate neurons in FFN modules based on the learnable coefficients
        bert_layers = []
        for m in model.modules():
            if isinstance(m, BertLayer):
                bert_layers.append(m)
        # Get the coefficients for pruning, which has shape (num_hidden_layers, num_inter_neurons)
        if training_args.slimming_coef_step > 0:
            slimming_coefs = np.load(training_args.inter_slimming_coef_file)[:, training_args.slimming_coef_step-1, :]
        else:
            # Random pruning
            # Get internal state of the random generator first
            rand_state = np.random.get_state()
            # Set random seed
            np.random.seed(-training_args.slimming_coef_step)
            slimming_coefs = np.random.rand(
                len(bert_layers), bert_layers[0].intermediate.dense.out_features)
            # Reset internal state
            np.random.set_state(rand_state)
        # If we do layerwise pruning, calculate the threshold along the last dimension
        # of `slimming_coefs`, which corresponds to the self-attention heads in each layer;
        # otherwise, calculate the threshold along all dimensions in `slimming_coefs`.
        quantile_axis = -1 if training_args.inter_pruning_method == 'layerwise' else None
        threshold = np.quantile(slimming_coefs, training_args.inter_pruning_ratio, axis=quantile_axis, keepdims=True)
        layers_masks = slimming_coefs > threshold
        for m, mask in zip(bert_layers, layers_masks):
            pruned_inter_neurons = [i for i in range(new_config.intermediate_size) if mask[i] == 0]
            logger.info('{} neurons are pruned'.format(len(pruned_inter_neurons)))
            m.prune_inter_neurons(pruned_inter_neurons)

        # Prune self-attention heads based on the learnable coefficients
        attention_modules = []
        # slimming_coefs = []
        for m in model.modules():
            if isinstance(m, BertAttention):
                attention_modules.append(m)
        # Get the coefficients for pruning, which has shape (num_hidden_layers, num_attention_heads)
        if training_args.slimming_coef_step > 0:
            slimming_coefs = np.load(training_args.self_slimming_coef_file)[:, training_args.slimming_coef_step-1, :]
        else:
            # random pruning
            # get internal state of the random generator first
            rand_state = np.random.get_state()
            # set random seed
            np.random.seed(-training_args.slimming_coef_step)
            slimming_coefs = np.random.rand(len(attention_modules), new_config.num_attention_heads)
            # reset internal state
            np.random.set_state(rand_state)
        # If we do layerwise pruning, calculate the threshold along the last dimension
        # of `slimming_coefs`, which corresponds to the self-attention heads in each layer;
        # otherwise, calculate the threshold along all dimensions in `slimming_coefs`.
        quantile_axis = -1 if training_args.self_pruning_method == 'layerwise' else None
        threshold = np.quantile(slimming_coefs, training_args.self_pruning_ratio, axis=quantile_axis, keepdims=True)
        layers_masks = slimming_coefs > threshold
        for m, mask in zip(attention_modules, layers_masks):
            pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]
            logger.info('pruned heads: {}'.format(str(pruned_heads)))
            m.prune_heads(pruned_heads)

        train_outputs = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
            config=config, l1_loss_coef=0.0, lottery_ticket_training=True
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    # NOTE: Currently, we only support evaluation after training.
    # TODO: Add support of evaluating pre-trained checkpoints without training
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev")
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test")
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
