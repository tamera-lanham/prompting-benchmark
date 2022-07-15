from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from prompting_benchmark.models.model import Model
from transformers.generation_stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch as t
from typing import Iterable, Union

# StoppingCriteria info:
# https://github.com/huggingface/transformers/blob/09178705101b9803e7b9ea7f79a46b4c242dd4bf/src/transformers/generation_stopping_criteria.py
# https://github.com/huggingface/transformers/blob/09178705101b9803e7b9ea7f79a46b4c242dd4bf/src/transformers/generation_utils.py


class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids: list[int], prompt_lens: list[int]):
        self.stop_token_ids = stop_token_ids
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids: t.LongTensor, scores: t.FloatTensor, **kwargs) -> bool:
        return all(
            any(stop_token in completion[prompt_len:] for stop_token in self.stop_token_ids)
            for completion, prompt_len in zip(input_ids, self.prompt_lens)
        )


class HuggingfaceModel(Model):
    def _post_init(self, **kwargs):

        self.device = t.device(kwargs["device"]) if "device" in kwargs else t.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model_name in self.multi_gpu_models:
            self.model = self._get_model_multi_gpu(self.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        if not isinstance(self.stop_tokens, list):
            raise ValueError("Stop tokens must be list")

        stop_token_ids_nested = self.tokenizer(self.stop_tokens + [self.tokenizer.eos_token])["input_ids"]
        self.stop_token_ids = [id_ for sublist in stop_token_ids_nested for id_ in sublist]

    def _get_model_multi_gpu(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        checkpoint_path, no_split, device_map = self.multi_gpu_models[model_name]

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        model = load_checkpoint_and_dispatch(
            model, checkpoint_path, device_map=device_map, no_split_module_classes=no_split
        )

        return model

    def set_stop_tokens(self, stop_tokens: list[str]):
        self.stop_tokens = stop_tokens
        stop_token_ids_nested = self.tokenizer(self.stop_tokens + [self.tokenizer.eos_token])["input_ids"]
        self.stop_token_ids = [id_ for sublist in stop_token_ids_nested for id_ in sublist]

    def _stopping_criteria(self, prompt_lens):
        if isinstance(self.stop_tokens, list):
            return StoppingCriteriaList([StopTokenCriteria(self.stop_token_ids, prompt_lens)])
        else:  # Slightly different behavior than stop_tokens=[]; currently never triggered because stop_tokens isn't optional
            return StoppingCriteriaList([])

    def _trim(self, output_ids: t.Tensor, attn_mask: t.Tensor):
        # Remove everything that's after the first occurence of a stop token, or excluded by the attention mask

        def element_in(tensor: t.Tensor, set: Iterable) -> t.Tensor:
            return sum((tensor == element).to(t.bool) for element in set).to(t.bool)

        prompt_len = len(attn_mask)
        prompt_ids = output_ids[:prompt_len][attn_mask.to(t.bool)]

        completion_ids = output_ids[prompt_len:]
        stop_indices = element_in(completion_ids, self.stop_token_ids).nonzero()
        first_stop_index = stop_indices[0, 0].item() if stop_indices.numel() else None

        return t.cat([prompt_ids, completion_ids[:first_stop_index]])

    def complete(self, prompts: list[str], **generate_kwargs) -> list[str]:

        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompts_ids, attention_mask = tokens.input_ids, tokens.attention_mask

        if self.model_name not in self.multi_gpu_models:
            prompts_ids = prompts_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        prompt_lens = [row.sum().item() for row in tokens.attention_mask]

        outputs_ids = self.model.generate(
            prompts_ids,
            do_sample=True if self.temperature != 0 else False,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            stopping_criteria=self._stopping_criteria(prompt_lens),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs
        )
        output_ids_trimmed = [
            self._trim(output_ids, attn_mask) for output_ids, attn_mask in zip(outputs_ids, tokens.attention_mask)
        ]
        outputs = self.tokenizer.batch_decode(output_ids_trimmed)

        del prompts_ids
        del attention_mask

        return outputs

    multi_gpu_models = {
        "EleutherAI/gpt-j-6B": (
            "/home/ubuntu/mats-2022/sharded-gpt-j-6B",
            ["GPTJBlock"],
            {
                "transformer.wte": 0,
                "transformer.drop": 0,
                "transformer.h.0": 0,
                "transformer.h.1": 1,
                "transformer.h.2": 1,
                "transformer.h.3": 1,
                "transformer.h.4": 1,
                "transformer.h.5": 1,
                "transformer.h.6": 1,
                "transformer.h.7": 1,
                "transformer.h.8": 1,
                "transformer.h.9": 1,
                "transformer.h.10": 2,
                "transformer.h.11": 2,
                "transformer.h.12": 2,
                "transformer.h.13": 2,
                "transformer.h.14": 2,
                "transformer.h.15": 2,
                "transformer.h.16": 2,
                "transformer.h.17": 2,
                "transformer.h.18": 2,
                "transformer.h.19": 2,
                "transformer.h.20": 3,
                "transformer.h.21": 3,
                "transformer.h.22": 3,
                "transformer.h.23": 3,
                "transformer.h.24": 3,
                "transformer.h.25": 3,
                "transformer.h.26": 3,
                "transformer.h.27": 3,
                "transformer.ln_f": 3,
                "lm_head": 3,
            },
        )
    }
