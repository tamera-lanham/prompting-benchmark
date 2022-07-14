from re import sub
from prompting_benchmark.models.model import Model
from transformers.generation_stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    def _post_init(self):

        self.device = t.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        if not isinstance(self.stop_tokens, list):
            raise ValueError("Stop tokens must be list")

        stop_token_ids_nested = self.tokenizer(self.stop_tokens + [self.tokenizer.eos_token])["input_ids"]
        self.stop_token_ids = [id_ for sublist in stop_token_ids_nested for id_ in sublist]

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
        prompts_ids = tokens.input_ids.to(self.device)
        prompt_lens = [row.sum().item() for row in tokens.attention_mask]

        outputs_ids = self.model.generate(
            prompts_ids,
            do_sample=True if self.temperature != 0 else False,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            stopping_criteria=self._stopping_criteria(prompt_lens),
            attention_mask=tokens.attention_mask.to(self.device),
            pad_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs
        )
        output_ids_trimmed = [
            self._trim(output_ids, attn_mask) for output_ids, attn_mask in zip(outputs_ids, tokens.attention_mask)
        ]
        outputs = self.tokenizer.batch_decode(output_ids_trimmed)

        return outputs
