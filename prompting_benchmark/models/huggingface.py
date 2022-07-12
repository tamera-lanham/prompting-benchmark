from re import sub
from prompting_benchmark.models.model import Model
from transformers.generation_stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import GPTJForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import torch as t
from typing import Union

# StoppingCriteria info:
# https://github.com/huggingface/transformers/blob/09178705101b9803e7b9ea7f79a46b4c242dd4bf/src/transformers/generation_stopping_criteria.py
# https://github.com/huggingface/transformers/blob/09178705101b9803e7b9ea7f79a46b4c242dd4bf/src/transformers/generation_utils.py


class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids: list[int], prompt_lens: list[int]):
        self.stop_token_ids = stop_token_ids
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids: t.LongTensor, scores: t.FloatTensor, **kwargs) -> bool:
        return all(
            any(token_id in completion[prompt_len:] for token_id in self.stop_token_ids)
            for completion, prompt_len in zip(input_ids, self.prompt_lens)
        )


class HuggingfaceModel(Model):
    def _post_init(self):

        self.device = t.device("cuda:0")
        self.model, self.tokenizer = self._init_model_and_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if not isinstance(self.stop_tokens, list):
            raise ValueError("Stop tokens must be list")

        stop_token_ids_nested = self.tokenizer(self.stop_tokens + [self.tokenizer.eos_token])["input_ids"]
        self.stop_token_ids = [id_ for sublist in stop_token_ids_nested for id_ in sublist]

    def _stopping_criteria(self, prompt_lens):
        if self.stop_tokens:
            return StoppingCriteriaList([StopTokenCriteria(self.stop_token_ids, prompt_lens)])
        else:
            return StoppingCriteriaList([])

    def _truncate(self, output_ids: t.Tensor, prompt_len):
        # Remove everything after the first occurence of a stop token
        # TODO: confirm that the logits are the same in these two situations:
        # Token1 token2 token3 token4; attention_mask [1 1 1 1]
        # Token1 token2 <|eos|> token3 token4; attention_mask [1 1 0 1 1]
        prompt_id, completion_ids = output_ids[:prompt_len], output_ids[prompt_len:]

    def complete(self, prompts: list[str]) -> list[str]:

        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompts_ids = prompts_tokenized.input_ids.to(self.device)
        prompt_lens = [row.sum().item() for row in prompts_tokenized.attention_mask]

        outputs_ids = self.model.generate(
            prompts_ids,
            do_sample=True,
            temperature=self.temperature,
            max_length=self.max_tokens,
            stopping_criteria=self._stopping_criteria(prompt_lens),
            attention_mask=prompts_tokenized.attention_mask.to(self.device),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_ids_truncated = [output_ids for output_ids, prompt_len in zip(outputs_ids, prompt_lens)]
        outputs = self.tokenizer.batch_decode(output_ids_truncated)

        return outputs


class GPTJ(HuggingfaceModel):
    def _init_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", low_cpu_mem_usage=True).to(self.device)

        return model, tokenizer


class GPT2(HuggingfaceModel):
    def _init_model_and_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

        return model, tokenizer
