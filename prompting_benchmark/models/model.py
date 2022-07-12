# __init__ args:
# model name or smth
# temperature
# completion stop token
# max tokens?

# methods: complete(str) -> str
from typing import Optional, Union


class Model:
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        stop_tokens: list[str] = [],
        max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens

        self._post_init()

    def _post_init(self):
        return

    def complete(self, prompts: list[str]) -> Union[str, list[str]]:
        raise NotImplementedError()
