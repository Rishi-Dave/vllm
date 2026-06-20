# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.pythonic_tool_parser import PythonicToolParser


class Llama4PythonicToolParser(PythonicToolParser):
    """
    Toolcall parser for Llama4 that produce tool calls in a pythonic style
    Use --enable-auto-tool-choice --tool-call-parser llama4_pythonic
    """

    _PYTHON_START = "<|python_start|>"
    _PYTHON_END = "<|python_end|>"

    def _preprocess_model_output(self, model_output: str) -> str:
        # remove <|python_start|> and <|python_end|>
        # as Llama 4 model sometime will output those tokens
        if model_output.startswith(self._PYTHON_START):
            model_output = model_output[len(self._PYTHON_START) :]
            model_output = model_output.replace(self._PYTHON_END, "")
        return model_output

    def _preprocess_streaming_text(self, current_text: str) -> str:
        if current_text.startswith(self._PYTHON_START):
            current_text = current_text[len(self._PYTHON_START) :]
        if current_text.endswith(self._PYTHON_END):
            current_text = current_text[: current_text.rfind(self._PYTHON_END)]
        return current_text

    def _streaming_prefix_allowed(self, current_text: str) -> bool:
        return current_text.startswith("[") or current_text.startswith(
            self._PYTHON_START
        )
