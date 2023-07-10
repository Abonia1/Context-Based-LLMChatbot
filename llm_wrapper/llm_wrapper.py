"""Fake LLM wrapper for testing purposes."""
from typing import Any, List, Mapping, Optional

import httpx
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class IdiomaLLM(LLM, ob_url: str):
    #ob_url: str = "https://voters-marsh-drawn-pledge.trycloudflare.com/api/v1/generate"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "idioma"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        with httpx.Client(timeout=60) as client:
            request_data = {
                "prompt": prompt,
                "max_new_tokens": 250,
                "do_sample": True,
                "temperature": 1.3,
                "top_p": 0.1,
                "typical_p": 1,
                "epsilon_cutoff": 0,
                "eta_cutoff": 0,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "min_length": 0,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "penalty_alpha": 0,
                "length_penalty": 1,
                "early_stopping": False,
                "seed": -1,
                "add_bos_token": True,
                "truncation_length": 2048,
                "ban_eos_token": False,
                "skip_special_tokens": True,
                "stopping_strings": [],
            }
            r = client.post(self.ob_url, json=request_data)
            data = r.json()
            return data.get("results")[0].get("text")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"ob_url": self.ob_url}