"""Token / call accounting for the optional LLM step (default run makes zero LLM calls)."""
import json
from dataclasses import asdict, dataclass


@dataclass
class CostLog:
    mode: str = "off"  # off | dry-run | live
    model: str = "none"
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_are_estimates: bool = False
    fallbacks_to_template: int = 0
    price_in_per_million_usd: float = 0.0
    price_out_per_million_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, estimated: bool = False):
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.tokens_are_estimates = self.tokens_are_estimates or estimated

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * self.price_in_per_million_usd
                + self.output_tokens * self.price_out_per_million_usd) / 1_000_000

    def to_json(self, path):
        data = asdict(self)
        data["estimated_cost_usd"] = round(self.cost_usd, 6)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data


def estimate_tokens(text: str) -> int:
    """Rough rule of thumb (~4 characters per token) used only for dry-run estimates."""
    return max(1, round(len(text) / 4))
