"""Configuration loading for PyEval."""

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class EvalConfig:
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "default"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 30
    sandbox_timeout: int = 10
    max_concurrent_api: int = 4
    max_concurrent_sandbox: int = 8
    categories: list[str] = field(default_factory=list)
    difficulties: list[str] = field(default_factory=list)
    problem_ids: list[str] = field(default_factory=list)
    verbose: bool = False
    dry_run: bool = False
    extra_body: dict = field(default_factory=dict)
    output_dir: str = "pyeval_results"
    problems_dir: str = ""
    mode: str = "standard"
    max_attempts: int = 3

    def __post_init__(self):
        if not self.problems_dir:
            self.problems_dir = str(
                Path(__file__).parent / "problems" / "bank"
            )
        if self.mode not in ("standard", "bugfix", "multiturn"):
            raise ValueError(f"Invalid mode: {self.mode!r}. Must be 'standard', 'bugfix', or 'multiturn'.")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")


def load_config(config_path: str | None = None) -> EvalConfig:
    """Load config from JSON file, returning defaults if file not found."""
    if config_path is None:
        default_path = Path(__file__).parent / "pyeval.json"
        if default_path.exists():
            config_path = str(default_path)
        else:
            return EvalConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = json.load(f)

    return EvalConfig(**{k: v for k, v in data.items() if k in EvalConfig.__dataclass_fields__})


def merge_cli_args(config: EvalConfig, args: dict) -> EvalConfig:
    """Override config fields with non-None CLI arguments."""
    for key, value in args.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config
