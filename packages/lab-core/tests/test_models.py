"""Tests for lab-core Pydantic models."""

from lab_core.models import ModelSpec, GPUConfig, BenchmarkResult, EvalResult


def test_model_spec_defaults():
    spec = ModelSpec(name="test", repo_id="org/model")
    assert spec.architecture == "dense"
    assert spec.gpu_config == GPUConfig.SINGLE_0
    assert spec.cuda_graphs is True
    assert spec.max_context == 131072


def test_model_spec_moe():
    spec = ModelSpec(
        name="qwen-35b",
        repo_id="Qwen/Qwen3.5-35B-A3B",
        architecture="moe",
        total_params_b=35,
        active_params_b=3,
    )
    assert spec.architecture == "moe"
    assert spec.active_params_b == 3


def test_benchmark_result():
    result = BenchmarkResult(model="test", task="T02", score=0.87, passed=True, tokens_per_sec=44.0)
    assert result.score == 0.87
    assert result.tokens_per_sec == 44.0


def test_eval_result():
    result = EvalResult(
        model="qwen-27b-int4",
        suite="claw",
        scores={"T02": 0.87, "T06": 0.89},
        pass_rate=0.75,
        avg_score=0.79,
    )
    assert result.pass_rate == 0.75
    assert result.pass_k == 3
