"""Offline accuracy evaluation utilities for the annotation pipeline."""

from annotation_pipeline_skill.eval.consensus_accuracy import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PASSES,
    DEFAULT_WORKERS,
    ConsensusResult,
    build_qc_judge,
    count_spans,
    evaluate_accuracy,
    run_consensus,
    run_single_pass,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_PASSES",
    "DEFAULT_WORKERS",
    "ConsensusResult",
    "build_qc_judge",
    "count_spans",
    "evaluate_accuracy",
    "run_consensus",
    "run_single_pass",
]
