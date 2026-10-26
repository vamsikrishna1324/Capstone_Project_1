-- Migration: 001_simplify_schema.sql
-- Simplifies benchmark_runs (11 -> 10 columns) and model_measurements (44 -> 21 columns)
-- Drops redundant secondary diagnostics, unused file paths, and obsolete hash columns.

-- 1. Simplify parent table: benchmark_runs (11 -> 10 columns)
ALTER TABLE `benchmark_runs`
    DROP COLUMN `artifact_hash`;

-- 2. Simplify child table: model_measurements (44 -> 21 columns)
ALTER TABLE `model_measurements`
    DROP COLUMN `weights`,
    DROP COLUMN `backend_provider`,
    DROP COLUMN `input_shape`,
    DROP COLUMN `dataset_path`,
    DROP COLUMN `total_samples`,
    DROP COLUMN `skipped_images`,
    DROP COLUMN `accuracy_evaluation_time_seconds`,
    DROP COLUMN `accuracy_throughput_samples_per_sec`,
    DROP COLUMN `latency_median_ms`,
    DROP COLUMN `latency_std_ms`,
    DROP COLUMN `latency_min_ms`,
    DROP COLUMN `latency_max_ms`,
    DROP COLUMN `latency_p99_ms`,
    DROP COLUMN `ram_delta_mb`,
    DROP COLUMN `baseline_ram_mb`,
    DROP COLUMN `post_load_ram_mb`,
    DROP COLUMN `inference_ram_delta_mb`,
    DROP COLUMN `device_memory_mb`,
    DROP COLUMN `memory_notes`,
    DROP COLUMN `file_size_bytes`,
    DROP COLUMN `artifact_path`,
    DROP COLUMN `artifact_version`,
    DROP COLUMN `config_json`;
