-- Rollback Migration: 001_simplify_schema_rollback.sql
-- Restores all dropped columns to benchmark_runs and model_measurements

-- 1. Restore artifact_hash to benchmark_runs
ALTER TABLE `benchmark_runs`
    ADD COLUMN `artifact_hash` VARCHAR(64) NULL,
    ADD UNIQUE KEY `ix_benchmark_runs_artifact_hash` (`artifact_hash`);

-- 2. Restore 23 columns to model_measurements
ALTER TABLE `model_measurements`
    ADD COLUMN `weights` VARCHAR(128) NULL AFTER `architecture`,
    ADD COLUMN `backend_provider` VARCHAR(128) NULL AFTER `device`,
    ADD COLUMN `input_shape` VARCHAR(64) NULL AFTER `num_runs`,
    ADD COLUMN `dataset_path` VARCHAR(512) NULL AFTER `error_message`,
    ADD COLUMN `total_samples` INT NULL AFTER `dataset_path`,
    ADD COLUMN `skipped_images` INT NULL AFTER `evaluated_samples`,
    ADD COLUMN `accuracy_evaluation_time_seconds` FLOAT NULL AFTER `top5_accuracy`,
    ADD COLUMN `accuracy_throughput_samples_per_sec` FLOAT NULL AFTER `accuracy_evaluation_time_seconds`,
    ADD COLUMN `latency_median_ms` FLOAT NULL AFTER `latency_mean_ms`,
    ADD COLUMN `latency_std_ms` FLOAT NULL AFTER `latency_median_ms`,
    ADD COLUMN `latency_min_ms` FLOAT NULL AFTER `latency_std_ms`,
    ADD COLUMN `latency_max_ms` FLOAT NULL AFTER `latency_min_ms`,
    ADD COLUMN `latency_p99_ms` FLOAT NULL AFTER `latency_p95_ms`,
    ADD COLUMN `ram_delta_mb` FLOAT NULL AFTER `peak_ram_mb`,
    ADD COLUMN `baseline_ram_mb` FLOAT NULL AFTER `ram_delta_mb`,
    ADD COLUMN `post_load_ram_mb` FLOAT NULL AFTER `baseline_ram_mb`,
    ADD COLUMN `inference_ram_delta_mb` FLOAT NULL AFTER `post_load_ram_mb`,
    ADD COLUMN `device_memory_mb` FLOAT NULL AFTER `inference_ram_delta_mb`,
    ADD COLUMN `memory_notes` TEXT NULL AFTER `device_memory_mb`,
    ADD COLUMN `file_size_bytes` BIGINT NULL AFTER `memory_notes`,
    ADD COLUMN `artifact_path` VARCHAR(512) NULL AFTER `parameter_count`,
    ADD COLUMN `artifact_version` VARCHAR(64) NULL AFTER `artifact_path`,
    ADD COLUMN `config_json` JSON NULL AFTER `artifact_version`;
