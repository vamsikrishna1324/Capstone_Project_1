-- ==============================================================================
-- Automated Image Model Benchmarking for Edge Deployment
-- MySQL Database Schema DDL (Step 5 Persistence Layer - Simplified Schema)
-- ==============================================================================
-- Character Set: utf8mb4
-- Collation:     utf8mb4_unicode_ci
-- Storage Engine: InnoDB
-- ==============================================================================

-- Create default database if needed
CREATE DATABASE IF NOT EXISTS `edge_model_benchmark`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `edge_model_benchmark`;

-- ------------------------------------------------------------------------------
-- Table 1: benchmark_runs (10 Columns)
-- Header table representing benchmark execution or comparison sessions.
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `benchmark_runs` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `run_name` VARCHAR(255) NULL COMMENT 'Human-readable title for the comparison run',
    `description` TEXT NULL COMMENT 'Optional notes or user tags',
    `device` VARCHAR(64) NOT NULL DEFAULT 'cpu' COMMENT 'Target hardware device (cpu, mps, etc.)',
    `status` VARCHAR(32) NOT NULL DEFAULT 'completed' COMMENT 'Execution status: pending, running, completed, failed',
    `error_message` TEXT NULL COMMENT 'Error message if run execution failed',
    `environment_info` JSON NULL COMMENT 'Hardware, OS, and library versions in JSON format',
    `provenance` VARCHAR(32) NOT NULL DEFAULT 'live_execution' COMMENT 'Source origin: live_execution or imported_artifact',
    `created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Run initialization timestamp',
    `completed_at` DATETIME(6) NULL COMMENT 'Run completion timestamp',
    INDEX `ix_benchmark_runs_created_at` (`created_at` DESC),
    INDEX `ix_benchmark_runs_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- Table 2: model_measurements (21 Columns)
-- Granular benchmark performance measurements per model and runtime.
-- Supports associating multiple models and runtimes under one comparison run.
-- Unavailable metrics are explicitly stored as NULL (never invented zeros).
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `model_measurements` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `run_id` INT NOT NULL COMMENT 'Foreign key to parent benchmark_runs header',

    -- Model Identification
    `model_id` VARCHAR(64) NOT NULL COMMENT 'Registered model identifier (e.g., resnet50)',
    `architecture` VARCHAR(128) NULL COMMENT 'Display architecture name (e.g., ResNet-50)',

    -- Runtime & Execution Settings
    `runtime` VARCHAR(32) NOT NULL COMMENT 'Target runtime: pytorch, onnx, tflite',
    `device` VARCHAR(32) NOT NULL DEFAULT 'cpu' COMMENT 'Execution device: cpu, mps, cuda',
    `batch_size` INT NOT NULL DEFAULT 1 COMMENT 'Inference batch size',
    `warmup_runs` INT NULL COMMENT 'Number of warm-up iterations executed',
    `num_runs` INT NULL COMMENT 'Number of timed measured iterations',
    `status` VARCHAR(32) NOT NULL DEFAULT 'completed' COMMENT 'Measurement status: completed, failed',
    `error_message` TEXT NULL COMMENT 'Measurement-level error details if failed',

    -- Accuracy Metrics (NULL if not evaluated)
    `evaluated_samples` INT NULL COMMENT 'Evaluated validation samples count',
    `top1_accuracy` FLOAT NULL COMMENT 'Top-1 accuracy as fraction (0.0 to 1.0)',
    `top5_accuracy` FLOAT NULL COMMENT 'Top-5 accuracy as fraction (0.0 to 1.0)',

    -- Latency & Throughput Metrics (NULL if not evaluated)
    `latency_mean_ms` FLOAT NULL COMMENT 'Arithmetic mean latency per batch in ms',
    `latency_p95_ms` FLOAT NULL COMMENT '95th percentile latency in ms',
    `throughput_fps` FLOAT NULL COMMENT 'Inference throughput in frames per second',

    -- Memory Footprint Metrics (NULL if not evaluated)
    `peak_ram_mb` FLOAT NULL COMMENT 'Peak process Resident Set Size (RSS) in MiB',

    -- Storage & Parameter Metrics (NULL if not evaluated; PyTorch in-memory retains NULL)
    `file_size_mb` FLOAT NULL COMMENT 'Serialized on-disk artifact size in MiB (NULL for in-memory PyTorch)',
    `parameter_count` BIGINT NULL COMMENT 'Total trainable and non-trainable parameter count',

    -- Timestamps
    `created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'Measurement timestamp',

    -- Constraints & Indexes
    CONSTRAINT `fk_model_measurements_run`
        FOREIGN KEY (`run_id`)
        REFERENCES `benchmark_runs` (`id`)
        ON DELETE CASCADE,

    INDEX `ix_model_measurements_run_id` (`run_id`),
    INDEX `ix_model_measurements_model_id` (`model_id`),
    INDEX `ix_model_measurements_runtime` (`runtime`),
    INDEX `ix_model_measurements_lookup` (`model_id`, `runtime`, `device`),
    INDEX `ix_model_measurements_run_model` (`run_id`, `model_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
