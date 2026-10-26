# Automated Image Model Benchmarking for Edge Deployment

A two-semester B.Tech capstone project building an automated framework to convert, benchmark, analyze, and recommend pretrained computer vision models for resource-constrained edge hardware.

---

## Project Overview

Deploying deep learning models to edge devices—such as embedded systems, mobile devices, and IoT nodes—requires balancing operational performance against strict hardware boundaries. This project provides an automated, end-to-end benchmarking framework where users can select standard pretrained image classification models, convert them into optimized edge-deployment formats, benchmark them under controlled conditions, store and visualize performance metrics, and receive tailored model recommendations.

---

## Problem Statement

Deploying computer vision models on edge platforms introduces complex trade-offs between:
- **Inference Latency** (milliseconds per sample / frames per second)
- **Memory Footprint** (RAM usage / peak memory during execution)
- **Model Storage Footprint** (disk size / parameter count / quantization precision)
- **Task Accuracy** (Top-1 / Top-5 classification metrics)

Different edge execution runtimes (such as PyTorch, ONNX Runtime, and TensorFlow Lite) exhibit distinct performance characteristics across target architectures. Manually converting, profiling, and comparing these models is labor-intensive, error-prone, and inconsistent. There is a critical need for a unified, automated benchmarking harness that systematically evaluates these models and assists developers in selecting the optimal configuration for their deployment constraints.

---

## Objectives

1. **Model Management**: Ingest and manage pretrained image classification models with standardized preprocessing pipelines.
2. **Format Conversion**: Automate conversions across edge formats (e.g., PyTorch to ONNX and TensorFlow Lite).
3. **Controlled Benchmarking**: Profile runtime performance (latency, throughput, memory consumption) under isolated and reproducible test conditions.
4. **Persistent Metrics Storage**: Store structured benchmark results, hardware specs, and model metadata for historical tracking.
5. **Interactive Visualization**: Provide a web-based dashboard for exploring comparative metrics and trends.
6. **Preference-Based Recommendation**: Help users select optimal models based on their specific priorities (e.g., maximizing accuracy, minimizing latency, minimizing memory footprint, or reducing model size).

---

## Supported Models

The framework currently supports four standard computer vision models using official `torchvision` implementations and official pretrained ImageNet-1K weights (`DEFAULT` enums):

| Model ID | Display Name | Architecture | Torchvision Weights Enum | Input Size | Classes |
|---|---|---|---|---|---|
| `mobilenet_v3_small` | MobileNetV3-Small | MobileNetV3 | `MobileNet_V3_Small_Weights.DEFAULT` | (3, 224, 224) | 1000 |
| `resnet50` | ResNet50 | ResNet | `ResNet50_Weights.DEFAULT` | (3, 224, 224) | 1000 |
| `efficientnet_b0` | EfficientNet-B0 | EfficientNet | `EfficientNet_B0_Weights.DEFAULT` | (3, 224, 224) | 1000 |
| `densenet121` | DenseNet121 | DenseNet | `DenseNet121_Weights.DEFAULT` | (3, 224, 224) | 1000 |

Each model is ingested with its official torchvision preprocessing transforms (`weights.transforms()`), ensuring standardization across resize dimensions, center cropping, normalization means, and standard deviations.

---

## Planned System Modules

- `backend/`: FastAPI application exposing RESTful APIs to initiate conversions, execute benchmarks, and query results.
- `benchmarking/`: Core benchmarking harness responsible for running inference loops, recording execution latency, measuring throughput, and tracking memory usage.
- `conversion/`: Format conversion engines transforming PyTorch models into target edge formats (ONNX, TFLite).
- `database/`: Persistence layer (e.g., SQLite via SQLAlchemy) storing benchmark run records, model architectures, and performance metrics.
- `dataset/`: Sample validation datasets (e.g., ImageNet subsets) for evaluation and sanity checking.
- `frontend/`: Single-page web dashboard (React / Vite) for interacting with the benchmarking suite, triggering runs, and viewing visual analytics.
- `models/`: Model registry (`model_registry.py`), loader (`model_loader.py`), and verifier (`model_verifier.py`) for supported computer vision architectures.
- `recommendation/`: Preference-based recommendation engine that ranks models according to user-selected priorities (accuracy, latency, memory usage, and model size). Pareto frontier analysis is planned as a possible additional decision-support method to identify non-dominated trade-offs.
- `results/`: Output store for generated benchmark reports, exported charts, and raw execution logs.
- `tests/`: Automated test suite covering environment validation (`test_environment.py`), model tests (`test_models.py`), and future integration workflows.

---

## Planned Technology Stack

| Layer | Technologies |
|---|---|
| **Programming Language** | Python 3.11 |
| **Deep Learning Framework** | PyTorch, Torchvision |
| **Edge Runtimes / Formats** | ONNX, ONNX Runtime, TensorFlow / TensorFlow Lite |
| **Backend API** | FastAPI, Uvicorn |
| **Data & Storage** | SQLite, SQLAlchemy, NumPy, Pandas |
| **Frontend UI** | React, Vite, Vanilla CSS |
| **Environment Management** | `uv` / Python Virtual Environment (`.venv311`) |

---

## Development Status

> [!NOTE]
> **Current Status: Step 4D.6 (ImageNet Validation Image Extraction & Manifest Generation) Complete**
> - **Step 1 (Foundation Setup)**: Project directory scaffolding, version control configuration, Python 3.11 environment (`.venv311`), and initial environment validation completed and verified.
> - **Step 2 (Model Selection & Verification)**:
>   - Single-source-of-truth model registry (`models/model_registry.py`) providing structured metadata for `mobilenet_v3_small`, `resnet50`, `efficientnet_b0`, and `densenet121`.
>   - Modular model loader (`models/model_loader.py`) with automatic hardware acceleration detection (`mps` / `cpu`), evaluation mode enforcement, and official weights transform retrieval.
>   - Structured model verifier (`models/model_verifier.py`) executing deterministic dummy input preprocessing and single-pass forward inference.
>   - Comprehensive test suite (`tests/test_models.py`) with 35 passing tests across environment, registry, loading, evaluation mode, preprocessing, inference, and error handling.
> - **Step 3A (PyTorch → ONNX Conversion & Verification)**:
>   - Automated ONNX export module (`conversion/onnx_converter.py`) utilizing TorchScript tracing (`dynamo=False`) with constant folding and target opset 17.
>   - ONNX Runtime verification engine validating graph structure, input tensor mapping `[1, 3, 224, 224]`, and logits output shape `[1, 1000]`.
>   - Verified exported ONNX models stored in `results/conversions/onnx/`.
>   - Test suite (`tests/test_onnx_conversion.py`) with 13 passing tests.
> - **Step 3B (PyTorch → TFLite Conversion & Verification)**:
>   - Automated TFLite conversion pipeline (`conversion/tflite_converter.py`) via `onnx2tf` bridging ONNX opset 17 graphs to TensorFlow Lite flatbuffers.
>   - TensorFlow Lite Interpreter verification engine validating tensor allocation, NHWC input execution, and logits output shape `[1, 1000]`.
>   - Deterministic offline conversion pipeline with isolated staging directories and dummy test-data mocking.
>   - Verified converted TFLite models stored in `results/conversions/tflite/`.
>   - Test suite (`tests/test_tflite_conversion.py`) with 13 passing tests.
> - **Step 4A (Benchmarking Engine Foundation)**:
>   - Benchmarking configuration structure (`benchmarking/config.py`) defining `BenchmarkRuntime` enum (`pytorch`, `onnx`, `tflite`), iteration controls, batch sizes, tensor shapes, dataset parameters (`dataset_path`, `max_samples`), and strict validation.
>   - Structured result containers (`benchmarking/results.py`) defining `BenchmarkResult`, `AccuracyMetrics`, `LatencyMetrics`, `MemoryMetrics`, and `ModelSizeMetrics` with dictionary serialization and human-readable summaries.
>   - Benchmarking engine foundation (`benchmarking/engine.py`) defining `BaseBenchmarkEngine` interface and `BenchmarkEngine` with automated hardware/environment discovery.
>   - Unit test suite (`tests/test_benchmarking.py`) with 69 passing tests across configuration, validation, metrics handling, engine initialization, and deferred execution stubs.
> - **Step 4B-1 (Model Size Benchmarking)**:
>   - Model size benchmarking module (`benchmarking/model_size.py`) measuring disk storage footprints for ONNX and TFLite flatbuffers, and calculating theoretical parameter counts from PyTorch architectures.
>   - PyTorch in-memory model handling: explicitly leaves serialized file size as `None` when no serialized checkpoint file exists, avoiding synthetic estimations or unnecessary file duplication.
>   - Byte-to-MB consistency verification across all supported architectures and formats.
>   - Full integration with `ModelSizeMetrics`, `BenchmarkConfig`, and `BenchmarkEngine.run_benchmark()`.
>   - Comprehensive test suite (`tests/test_model_size.py`) with 39 passing unit and end-to-end tests.
> - **Step 4B-2 (Inference Latency Benchmarking)**:
>   - Latency benchmarking module (`benchmarking/latency.py`) implementing high-resolution monotonic timing (`time.perf_counter()`), warm-up execution exclusion, and accelerator synchronization (`torch.mps.synchronize()`).
>   - Cross-runtime inference profiling supporting PyTorch (CPU and Apple Silicon MPS), ONNX Runtime (CPUExecutionProvider), and TensorFlow Lite (`tf.lite.Interpreter`).
>   - Input layout handling: NCHW `(batch_size, 3, 224, 224)` for PyTorch/ONNX with static batch validation, and NHWC `(batch_size, 224, 224, 3)` for TFLite with dynamic batch resizing.
>   - Distribution statistics and throughput: computes `mean_ms`, `median_ms`, `std_ms`, `min_ms`, `max_ms`, `p95_ms`, `p99_ms`, and `throughput_fps` (`(batch_size * 1000.0) / mean_ms`).
>   - Comprehensive test suite (`tests/test_latency.py`) with 31 passing unit and integration tests.
> - **Step 4B-3 (Memory Footprint Benchmarking)**:
>   - Memory benchmarking module (`benchmarking/memory.py`) measuring process-level Resident Set Size (RSS) memory consumption in MiB using `psutil`.
>   - Distinguishes baseline memory (prior to model loading, after garbage collection), post-load memory, peak memory during measured inference, total RAM delta, and inference RAM delta.
>   - Cross-runtime memory profiling for PyTorch CPU, PyTorch Apple Silicon MPS, ONNX Runtime CPU, and TensorFlow Lite CPU.
>   - For PyTorch MPS, captures unified memory allocation via `torch.mps.current_allocated_memory()` while explicitly reporting that unified memory is shared with host RAM and does not represent discrete GPU VRAM.
>   - Comprehensive test suite (`tests/test_memory.py`) with 27 passing tests.
> - **Step 4C (ImageNet Accuracy Benchmarking Engine)**:
>   - Accuracy benchmarking module (`benchmarking/accuracy.py`) calculating exact Top-1 and Top-5 classification metrics (`compute_topk_accuracy`).
>   - Cross-runtime evaluation engine supporting PyTorch (`evaluate_pytorch_accuracy` with CPU/MPS and mini-batching), ONNX Runtime (`evaluate_onnx_accuracy` with NCHW layout and static batch size 1), and TensorFlow Lite (`evaluate_tflite_accuracy` with NHWC channel-last transpose).
>   - Preprocessing parity: uses official torchvision preprocessing transforms matching model registry weights metadata (`weights.transforms()`).
>   - Deterministic dataset abstraction (`EvaluationDataset`, `LabeledImageSample`, `resolve_dataset`): supports synthetic validation data (`create_deterministic_synthetic`), directory-based ImageFolder hierarchies, and sample lists without requiring multi-gigabyte dataset downloads.
>   - Strict honest accuracy reporting: reports accuracy only for evaluated sample counts; never substitutes theoretical or official torchvision metadata accuracy.
>   - Host machine clarification: benchmarks run on host development hardware (MacBook Air Apple Silicon) as software-level edge simulations, not physical embedded target nodes.
>   - Comprehensive test suite (`tests/test_accuracy.py`) with 29 passing tests.
> - **Step 4D.2 (ImageNet Validation Manifest Loader)**:
>   - Dedicated manifest parsing and validation utility (`benchmarking/imagenet_manifest.py`) supporting CSV (`image_relpath`, `true_label_idx`) and JSON formats (lists, sample dictionaries, key-value mappings).
>   - Strict input validation: ensures image files exist on disk, validates image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`), enforces class index bounds (`0 <= label <= 999`), and rejects duplicate image paths or malformed syntax with explicit error messages.
>   - External WNID mapping support (`validate_wnid_mapping`) allowing synset translation without hardcoded labels or heuristic guessing.
>   - Direct integration with `EvaluationDataset` (`EvaluationDataset.from_manifest()`) and `resolve_dataset()`, allowing manifest file paths to be passed directly to `BenchmarkConfig.dataset_path`.
>   - Comprehensive test suite (`tests/test_manifest.py`) with 32 passing tests.
> - **Step 4D.5 (ImageNet Validation Metadata Extraction & Mapping)**:
>   - Metadata extraction module (`benchmarking/imagenet_metadata.py`) designed to safely extract only `ILSVRC2012_validation_ground_truth.txt` and `meta.mat` from `dataset/raw/ILSVRC2012_devkit_t12.tar.gz` into `dataset/metadata/`.
>   - Security checks: strictly validates member paths, rejects parent traversal (`..`), absolute paths, and symlinks, avoiding dangerous `extractall()`.
>   - Ground-truth parser: validates exactly 50,000 validation labels in range `1..1000` with balanced 50 samples/class distribution check.
>   - Synset mapping parser: extracts the 1,000 leaf synsets (`num_children == 0`) from `meta.mat` and derives the canonical 0-based torchvision class index mapping (`0..999`) based on alphabetical sort of WNIDs.
>   - CLI extraction runner (`benchmarking/extract_metadata.py`) executed and verified, generating `dataset/metadata/devkit_to_torchvision_mapping.json`.
>   - Comprehensive test suite (`tests/test_metadata.py`) with 24 passing tests.
> - **Step 4D.6 (ImageNet Validation Image Extraction & Manifest Generation)**:
>   - Image extraction and manifest engine (`benchmarking/imagenet_images.py`) implementing safe, chunk-by-chunk streaming extraction from `dataset/raw/ILSVRC2012_img_val.tar` into `dataset/val/` without calling `extractall()`.
>   - Safe Resume & Overwrite Protection: Never blindly overwrites existing files. Skips existing images only if byte size matches archive member and passes full Pillow image decoding (`img.verify()` and `img.load()`). Halts with `ValueError` on corrupt or wrong-sized files.
>   - Atomic Promotion: Writes new images to temporary `.tmp_*` files, verifies decoding, and promotes atomically using safe no-overwrite linking (`os.link`), immediately cleaning up temporary files on error.
>   - Ground-truth alignment: Generates canonical `dataset/val_manifest.csv` and `dataset/val_manifest.json` mapping each `ILSVRC2012_val_{i:08d}.JPEG` directly to its pre-translated 0-based torchvision class index (`validation_ground_truth.labels_0_indexed[i-1]`). Full JSON metadata includes WNID, devkit_id, and human-readable class name.
>   - Manifest verification: Validates generated manifests directly through `benchmarking.imagenet_manifest.load_imagenet_manifest()`, enforcing sequence continuity and expected count.
>   - CLI extraction runner (`benchmarking/extract_validation_images.py`) supporting `--dry-run`, `--limit`, periodic progress logging (every 5,000 images), safe resume tracking, and non-destructive disk pre-checks.
>   - Comprehensive test suite (`tests/test_image_extraction.py`) with 34 passing tests (346 total passing tests across the entire project).
>   - *Note: In accordance with project instructions, actual extraction of the 50,000 validation images has not been executed during this coding task; the validated CLI tool is ready for user review and execution. Real ImageNet validation accuracy has not yet been measured.*
> - **Step 5 (MySQL Persistence Layer)**:
>   - SQLAlchemy 2.0 ORM models (`database/models.py`) defining `benchmark_runs` and `model_measurements` with strict foreign key constraints, cascading deletes, JSON environment tracking, and SHA-256 artifact hash uniqueness.
>   - Repository CRUD operations (`database/repository.py`) supporting multi-model comparison sessions, transactional session scopes, and query filters.
>   - MySQL engine configuration (`database/config.py`, `database/connection.py`) enforcing `mysql+pymysql` and masking credentials in all string representations. Prohibits SQLite.
>   - Comprehensive database test suite (`tests/test_database.py`, `tests/test_database_config.py`) with 49 passing tests.
> - **Step 6 (FastAPI Backend Services & Recommendation Engine)**:
>   - Modular FastAPI architecture (`backend/`):
>     - `backend/main.py`: Application entrypoint with CORS middleware, lifespan handler, and router aggregation.
>     - `backend/config.py`: Environment-aware API settings (`/api` prefix, CORS origins).
>     - `backend/schemas.py`: Pydantic v2 schemas for all requests, responses, filters, and MCDA parameters.
>     - `backend/dependencies.py`: Database engine and session dependency injection (`get_db`) with automatic commit/rollback.
>     - `backend/routers/health.py`: `GET /api/health` reporting live database connectivity without exposing credentials.
>     - `backend/routers/models.py`: `GET /api/models` returning registered models, runtimes, and exact on-disk artifact sizes.
>     - `backend/routers/results.py`: `GET /api/results` querying saved measurements with pagination and multi-parameter filters (`model_ids`, `runtimes`, `device`, `run_id`).
>     - `backend/routers/benchmarks.py`: `POST /api/benchmarks` (sync/async comparison execution and persistence) and `GET /api/benchmarks/{run_id}`.
>     - `backend/routers/recommendations.py`: `POST /api/recommendations` executing preference-based model ranking.
>   - Multi-Criteria Decision Analysis (MCDA) Engine (`recommendation/engine.py`):
>     - Evaluates trade-offs across Accuracy, Latency, Memory Footprint, and Model Storage.
>     - Standard presets: `balanced`, `speed_optimized`, `edge_constrained`, `accuracy_first`.
>     - Ratio-based normalization preventing artificial zero-anchoring of high-performing edge models.
>     - Hard edge guardrails: `max_latency_ms`, `max_memory_mb`, `max_model_size_mb`, `min_accuracy`.
>     - Transparent handling of missing metrics: never invents values or substitutes zeros.
>     - Generates detailed comparative trade-off and speedup/size ratios in natural language.
>   - Comprehensive API test suite (`tests/test_api.py`) with 17 passing tests (417 passing tests across the entire project).
> - **Step 7 (React Frontend Dashboard)**:
>   - Responsive Single-Page Application (`frontend/`) built with React 19, Vite, Lucide icons, and a custom dark-mode design system with glassmorphism and ambient mesh accents.
>   - Modular UI architecture:
>     - **Overview Dashboard**: System status indicators, 5 KPI cards (Total Models, Saved Measurements, Lowest Latency, Highest Accuracy, Smallest Size).
>     - **Model Matrix**: Interactive cards for all 4 models (*MobileNetV3-Small*, *ResNet50*, *EfficientNet-B0*, *DenseNet121*) with runtime selectors (PyTorch, ONNX, TFLite) and accurate on-disk artifact size badges.
>     - **Benchmark Runner**: Metric configuration, iteration presets (10, 50, 100 runs), verified 50k accuracy reuse by default, and explicit confirmation modal for live sample evaluation.
>     - **Comparison Results**: Side-by-side performance table with strict "N/A" representation for unmeasured metrics and host development CPU baseline notice.
>     - **Visual Charts**: Interactive SVG performance distribution charts for Latency, Accuracy, Memory Footprint, and Storage Size.
>     - **MCDA Recommendations**: Preset selectors (Balanced, Speed-Optimized, Edge-Constrained, Accuracy-First), custom weights, edge constraints, top-candidate showcase card, and full ranking table.
>     - **Run History**: Historical session inspector allowing users to drill down into discrete execution measurements.
>   - Production build verified (`npm run build` in 278ms); component test suite passing (`npm test`, 7 tests in 635ms).

---

## Running the Application

### 1. Prerequisites
- Docker container `edge-mysql` running on `127.0.0.1:3306`.
- Verified `.env` file with MySQL connection credentials.

### 2. Start the FastAPI Backend Server
```bash
# From project root:
PYTHONPATH=. ./.venv311/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Health: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Start the React Frontend Dashboard
```bash
# In a separate terminal (from project root):
npm run dev --prefix frontend -- --port 3000
```
- Web Application: [http://localhost:3000](http://localhost:3000)

### 4. Running Tests
```bash
# Backend & Database Python Tests (417 passing tests)
PYTHONPATH=. ./.venv311/bin/pytest tests/ -v

# Frontend React & Vitest Tests (7 passing tests)
npm test --prefix frontend
```


