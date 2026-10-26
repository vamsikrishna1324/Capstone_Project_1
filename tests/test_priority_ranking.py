"""
Regression tests for EdgeBench Priority-Driven Recommendation Ranking.

Tests verify:
1. Accuracy First ranks highest-accuracy eligible model first when verified accuracy is available.
2. Speed Optimized ranks lowest-latency eligible model first.
3. Edge Constrained prioritizes size and memory according to its defined weights.
4. Balanced and Custom Weights follow their respective weights.
5. Missing accuracy cannot win Accuracy First due to score renormalization.
6. Hard constraints exclude ineligible candidates before ranking.
7. Displayed rationales match the effective weights and actual ranking.
"""

import pytest
from recommendation.engine import RecommendationEngine, PRESET_WEIGHTS


def _make_candidate(
    model_id: str,
    runtime: str = "onnx",
    latency: float = 10.0,
    accuracy: float = 0.75,
    memory: float = 200.0,
    model_size: float = 30.0,
):
    """Helper to generate mock candidate measurement dict."""
    return {
        "model_id": model_id,
        "runtime": runtime,
        "device": "cpu",
        "top1_accuracy": accuracy,
        "top5_accuracy": accuracy + 0.15 if accuracy is not None else None,
        "latency_mean_ms": latency,
        "peak_ram_mb": memory,
        "file_size_mb": model_size,
    }


class TestPriorityDrivenRecommendationRanking:
    """Test suite proving priority-driven recommendation behavior."""

    def test_preset_weights_exact_values(self):
        """Verify the exact preset weights documented and used in EdgeBench."""
        assert PRESET_WEIGHTS["balanced"] == {
            "accuracy": 0.35,
            "latency": 0.30,
            "memory": 0.20,
            "model_size": 0.15,
        }
        assert PRESET_WEIGHTS["speed_optimized"] == {
            "accuracy": 0.10,
            "latency": 0.75,
            "memory": 0.075,
            "model_size": 0.075,
        }
        assert PRESET_WEIGHTS["edge_constrained"] == {
            "accuracy": 0.10,
            "latency": 0.20,
            "memory": 0.30,
            "model_size": 0.40,
        }
        assert PRESET_WEIGHTS["accuracy_first"] == {
            "accuracy": 0.75,
            "latency": 0.10,
            "memory": 0.075,
            "model_size": 0.075,
        }

    def test_1_accuracy_first_ranks_highest_accuracy_model_first(self):
        """Accuracy First ranks the highest-accuracy eligible model first when verified accuracy values are available."""
        candidates = [
            _make_candidate("mobilenet_v3_small", latency=2.0, accuracy=0.677, memory=100.0, model_size=9.7),
            _make_candidate("efficientnet_b0", latency=6.5, accuracy=0.771, memory=180.0, model_size=20.5),
            _make_candidate("resnet50", latency=14.0, accuracy=0.809, memory=350.0, model_size=97.8),
        ]

        engine = RecommendationEngine(preset="accuracy_first")
        result = engine.evaluate_candidates(candidates)

        # ResNet50 has highest accuracy (80.9%)
        assert result["recommended_model"] == "resnet50"
        assert result["rankings"][0]["model_id"] == "resnet50"
        assert result["rankings"][0]["rank"] == 1
        assert result["rankings"][1]["model_id"] == "efficientnet_b0"
        assert result["rankings"][2]["model_id"] == "mobilenet_v3_small"

        # Check weights applied
        applied = result["criteria_weights_applied"]
        assert applied["accuracy"] == 0.75
        assert applied["latency"] == 0.10

    def test_2_speed_optimized_ranks_lowest_latency_model_first(self):
        """Speed Optimized ranks the lowest-latency eligible model first."""
        candidates = [
            _make_candidate("mobilenet_v3_small", latency=1.48, accuracy=0.677, memory=120.0, model_size=9.7),
            _make_candidate("efficientnet_b0", latency=6.5, accuracy=0.771, memory=180.0, model_size=20.5),
            _make_candidate("resnet50", latency=13.0, accuracy=0.809, memory=350.0, model_size=97.8),
        ]

        engine = RecommendationEngine(preset="speed_optimized")
        result = engine.evaluate_candidates(candidates)

        # MobileNetV3 has lowest latency (1.48ms)
        assert result["recommended_model"] == "mobilenet_v3_small"
        assert result["rankings"][0]["model_id"] == "mobilenet_v3_small"
        assert result["rankings"][0]["rank"] == 1

        applied = result["criteria_weights_applied"]
        assert applied["latency"] == 0.75
        assert applied["accuracy"] == 0.10

    def test_3_edge_constrained_prioritizes_size_and_memory(self):
        """Edge Constrained prioritizes size (40%) and memory (30%) according to its defined weights."""
        # Candidate Tiny: lowest size (5MB) and memory (60MB), but slower latency (25ms) and lower accuracy (65%)
        # Candidate Fast: faster latency (2ms) and higher accuracy (78%), but large size (80MB) and high RAM (400MB)
        candidates = [
            _make_candidate("tiny_model", latency=25.0, accuracy=0.65, memory=60.0, model_size=5.0),
            _make_candidate("fast_heavy_model", latency=2.0, accuracy=0.78, memory=400.0, model_size=80.0),
        ]

        engine = RecommendationEngine(preset="edge_constrained")
        result = engine.evaluate_candidates(candidates)

        assert result["recommended_model"] == "tiny_model"
        assert result["rankings"][0]["model_id"] == "tiny_model"

        applied = result["criteria_weights_applied"]
        assert applied["model_size"] == 0.40
        assert applied["memory"] == 0.30
        assert applied["latency"] == 0.20
        assert applied["accuracy"] == 0.10

    def test_4_balanced_and_custom_weights(self):
        """Balanced and Custom Weights follow their respective weights."""
        candidates = [
            _make_candidate("mobilenet_v3_small", latency=2.0, accuracy=0.68, memory=100.0, model_size=10.0),
            _make_candidate("efficientnet_b0", latency=6.0, accuracy=0.77, memory=180.0, model_size=20.0),
            _make_candidate("resnet50", latency=14.0, accuracy=0.81, memory=350.0, model_size=98.0),
        ]

        # 1. Balanced: EfficientNet-B0 offers the optimal compromise across accuracy, latency, memory, size
        engine_bal = RecommendationEngine(preset="balanced")
        res_bal = engine_bal.evaluate_candidates(candidates)
        assert res_bal["criteria_weights_applied"] == {
            "accuracy": 0.35,
            "latency": 0.30,
            "memory": 0.20,
            "model_size": 0.15,
        }
        assert res_bal["recommended_model"] == "efficientnet_b0"

        # 2. Custom Weights: set 90% memory, 10% latency
        custom = {"accuracy": 0.0, "latency": 0.10, "memory": 0.90, "model_size": 0.0}
        engine_cust = RecommendationEngine(custom_weights=custom)
        res_cust = engine_cust.evaluate_candidates(candidates)
        assert res_cust["criteria_weights_applied"]["memory"] == 0.90
        assert res_cust["criteria_weights_applied"]["latency"] == 0.10
        assert res_cust["recommended_model"] == "mobilenet_v3_small"

        # 3. Custom Weights with unnormalized inputs: 2.0 and 2.0 -> normalized to 0.5 and 0.5
        engine_norm = RecommendationEngine(custom_weights={"accuracy": 2.0, "latency": 2.0, "memory": 0.0, "model_size": 0.0})
        res_norm = engine_norm.evaluate_candidates(candidates)
        assert pytest.approx(res_norm["criteria_weights_applied"]["accuracy"]) == 0.5
        assert pytest.approx(res_norm["criteria_weights_applied"]["latency"]) == 0.5

        # 4. All-zero custom weights must raise ValueError
        with pytest.raises(ValueError, match="Set at least one criterion weight above zero"):
            RecommendationEngine(custom_weights={"accuracy": 0.0, "latency": 0.0, "memory": 0.0, "model_size": 0.0})

    def test_5_missing_accuracy_cannot_win_accuracy_first_due_to_renormalization(self):
        """Missing accuracy cannot win Accuracy First due to score renormalization."""
        candidates = [
            # Candidate A: Missing accuracy completely, but blazing fast latency, tiny memory & size
            _make_candidate("missing_acc_speedster", latency=1.0, accuracy=None, memory=50.0, model_size=5.0),
            # Candidate B: Verified accuracy, but slower latency and larger size
            _make_candidate("verified_acc_model", latency=12.0, accuracy=0.82, memory=250.0, model_size=60.0),
        ]

        engine = RecommendationEngine(preset="accuracy_first")
        result = engine.evaluate_candidates(candidates)

        # Verified accuracy candidate MUST win under Accuracy First
        assert result["recommended_model"] == "verified_acc_model"
        assert result["rankings"][0]["model_id"] == "verified_acc_model"
        assert result["rankings"][0]["raw_metrics"]["accuracy"] == 0.82

        # The candidate with missing accuracy must rank lower and not receive an inflated composite score
        missing_cand = result["rankings"][1]
        assert missing_cand["model_id"] == "missing_acc_speedster"
        assert missing_cand["raw_metrics"]["accuracy"] is None
        assert missing_cand["subscores"]["accuracy"] is None
        assert missing_cand["composite_score"] <= 25.0  # Cannot get more than 25% because 75% accuracy weight was unearned
        assert any("accuracy" in m for m in missing_cand["missing_metrics"])
        assert "unverified" in missing_cand["trade_offs"]

    def test_5b_accuracy_first_all_missing_accuracy_discloses_limitation(self):
        """If accuracy is unavailable across all candidates in Accuracy First, clearly disclose that limitation."""
        candidates = [
            _make_candidate("model_x", latency=2.0, accuracy=None, memory=100.0, model_size=10.0),
            _make_candidate("model_y", latency=5.0, accuracy=None, memory=200.0, model_size=30.0),
        ]

        engine = RecommendationEngine(preset="accuracy_first")
        result = engine.evaluate_candidates(candidates)

        # Clear disclosure in explanation and trade-offs
        assert "Verified Top-1 accuracy is unavailable across all candidates" in result["explanation"]
        assert "Verified accuracy data was unavailable" in result["trade_off_analysis"]
        for cand in result["rankings"]:
            assert "Accuracy benchmarks unavailable" in cand["trade_offs"]

    def test_6_hard_constraints_exclude_ineligible_candidates_before_ranking(self):
        """Hard constraints exclude ineligible candidates before ranking."""
        candidates = [
            _make_candidate("slow_accurate", latency=25.0, accuracy=0.88, memory=200.0, model_size=40.0),
            _make_candidate("fast_eligible", latency=4.0, accuracy=0.74, memory=150.0, model_size=20.0),
            _make_candidate("huge_ineligible", latency=3.0, accuracy=0.80, memory=800.0, model_size=300.0),
        ]

        constraints = {
            "max_latency_ms": 10.0,   # Disqualifies slow_accurate (25.0ms)
            "max_memory_mb": 500.0,   # Disqualifies huge_ineligible (800MB)
            "max_model_size_mb": 100.0, # Disqualifies huge_ineligible (300MB)
        }

        engine = RecommendationEngine(preset="accuracy_first", constraints=constraints)
        result = engine.evaluate_candidates(candidates)

        # Only fast_eligible satisfies all hard constraints
        assert result["recommended_model"] == "fast_eligible"
        assert len(result["top_recommendations"]) == 1
        assert result["top_recommendations"][0]["model_id"] == "fast_eligible"

        # Ineligible candidates must NOT appear in top_recommendations
        top_ids = [c["model_id"] for c in result["top_recommendations"]]
        assert "slow_accurate" not in top_ids
        assert "huge_ineligible" not in top_ids

        # Ineligible candidates in full rankings must be flagged
        for c in result["rankings"]:
            if c["model_id"] == "slow_accurate":
                assert c["eligible"] is False
                assert any("latency" in r for r in c["disqualification_reasons"])
            elif c["model_id"] == "huge_ineligible":
                assert c["eligible"] is False
                assert len(c["disqualification_reasons"]) >= 1

    def test_7_displayed_rationales_match_effective_weights_and_ranking(self):
        """Displayed rationales match the effective weights and actual ranking."""
        candidates = [
            _make_candidate("fast_model", latency=2.0, accuracy=0.70, memory=100.0, model_size=10.0),
            _make_candidate("accurate_model", latency=12.0, accuracy=0.84, memory=250.0, model_size=50.0),
        ]

        # 1. Accuracy First rationale
        engine_acc = RecommendationEngine(preset="accuracy_first")
        res_acc = engine_acc.evaluate_candidates(candidates)
        winner_acc = res_acc["rankings"][0]
        assert winner_acc["model_id"] == "accurate_model"
        assert "accuracy (75%)" in winner_acc["rationale"]
        assert "Top-ranked match" in winner_acc["rationale"]
        assert "84.0% Top-1 accuracy" in winner_acc["rationale"]
        assert "accuracy (75%)" in res_acc["explanation"]

        # 2. Speed Optimized rationale
        engine_speed = RecommendationEngine(preset="speed_optimized")
        res_speed = engine_speed.evaluate_candidates(candidates)
        winner_speed = res_speed["rankings"][0]
        assert winner_speed["model_id"] == "fast_model"
        assert "latency (75%)" in winner_speed["rationale"]
        assert "2.00ms latency" in winner_speed["rationale"]
        assert "latency (75%)" in res_speed["explanation"]

        # 3. Runner-up trade-offs compare against #1
        runner_up = res_speed["rankings"][1]
        assert runner_up["rank"] == 2
        assert "slower latency than #1" in runner_up["trade_offs"]
        assert "higher Top-1 accuracy vs #1" in runner_up["trade_offs"]

    def test_8_distinct_rationales_per_candidate_not_repeated(self):
        """Each recommendation candidate receives its own distinct rationale and trade-offs."""
        candidates = [
            _make_candidate("mobilenet_v3_small", latency=1.5, accuracy=0.67, memory=100.0, model_size=10.0),
            _make_candidate("efficientnet_b0", latency=6.0, accuracy=0.77, memory=200.0, model_size=20.0),
            _make_candidate("resnet50", latency=12.0, accuracy=0.81, memory=350.0, model_size=98.0),
        ]

        engine = RecommendationEngine(preset="accuracy_first")
        res = engine.evaluate_candidates(candidates)

        top = res["top_recommendations"]
        assert len(top) == 3

        # Every candidate must have a non-empty rationale and trade_offs
        rationales = [c["rationale"] for c in top]
        trade_offs = [c["trade_offs"] for c in top]

        # Verify all rationales are distinct (no two candidates share the exact same rationale)
        assert len(set(rationales)) == 3
        assert len(set(trade_offs)) == 3

        # Candidate 1 is top-ranked
        assert "Top-ranked match" in rationales[0]
        assert "81.0% Top-1 accuracy" in rationales[0]

        # Candidate 2 is rank #2
        assert "Rank #2" in rationales[1]
        assert "77.0% Top-1 accuracy" in rationales[1]

        # Candidate 3 is rank #3
        assert "Rank #3" in rationales[2]
        assert "67.0% Top-1 accuracy" in rationales[2]

    def test_9_preset_precedence_over_conflicting_custom_weights(self):
        """Selected operational preset takes precedence even if conflicting custom_weights are passed."""
        # Custom weights specifying 90% latency
        conflicting = {"accuracy": 0.05, "latency": 0.90, "memory": 0.025, "model_size": 0.025}

        # But preset is accuracy_first (75% accuracy)
        engine = RecommendationEngine(preset="accuracy_first", custom_weights=conflicting)
        applied = engine.weights
        assert applied["accuracy"] == 0.75
        assert applied["latency"] == 0.10
