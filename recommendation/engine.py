"""
Multi-Criteria Decision Analysis (MCDA) Recommendation Engine.

Evaluates benchmarked vision models for edge deployment trade-offs.
Supports custom criteria weighting, preset operational modes, hard edge constraints,
transparent handling of missing metrics (no invented values, no synthetic zeros),
and comprehensive qualitative trade-off explanations.
"""

from typing import Any, Dict, List, Optional, Tuple

from models.model_registry import get_model_metadata, is_model_supported


PRESET_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "accuracy": 0.35,
        "latency": 0.30,
        "memory": 0.20,
        "model_size": 0.15,
    },
    "speed_optimized": {
        "accuracy": 0.10,
        "latency": 0.75,
        "memory": 0.075,
        "model_size": 0.075,
    },
    "edge_constrained": {
        "accuracy": 0.10,
        "latency": 0.20,
        "memory": 0.30,
        "model_size": 0.40,
    },
    "accuracy_first": {
        "accuracy": 0.75,
        "latency": 0.10,
        "memory": 0.075,
        "model_size": 0.075,
    },
}


class RecommendationEngine:
    """Multi-criteria decision analysis (MCDA) recommender for edge models."""

    def __init__(
        self,
        preset: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        """Initialize recommendation engine with weights and constraints.

        Args:
            preset: Optional name of priority preset ('balanced', 'speed_optimized', etc.).
            custom_weights: Optional dictionary of raw weights for accuracy, latency, memory, model_size.
            constraints: Optional hard thresholds (max_latency_ms, max_memory_mb, max_model_size_mb, min_accuracy).
        """
        self.preset = preset
        self.weights = self._resolve_weights(preset, custom_weights)
        self.constraints = constraints or {}
        self._validate_constraints()

    def _resolve_weights(
        self,
        preset: Optional[str],
        custom_weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Normalize criteria weights to sum to 1.0."""
        # Standard operational preset takes precedence when specified
        if preset and preset in PRESET_WEIGHTS and preset != "custom":
            return dict(PRESET_WEIGHTS[preset])

        if custom_weights is not None:
            raw = {}
            for k in ("accuracy", "latency", "memory", "model_size"):
                if k in custom_weights and custom_weights[k] is not None:
                    raw[k] = max(0.0, float(custom_weights[k]))
                else:
                    raw[k] = 0.0
            total = sum(raw.values())
            if total <= 0:
                raise ValueError("Set at least one criterion weight above zero.")
            return {k: round(v / total, 4) for k, v in raw.items()}

        if preset and preset in PRESET_WEIGHTS:
            return dict(PRESET_WEIGHTS[preset])

        return dict(PRESET_WEIGHTS["balanced"])

    def _validate_constraints(self) -> None:
        """Validate numeric bounds of supplied edge constraints."""
        for k in ("max_latency_ms", "max_memory_mb", "max_model_size_mb"):
            val = self.constraints.get(k)
            if val is not None:
                if val <= 0:
                    raise ValueError(f"Constraint {k} must be positive, got {val}.")
        min_acc = self.constraints.get("min_accuracy")
        if min_acc is not None:
            if min_acc < 0.0 or min_acc > 100.0:
                raise ValueError(f"Constraint min_accuracy must be between 0 and 100%, got {min_acc}.")

    def evaluate_candidates(
        self,
        candidate_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Rank and evaluate candidates across multi-criteria metrics.

        Args:
            candidate_data: List of candidate metric dictionaries. Each item contains:
                model_id, runtime, device, and optionally:
                top1_accuracy, latency_mean_ms, peak_ram_mb, file_size_mb, etc.

        Returns:
            Structured recommendation response including rankings, top recommendations,
            recommended model, applied criteria, and qualitative trade-off explanation.
        """
        if not candidate_data:
            return {
                "recommended_model": None,
                "recommended_display_name": None,
                "recommended_runtime": None,
                "composite_score": None,
                "explanation": "No benchmark measurements are available. Please execute a benchmark to evaluate model deployment recommendations.",
                "trade_off_analysis": "Run benchmarks from the dashboard to populate performance metrics for comparative analysis.",
                "criteria_weights_applied": self.weights,
                "constraints_applied": self.constraints,
                "rankings": [],
                "top_recommendations": [],
            }

        # Step 1: Pre-process raw metrics and check hard constraints
        candidates = []
        for item in candidate_data:
            # Analytical estimates must not be normalized or ranked against live Host CPU benchmarks
            if item.get("provenance") == "edge_impulse_estimation" or (item.get("device") and item.get("device") != "cpu"):
                continue

            mid = item.get("model_id", "unknown")
            runtime = item.get("runtime", "unknown")
            device = item.get("device", "cpu")

            display_name = mid
            if is_model_supported(mid):
                display_name = get_model_metadata(mid).display_name

            raw_acc = item.get("top1_accuracy")
            if raw_acc is not None:
                raw_acc = raw_acc if raw_acc <= 1.0 else raw_acc / 100.0

            raw_metrics = {
                "accuracy": raw_acc,
                "latency": item.get("latency_mean_ms"),
                "memory": item.get("peak_ram_mb"),
                "model_size": item.get("file_size_mb"),
            }

            # Check missing metrics transparently
            missing_metrics = [k for k, v in raw_metrics.items() if v is None]
            data_complete = len(missing_metrics) == 0

            # Check hard constraints
            eligible = True
            disqualification_reasons = []
            satisfied_constraints = []

            max_lat = self.constraints.get("max_latency_ms")
            if max_lat is not None:
                if raw_metrics["latency"] is None:
                    eligible = False
                    disqualification_reasons.append(
                        f"Latency is unmeasured; cannot verify constraint of {max_lat:.2f}ms"
                    )
                elif raw_metrics["latency"] > max_lat:
                    eligible = False
                    disqualification_reasons.append(
                        f"Mean latency {raw_metrics['latency']:.2f}ms exceeds constraint of {max_lat:.2f}ms"
                    )
                else:
                    satisfied_constraints.append(
                        f"Latency {raw_metrics['latency']:.2f}ms ≤ {max_lat:.2f}ms"
                    )

            max_mem = self.constraints.get("max_memory_mb")
            if max_mem is not None:
                if raw_metrics["memory"] is None:
                    eligible = False
                    disqualification_reasons.append(
                        f"Memory is unmeasured; cannot verify constraint of {max_mem:.2f}MB"
                    )
                elif raw_metrics["memory"] > max_mem:
                    eligible = False
                    disqualification_reasons.append(
                        f"Peak RAM {raw_metrics['memory']:.2f}MB exceeds constraint of {max_mem:.2f}MB"
                    )
                else:
                    satisfied_constraints.append(
                        f"Peak RAM {raw_metrics['memory']:.1f}MB ≤ {max_mem:.1f}MB"
                    )

            max_size = self.constraints.get("max_model_size_mb")
            if max_size is not None:
                if raw_metrics["model_size"] is None:
                    eligible = False
                    disqualification_reasons.append(
                        f"Model size is unmeasured; cannot verify constraint of {max_size:.2f}MB"
                    )
                elif raw_metrics["model_size"] > max_size:
                    eligible = False
                    disqualification_reasons.append(
                        f"Model size {raw_metrics['model_size']:.2f}MB exceeds constraint of {max_size:.2f}MB"
                    )
                else:
                    satisfied_constraints.append(
                        f"Model size {raw_metrics['model_size']:.2f}MB ≤ {max_size:.2f}MB"
                    )

            min_acc = self.constraints.get("min_accuracy")
            if min_acc is not None:
                threshold = min_acc if min_acc <= 1.0 else min_acc / 100.0
                if raw_metrics["accuracy"] is None:
                    eligible = False
                    disqualification_reasons.append(
                        f"Accuracy is unmeasured; cannot verify constraint of {threshold * 100:.2f}%"
                    )
                elif raw_metrics["accuracy"] < threshold:
                    eligible = False
                    disqualification_reasons.append(
                        f"Top-1 accuracy {raw_metrics['accuracy'] * 100:.2f}% is below minimum required {threshold * 100:.2f}%"
                    )
                else:
                    satisfied_constraints.append(
                        f"Top-1 accuracy {raw_metrics['accuracy'] * 100:.1f}% ≥ {threshold * 100:.1f}%"
                    )

            notes = []
            if missing_metrics:
                notes.append(
                    f"Metrics missing: {', '.join(missing_metrics)}. Score normalized across available dimensions without penalty or synthetic zero."
                )

            candidates.append({
                "model_id": mid,
                "display_name": display_name,
                "runtime": runtime,
                "device": device,
                "raw_metrics": raw_metrics,
                "missing_metrics": missing_metrics,
                "data_complete": data_complete,
                "eligible": eligible,
                "disqualification_reasons": disqualification_reasons,
                "satisfied_constraints": satisfied_constraints,
                "notes": notes,
                "subscores": {},
                "composite_score": 0.0,
                "rationale": None,
                "trade_offs": None,
            })

        # Step 2: Calculate Min-Max bounds across all available metrics
        bounds: Dict[str, Tuple[float, float]] = {}
        for metric in ("accuracy", "latency", "memory", "model_size"):
            vals = [
                c["raw_metrics"][metric]
                for c in candidates
                if c["raw_metrics"][metric] is not None and c["raw_metrics"][metric] >= 0
            ]
            if vals:
                bounds[metric] = (min(vals), max(vals))

        # Step 3: Compute normalized subscores (0 to 100) using MCDA linear scaling
        for c in candidates:
            subscores: Dict[str, Optional[float]] = {}
            for metric in ("accuracy", "latency", "memory", "model_size"):
                val = c["raw_metrics"][metric]
                if val is None or metric not in bounds:
                    subscores[metric] = None
                    continue

                min_v, max_v = bounds[metric]
                if max_v == min_v:
                    score = 100.0
                elif metric == "accuracy":
                    # Benefit metric: higher is better
                    score = 100.0 * (val - min_v) / (max_v - min_v)
                else:
                    # Cost metrics (latency, memory, model size): lower is better
                    score = 100.0 * (max_v - val) / (max_v - min_v)

                subscores[metric] = round(max(0.0, min(100.0, score)), 2)

            c["subscores"] = subscores

        # Step 4: Calculate composite score over available metrics
        is_accuracy_priority = (
            self.preset == "accuracy_first"
            or self.weights.get("accuracy", 0.0) >= 0.50
        )
        any_verified_accuracy = any(
            c["raw_metrics"]["accuracy"] is not None for c in candidates if c["eligible"]
        )

        for c in candidates:
            available_metrics = [
                m for m in ("accuracy", "latency", "memory", "model_size")
                if c["subscores"][m] is not None
            ]
            if available_metrics:
                # If accuracy priority is selected and accuracy is unmeasured while verified candidates exist:
                # Do NOT divide by sum of non-accuracy weights, which would renormalize away the accuracy weight.
                if is_accuracy_priority and c["subscores"]["accuracy"] is None and any_verified_accuracy:
                    composite = sum(self.weights[m] * (c["subscores"][m] or 0.0) for m in available_metrics)
                else:
                    avail_weight = sum(self.weights[m] for m in available_metrics)
                    if avail_weight > 0:
                        composite = sum(self.weights[m] * (c["subscores"][m] or 0.0) for m in available_metrics) / avail_weight
                    else:
                        composite = 0.0
                c["composite_score"] = round(composite, 2)
            else:
                c["composite_score"] = 0.0

        # Step 5: Rank candidates (eligible first, verified accuracy if accuracy_first, then by composite_score desc)
        if is_accuracy_priority and any_verified_accuracy:
            candidates.sort(
                key=lambda x: (
                    1 if x["eligible"] else 0,
                    1 if x["raw_metrics"]["accuracy"] is not None else 0,
                    x["composite_score"],
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda x: (1 if x["eligible"] else 0, x["composite_score"]),
                reverse=True,
            )

        for rank_idx, c in enumerate(candidates, start=1):
            c["rank"] = rank_idx

        # Step 6: Identify recommended winner and top candidates
        eligible_candidates = [c for c in candidates if c["eligible"]]
        recommended = eligible_candidates[0] if eligible_candidates else None

        # Enrich candidates with data-driven rationale and trade-offs
        self._enrich_candidate_explanations(
            candidates, eligible_candidates, is_accuracy_priority, any_verified_accuracy
        )

        top_recommendations = eligible_candidates[:3]

        # Step 7: Generate rich explanation and trade-off analysis
        explanation, trade_off_analysis = self._generate_explanation(
            recommended, candidates, eligible_candidates, is_accuracy_priority, any_verified_accuracy
        )

        return {
            "recommended_model": recommended["model_id"] if recommended else None,
            "recommended_display_name": recommended["display_name"] if recommended else None,
            "recommended_runtime": recommended["runtime"] if recommended else None,
            "composite_score": recommended["composite_score"] if recommended else None,
            "explanation": explanation,
            "trade_off_analysis": trade_off_analysis,
            "criteria_weights_applied": self.weights,
            "constraints_applied": self.constraints,
            "rankings": candidates,
            "top_recommendations": top_recommendations,
        }

    def _enrich_candidate_explanations(
        self,
        all_candidates: List[Dict[str, Any]],
        eligible_candidates: List[Dict[str, Any]],
        is_accuracy_priority: bool = False,
        any_verified_accuracy: bool = False,
    ) -> None:
        """Populate data-driven rationale and trade-offs on each candidate."""
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        active_weights = [f"{k.replace('_', ' ')} ({int(round(v * 100))}%)" for k, v in sorted_weights if v > 0]
        weights_summary = ", ".join(active_weights) or "balanced criteria"

        best_match = eligible_candidates[0] if eligible_candidates else None

        for c in all_candidates:
            if not c["eligible"]:
                reasons = "; ".join(c["disqualification_reasons"])
                c["rationale"] = f"Disqualified from deployment recommendations: {reasons}."
                c["trade_offs"] = "Does not satisfy hard operational edge constraints."
                continue

            rank = c["rank"]
            raw = c["raw_metrics"]
            strengths = []
            if raw.get("latency") is not None:
                strengths.append(f"{raw['latency']:.2f}ms latency")
            if raw.get("accuracy") is not None:
                strengths.append(f"{raw['accuracy'] * 100:.1f}% Top-1 accuracy")
            if raw.get("memory") is not None:
                strengths.append(f"{raw['memory']:.1f}MB RAM")
            if raw.get("model_size") is not None:
                strengths.append(f"{raw['model_size']:.2f}MB size")

            strengths_str = ", ".join(strengths) if strengths else "standard edge profile"

            if rank == 1:
                c["rationale"] = (
                    f"Top-ranked match scoring {c['composite_score']:.1f}/100 aligned with requested priorities "
                    f"({weights_summary}). Delivers {strengths_str}."
                )
                if len(eligible_candidates) > 1:
                    runner_up = eligible_candidates[1]
                    c["trade_offs"] = (
                        f"Outranks runner-up {runner_up['display_name']} [{runner_up['runtime'].upper()}] "
                        f"(score {runner_up['composite_score']:.1f})."
                    )
                else:
                    c["trade_offs"] = "Sole eligible candidate satisfying all specified operational constraints."
            else:
                c["rationale"] = (
                    f"Rank #{rank} eligible option scoring {c['composite_score']:.1f}/100 based on {weights_summary}. "
                    f"Features {strengths_str}."
                )
                comp_notes = []
                if best_match:
                    bm_raw = best_match["raw_metrics"]
                    if raw.get("latency") is not None and bm_raw.get("latency") is not None:
                        diff_lat = raw["latency"] - bm_raw["latency"]
                        if diff_lat > 0:
                            comp_notes.append(f"+{diff_lat:.2f}ms slower latency than #1")
                        elif diff_lat < 0:
                            comp_notes.append(f"{-diff_lat:.2f}ms faster latency than #1")
                    if raw.get("accuracy") is not None and bm_raw.get("accuracy") is not None:
                        diff_acc = (raw["accuracy"] - bm_raw["accuracy"]) * 100
                        if diff_acc < 0:
                            comp_notes.append(f"{diff_acc:.1f}% lower Top-1 accuracy vs #1")
                        elif diff_acc > 0:
                            comp_notes.append(f"+{diff_acc:.1f}% higher Top-1 accuracy vs #1")
                    if raw.get("model_size") is not None and bm_raw.get("model_size") is not None:
                        diff_size = raw["model_size"] - bm_raw["model_size"]
                        if diff_size > 0:
                            comp_notes.append(f"+{diff_size:.2f}MB larger size vs #1")
                        elif diff_size < 0:
                            comp_notes.append(f"{-diff_size:.2f}MB smaller size vs #1")

                c["trade_offs"] = ", ".join(comp_notes) if comp_notes else "Viable alternative with balanced edge trade-offs."

            if c["missing_metrics"]:
                if is_accuracy_priority and any_verified_accuracy and "accuracy" in c["missing_metrics"]:
                    c["trade_offs"] += (
                        " Note: Top-1 accuracy unverified; candidates with verified accuracy are prioritized under Accuracy First."
                    )
                elif is_accuracy_priority and not any_verified_accuracy:
                    c["trade_offs"] += (
                        " Note: Accuracy benchmarks unavailable; scored across secondary verified metrics."
                    )
                else:
                    c["trade_offs"] += (
                        f" Note: {', '.join(c['missing_metrics'])} not measured; scored on verified metrics without penalty."
                    )

    def _generate_explanation(
        self,
        recommended: Optional[Dict[str, Any]],
        all_candidates: List[Dict[str, Any]],
        eligible_candidates: List[Dict[str, Any]],
        is_accuracy_priority: bool = False,
        any_verified_accuracy: bool = False,
    ) -> Tuple[str, str]:
        """Synthesize qualitative rationale and multi-metric trade-off text."""
        if not recommended:
            if not all_candidates:
                explanation = (
                    "No benchmark measurements are available. Please execute a benchmark to evaluate "
                    "model deployment recommendations."
                )
                trade_off = (
                    "Run benchmarks from the dashboard to populate performance metrics for comparative analysis."
                )
                return explanation, trade_off
            explanation = (
                "No model satisfied the specified edge constraints. "
                f"All {len(all_candidates)} candidates were disqualified based on hardware thresholds."
            )
            trade_off = (
                "Consider relaxing latency, memory, or size constraints to find an eligible model."
            )
            return explanation, trade_off

        # Weight priorities text
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        primary_criteria = [f"{k.replace('_', ' ')} ({int(round(v * 100))}%)" for k, v in sorted_weights if v > 0]
        criteria_summary = ", ".join(primary_criteria) or "balanced criteria"

        rec_name = recommended["display_name"]
        rec_rt = recommended["runtime"].upper()
        rec_score = recommended["composite_score"]
        rec_metrics = recommended["raw_metrics"]

        explanation = (
            f"{rec_name} on {rec_rt} is the top recommended deployment configuration, "
            f"scoring {rec_score:.1f}/100 based on {criteria_summary}. "
        )

        # Highlight key strengths
        strengths = []
        if rec_metrics.get("accuracy") is not None:
            strengths.append(f"{rec_metrics['accuracy'] * 100:.2f}% Top-1 accuracy")
        if rec_metrics.get("latency") is not None:
            strengths.append(f"{rec_metrics['latency']:.2f}ms mean latency")
        if rec_metrics.get("model_size") is not None:
            strengths.append(f"{rec_metrics['model_size']:.2f}MB model size")
        if rec_metrics.get("memory") is not None:
            strengths.append(f"{rec_metrics['memory']:.1f}MB peak RAM")

        if strengths:
            explanation += f"It delivers an optimal balance with {', '.join(strengths)}."

        if is_accuracy_priority and not any_verified_accuracy:
            explanation += (
                " (Notice: Verified Top-1 accuracy is unavailable across all candidates; "
                "ranked using secondary operational edge metrics)."
            )

        # Trade-off comparative analysis vs other models
        trade_off_paragraphs = []
        if len(eligible_candidates) > 1:
            runner_up = eligible_candidates[1]
            ru_name = runner_up["display_name"]
            ru_rt = runner_up["runtime"].upper()
            ru_metrics = runner_up["raw_metrics"]

            comp_points = []
            # Latency comparison
            if rec_metrics.get("latency") and ru_metrics.get("latency"):
                diff = ru_metrics["latency"] - rec_metrics["latency"]
                if diff > 0:
                    speedup = ru_metrics["latency"] / rec_metrics["latency"]
                    comp_points.append(
                        f"is {speedup:.1f}x faster ({rec_metrics['latency']:.1f}ms vs {ru_metrics['latency']:.1f}ms)"
                    )
                elif diff < 0:
                    comp_points.append(
                        f"is slower than {ru_name} ({rec_metrics['latency']:.1f}ms vs {ru_metrics['latency']:.1f}ms)"
                    )

            # Model size comparison
            if rec_metrics.get("model_size") and ru_metrics.get("model_size"):
                size_ratio = ru_metrics["model_size"] / rec_metrics["model_size"] if rec_metrics["model_size"] > 0 else 1.0
                if size_ratio > 1.2:
                    comp_points.append(
                        f"requires {size_ratio:.1f}x less storage ({rec_metrics['model_size']:.1f}MB vs {ru_metrics['model_size']:.1f}MB)"
                    )

            # Accuracy comparison
            if rec_metrics.get("accuracy") and ru_metrics.get("accuracy"):
                acc_diff = (ru_metrics["accuracy"] - rec_metrics["accuracy"]) * 100
                if acc_diff > 1.0:
                    comp_points.append(
                        f"trades off {acc_diff:.1f}% Top-1 accuracy against {ru_name}"
                    )
                elif acc_diff < -1.0:
                    comp_points.append(
                        f"achieves {-acc_diff:.1f}% higher Top-1 accuracy than {ru_name}"
                    )

            if comp_points:
                trade_off_paragraphs.append(
                    f"Compared to the runner-up {ru_name} [{ru_rt}] (score {runner_up['composite_score']:.1f}), "
                    f"{rec_name} [{rec_rt}] {', '.join(comp_points)}."
                )

        disqualified_count = len(all_candidates) - len(eligible_candidates)
        if disqualified_count > 0:
            trade_off_paragraphs.append(
                f"{disqualified_count} candidate configuration(s) were excluded from final recommendation "
                "due to hard edge constraint violations."
            )

        incomplete_count = sum(1 for c in all_candidates if not c["data_complete"])
        if incomplete_count > 0:
            if is_accuracy_priority and not any_verified_accuracy:
                trade_off_paragraphs.append(
                    f"Verified accuracy data was unavailable for all {len(all_candidates)} candidates; "
                    "recommendations were scored transparently across secondary operational metrics."
                )
            else:
                trade_off_paragraphs.append(
                    f"{incomplete_count} configuration(s) had unmeasured metrics (such as unevaluated accuracy); "
                    "these were scored transparently across their verified dimensions without inventing data or penalizing with zero."
                )

        trade_off_text = " ".join(trade_off_paragraphs) or "Deployment profile aligns with requested priorities."
        return explanation, trade_off_text
