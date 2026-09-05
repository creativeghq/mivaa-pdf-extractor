"""Extraction evaluation — score what the pipeline extracted against golden cases.

A golden case (`extraction_eval_cases`) says what a person read on one page for one
product. A run compares that with what the platform holds for the product now
(`flatten_product_fields`: scalar columns + gold `attributes`), writes one
`extraction_eval_runs` row per case, and returns the micro-aggregated metrics.

Endpoints — trusted-service only (x-cron-secret, service-role JWT, the mk_ key):
  POST /api/internal/extraction-eval/run                  score cases → one batch
  GET  /api/internal/extraction-eval/summary/{batch_id}   re-read a batch, aggregated
  GET  /api/internal/extraction-eval/agreement/{case_id}  agreement across the last N runs

The rules the numbers obey live in `app/evaluation/extraction_eval.py` and in
docs/extraction-evaluation.md. In one line each: fixed denominator; a case that
cannot be scored is a row with a failure class, never a missing row; counts are
summed then derived, never averaged per case; agreement is read beside completeness.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.dependencies import require_trusted_service
from app.evaluation.extraction_eval import (
    RUN_FAILURE_CLASSES,
    Metrics,
    agreement,
    classify_run_failure,
    compare,
    flatten_product_fields,
    metrics,
    micro_aggregate,
)
from app.services.core.supabase_client import SupabaseClient, get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal/extraction-eval", tags=["Extraction evaluation"])

PRODUCT_COLUMNS = (
    "id, name, sku, external_sku, description, category, barcode, country_of_origin, "
    "measurement_unit_code, attributes"
)


class RunRequest(BaseModel):
    workspace_id: str
    document_id: Optional[str] = None
    case_ids: Optional[List[str]] = None
    batch_id: Optional[str] = None
    job_id: Optional[str] = None
    #: Defaults to the running MIVAA version — a number nobody can reproduce without it.
    pipeline_version: Optional[str] = None


class RunResponse(BaseModel):
    batch_id: str
    pipeline_version: str
    cases: int
    scored: int
    failure_classes: Dict[str, int]
    aggregate: Dict[str, Any]
    runs: List[Dict[str, Any]]


def _find_product(supabase: SupabaseClient, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The one product the case is about, bound to the case's workspace AND document.

    Two matches is no match: an ambiguous case scores as `no_product`, with the reason
    visible in the run, rather than silently picking the first row.
    """
    match = case.get("product_match") or {}
    q = (
        supabase.client.table("products")
        .select(PRODUCT_COLUMNS)
        .eq("workspace_id", case["workspace_id"])
        .eq("source_document_id", case["document_id"])
    )
    if match.get("product_id"):
        q = q.eq("id", match["product_id"])
    elif match.get("sku"):
        q = q.eq("sku", match["sku"])
    elif match.get("external_sku"):
        q = q.eq("external_sku", match["external_sku"])
    elif match.get("name"):
        q = q.ilike("name", match["name"])  # no wildcard: case-insensitive equality
    else:
        return None
    rows = q.limit(2).execute().data or []
    return rows[0] if len(rows) == 1 else None


def _metrics_row(m: Metrics) -> Dict[str, Any]:
    return {
        "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
        "hallucinated": m.hallucinated,
        "n_expected": m.n_expected, "n_extracted": m.n_extracted, "n_fields": m.n_fields,
        "precision": m.precision, "recall": m.recall, "f1": m.f1,
        "hallucination_rate": m.hallucination_rate, "exact_match_rate": m.exact_match_rate,
    }


def _metrics_from_row(row: Dict[str, Any]) -> Metrics:
    return Metrics(
        tp=int(row.get("tp") or 0), fp=int(row.get("fp") or 0), fn=int(row.get("fn") or 0),
        tn=int(row.get("tn") or 0), hallucinated=int(row.get("hallucinated") or 0),
        n_expected=int(row.get("n_expected") or 0), n_extracted=int(row.get("n_extracted") or 0),
        n_fields=int(row.get("n_fields") or 0),
    )


@router.post("/run", response_model=RunResponse, dependencies=[Depends(require_trusted_service)])
async def run_extraction_eval(
    body: RunRequest,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    batch_id = body.batch_id or str(uuid.uuid4())
    version = body.pipeline_version or get_settings().app_version

    q = (
        supabase.client.table("extraction_eval_cases")
        .select("*")
        .eq("workspace_id", body.workspace_id)
        .eq("is_active", True)
    )
    if body.document_id:
        q = q.eq("document_id", body.document_id)
    if body.case_ids:
        q = q.in_("id", body.case_ids)
    cases = q.order("created_at").execute().data or []
    if not cases:
        raise HTTPException(status_code=404, detail="No active extraction_eval_cases matched")

    failures = {k: 0 for k in RUN_FAILURE_CLASSES}
    per_case: List[Metrics] = []
    runs_out: List[Dict[str, Any]] = []

    for case in cases:
        product = _find_product(supabase, case)
        extracted = flatten_product_fields(product) if product else None
        failure = classify_run_failure(product_found=product is not None, extracted=extracted)
        failures[failure] += 1

        expected = case.get("expected") or {}
        verdicts = compare(expected, extracted, strict=bool(case.get("strict")))
        m = metrics(verdicts)
        per_case.append(m)

        row = {
            "batch_id": batch_id,
            "case_id": case["id"],
            "workspace_id": case["workspace_id"],
            "document_id": case["document_id"],
            "product_id": product["id"] if product else None,
            "job_id": body.job_id,
            "pipeline_version": version,
            "failure_class": failure,
            "extracted": extracted,
            "verdicts": [asdict(v) for v in verdicts],
            **_metrics_row(m),
        }
        ins = supabase.client.table("extraction_eval_runs").insert(row).execute()
        run_id = (ins.data or [{}])[0].get("id")
        runs_out.append({
            "run_id": run_id,
            "case_id": case["id"],
            "case_key": case.get("key"),
            "product_id": row["product_id"],
            "failure_class": failure,
            "metrics": m.as_dict(),
        })

    aggregate = micro_aggregate(per_case)
    return RunResponse(
        batch_id=batch_id,
        pipeline_version=version,
        cases=len(cases),
        scored=failures.get("none", 0),
        failure_classes=failures,
        aggregate=aggregate.as_dict(),
        runs=runs_out,
    )


@router.get("/summary/{batch_id}", dependencies=[Depends(require_trusted_service)])
async def batch_summary(batch_id: str, supabase: SupabaseClient = Depends(get_supabase_client)):
    rows = (
        supabase.client.table("extraction_eval_runs")
        .select("id, case_id, product_id, failure_class, pipeline_version, tp, fp, fn, tn, hallucinated, "
                "n_expected, n_extracted, n_fields, created_at")
        .eq("batch_id", batch_id)
        .order("created_at")
        .execute()
        .data or []
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No runs in that batch")
    per_case = [_metrics_from_row(r) for r in rows]
    failures = {k: 0 for k in RUN_FAILURE_CLASSES}
    for r in rows:
        failures[r.get("failure_class") or "none"] = failures.get(r.get("failure_class") or "none", 0) + 1
    return {
        "batch_id": batch_id,
        "pipeline_versions": sorted({r.get("pipeline_version") for r in rows if r.get("pipeline_version")}),
        "cases": len(rows),
        "scored": failures.get("none", 0),
        "failure_classes": failures,
        "aggregate": micro_aggregate(per_case).as_dict(),
        "runs": [{**{k: r[k] for k in ("id", "case_id", "product_id", "failure_class", "created_at")},
                  "metrics": m.as_dict()} for r, m in zip(rows, per_case)],
    }


@router.get("/agreement/{case_id}", dependencies=[Depends(require_trusted_service)])
async def case_agreement(
    case_id: str,
    limit: int = 5,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Did the same thing come out each time? Across the last `limit` runs of one case.

    Needs no ground truth — which is the point: it is answerable the day a change ships.
    Read `cell_mean` beside `completeness_mean`; a branch that leaves fields empty is
    perfectly stable.
    """
    case_rows = (
        supabase.client.table("extraction_eval_cases").select("id, key, expected").eq("id", case_id).limit(1).execute().data or []
    )
    if not case_rows:
        raise HTTPException(status_code=404, detail="Case not found")
    expected = case_rows[0].get("expected") or {}
    rows = (
        supabase.client.table("extraction_eval_runs")
        .select("id, batch_id, extracted, failure_class, pipeline_version, created_at")
        .eq("case_id", case_id)
        .order("created_at", desc=True)
        .limit(max(1, min(limit, 50)))
        .execute()
        .data or []
    )
    ag = agreement([r.get("extracted") for r in rows], fields=list(expected.keys()))
    return {
        "case_id": case_id,
        "case_key": case_rows[0].get("key"),
        "agreement": ag.as_dict(),
        "runs": [{k: r.get(k) for k in ("id", "batch_id", "failure_class", "pipeline_version", "created_at")} for r in rows],
    }
