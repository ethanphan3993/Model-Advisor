"""Browse the merged catalog from the local SQLite cache."""

from fastapi import APIRouter, HTTPException, Query

from backend.db import connect
from backend.models.schemas import (
    ArtifactInfo, BenchmarkScore, FacetCount, ModelDetailResponse,
    ModelFacets, ModelListResponse, ModelSummary,
)
from backend.services.recommender import BENCHMARK_SCALE

router = APIRouter()


SORT_OPTIONS = {
    "popular": "score_count DESC, total_params_b DESC",
    "name": "m.display_name ASC",
    "params_asc": "m.total_params_b ASC, m.display_name ASC",
    "params_desc": "m.total_params_b DESC, m.display_name ASC",
    "benchmarks": "score_count DESC, m.display_name ASC",
    "family": "m.family ASC, m.total_params_b ASC",
}


def _row_to_summary(row: dict, artifacts: list[dict], score_count: int) -> ModelSummary:
    return ModelSummary(
        canonical_id=row["canonical_id"],
        family=row["family"],
        parameter_size=row["parameter_size"],
        variant=row["variant"],
        display_name=row["display_name"],
        total_params_b=float(row["total_params_b"] or 0),
        active_params_b=float(row["active_params_b"] or 0),
        is_moe=bool(row["is_moe"]),
        context_length=row["context_length"] or 0,
        tool_calling=bool(row["tool_calling"]),
        vision=bool(row["vision"]),
        license=row["license"] or "",
        artifacts=[ArtifactInfo(
            source=a["source"], source_id=a["source_id"],
            quantization=a["quantization"] or "", size_mb=a["size_mb"] or 0,
            download_url=a["download_url"] or "", install_command=a["install_command"] or "",
        ) for a in artifacts],
        benchmark_count=score_count,
    )


def _build_filters(q: str | None, family: str | None, vision: bool | None,
                   tool_calling: bool | None, is_moe: bool | None,
                   has_benchmarks: bool | None, min_params: float | None,
                   max_params: float | None, source: str | None
                   ) -> tuple[str, list]:
    """Build the WHERE clause + params for both list and facet queries."""
    where: list[str] = ["1=1"]
    params: list = []
    if q:
        like = f"%{q.lower()}%"
        where.append("(LOWER(m.display_name) LIKE ? OR LOWER(m.family) LIKE ?)")
        params.extend([like, like])
    if family:
        where.append("m.family = ?")
        params.append(family)
    if vision is not None:
        where.append("m.vision = ?")
        params.append(1 if vision else 0)
    if tool_calling is not None:
        where.append("m.tool_calling = ?")
        params.append(1 if tool_calling else 0)
    if is_moe is not None:
        where.append("m.is_moe = ?")
        params.append(1 if is_moe else 0)
    if min_params is not None:
        where.append("m.total_params_b >= ?")
        params.append(min_params)
    if max_params is not None:
        where.append("m.total_params_b <= ?")
        params.append(max_params)
    if has_benchmarks:
        where.append("(SELECT COUNT(*) FROM scores s WHERE s.canonical_id = m.canonical_id) > 0")
    if source:
        where.append(
            "EXISTS (SELECT 1 FROM source_artifacts sa WHERE sa.canonical_id = m.canonical_id AND sa.source = ?)"
        )
        params.append(source)
    return " AND ".join(where), params


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    q: str | None = Query(None, description="Search across display name + family"),
    family: str | None = Query(None),
    vision: bool | None = Query(None),
    tool_calling: bool | None = Query(None),
    is_moe: bool | None = Query(None),
    has_benchmarks: bool | None = Query(None),
    min_params: float | None = Query(None, ge=0),
    max_params: float | None = Query(None, ge=0),
    source: str | None = Query(None, description="ollama | lmstudio-community | huggingface_gguf"),
    sort: str = Query("popular", pattern="|".join(SORT_OPTIONS.keys())),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    where_clause, where_params = _build_filters(
        q, family, vision, tool_calling, is_moe, has_benchmarks, min_params, max_params, source,
    )
    order_by = SORT_OPTIONS.get(sort, SORT_OPTIONS["popular"])

    list_sql = f"""
        SELECT m.*,
               (SELECT COUNT(*) FROM scores s WHERE s.canonical_id = m.canonical_id) AS score_count
        FROM models m
        WHERE {where_clause}
        ORDER BY {order_by}
        LIMIT ? OFFSET ?
    """
    count_sql = f"""SELECT COUNT(*) FROM models m WHERE {where_clause}"""

    with connect() as conn:
        total = conn.execute(count_sql, where_params).fetchone()[0]
        rows = conn.execute(list_sql, where_params + [limit, offset]).fetchall()

        out: list[ModelSummary] = []
        for r in rows:
            cid = r["canonical_id"]
            arts = [dict(a) for a in conn.execute(
                "SELECT * FROM source_artifacts WHERE canonical_id = ?", (cid,)).fetchall()]
            out.append(_row_to_summary(dict(r), arts, r["score_count"] or 0))

        # Facets — computed against the same filters as the result set so counts reflect the user's narrowing.
        family_rows = conn.execute(f"""
            SELECT m.family, COUNT(*) c FROM models m WHERE {where_clause}
            GROUP BY m.family ORDER BY c DESC, m.family ASC LIMIT 30
        """, where_params).fetchall()
        source_rows = conn.execute(f"""
            SELECT sa.source, COUNT(DISTINCT sa.canonical_id) c
            FROM source_artifacts sa
            JOIN models m ON m.canonical_id = sa.canonical_id
            WHERE {where_clause}
            GROUP BY sa.source ORDER BY c DESC
        """, where_params).fetchall()
        cap_row = conn.execute(f"""
            SELECT
              SUM(CASE WHEN (SELECT COUNT(*) FROM scores s WHERE s.canonical_id = m.canonical_id) > 0 THEN 1 ELSE 0 END) AS with_bench,
              SUM(m.is_moe) AS moe,
              SUM(m.tool_calling) AS tools,
              SUM(m.vision) AS vis
            FROM models m WHERE {where_clause}
        """, where_params).fetchone()

        facets = ModelFacets(
            families=[FacetCount(value=r["family"], count=r["c"]) for r in family_rows],
            sources=[FacetCount(value=r["source"], count=r["c"]) for r in source_rows],
            total_with_benchmarks=cap_row["with_bench"] or 0,
            total_moe=cap_row["moe"] or 0,
            total_tool_calling=cap_row["tools"] or 0,
            total_vision=cap_row["vis"] or 0,
        )

    return ModelListResponse(models=out, total=total, limit=limit, offset=offset, facets=facets)


@router.get("/models/{canonical_id:path}", response_model=ModelDetailResponse)
async def get_model(canonical_id: str):
    with connect() as conn:
        row = conn.execute("SELECT * FROM models WHERE canonical_id = ?", (canonical_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Model '{canonical_id}' not found")
        arts = [dict(a) for a in conn.execute(
            "SELECT * FROM source_artifacts WHERE canonical_id = ?", (canonical_id,)).fetchall()]
        scores = conn.execute(
            "SELECT benchmark, value, max_value, source, confidence FROM scores WHERE canonical_id = ?",
            (canonical_id,)).fetchall()
        score_models = [
            BenchmarkScore(
                benchmark=s["benchmark"], value=float(s["value"]),
                max_value=float(s["max_value"]),
                normalized=min(1.0, float(s["value"]) / BENCHMARK_SCALE.get(s["benchmark"], 100.0)),
                source=s["source"], confidence=s["confidence"],
            ) for s in scores
        ]
        summary = _row_to_summary(dict(row), arts, len(scores))

    return ModelDetailResponse(**summary.model_dump(), scores=score_models)
