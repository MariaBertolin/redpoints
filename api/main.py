from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from api.db import (
    get_listings_by_threshold,
    get_recent_listings,
    init_db,
    save_analysis,
)
from api.pipeline import PipelineService
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    MetadataResponse,
    RecentListingItem,
    RecentListingsResponse,
    SimilarMatch,
    SimilarityResult,
    Stage1Result,
    Stage2Result,
    ThresholdQueryResponse,
    ThresholdQueryResponseItem,
)

app = FastAPI(
    title="Asset Classification API",
    version="1.0.0",
    description="API for asset detection, similarity search, and suspicion scoring.",
)

pipeline_service: PipelineService | None = None


@app.on_event("startup")
def startup_event():
    global pipeline_service
    init_db()
    pipeline_service = PipelineService()


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_listing(payload: AnalyzeRequest):
    if pipeline_service is None:
        raise HTTPException(status_code=500, detail="Pipeline service not initialized")

    result = pipeline_service.analyze(payload.title, payload.top_k)
    listing_id = save_analysis(result)

    return AnalyzeResponse(
        listing_id=listing_id,
        title=result["title"],
        is_asset=result["is_asset"],
        stage1=Stage1Result(**result["stage1"]),
        similarity=SimilarityResult(
            score=result["similarity"]["score"],
            top_k=result["similarity"]["top_k"],
            matches=[SimilarMatch(**m) for m in result["similarity"]["matches"]],
        ),
        stage2=Stage2Result(**result["stage2"]),
        created_at=datetime.fromisoformat(result["created_at"]),
    )


@app.get("/listings/by-threshold", response_model=ThresholdQueryResponse)
def get_by_threshold(
    stage: Literal["stage1", "stage2"] = Query(...),
    threshold: float = Query(..., ge=0.0, le=1.0),
):
    rows = get_listings_by_threshold(stage, threshold)

    items = []
    for row in rows:
        score = row["stage1_score"] if stage == "stage1" else row["stage2_score"]
        items.append(
            ThresholdQueryResponseItem(
                listing_id=row["id"],
                title=row["title"],
                score=score,
                is_asset=bool(row["is_asset"]),
                suspicion_flag=(
                    None if row["suspicion_flag"] is None else bool(row["suspicion_flag"])
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
        )

    return ThresholdQueryResponse(
        stage=stage,
        threshold=threshold,
        count=len(items),
        items=items,
    )


@app.get("/listings/recent", response_model=RecentListingsResponse)
def get_recent(limit: int = Query(default=10, ge=1, le=100)):
    rows = get_recent_listings(limit)

    items = [
        RecentListingItem(
            listing_id=row["id"],
            title=row["title"],
            is_asset=bool(row["is_asset"]),
            stage1_score=row["stage1_score"],
            similarity_score=row["similarity_score"],
            stage2_score=row["stage2_score"],
            suspicion_flag=(
                None if row["suspicion_flag"] is None else bool(row["suspicion_flag"])
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]

    return RecentListingsResponse(
        count=len(items),
        items=items,
    )


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    if pipeline_service is None:
        raise HTTPException(status_code=500, detail="Pipeline service not initialized")
    return MetadataResponse(**pipeline_service.get_metadata())