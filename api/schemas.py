from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SimilarMatch(BaseModel):
    reference_id: int
    title: str
    score: float


class Stage1Result(BaseModel):
    score: float
    threshold: float


class SimilarityResult(BaseModel):
    score: float
    top_k: int
    matches: List[SimilarMatch]


class Stage2Result(BaseModel):
    executed: bool
    score: Optional[float] = None
    threshold: float
    suspicion_flag: Optional[bool] = None


class AnalyzeResponse(BaseModel):
    listing_id: int
    title: str
    is_asset: bool
    stage1: Stage1Result
    similarity: SimilarityResult
    stage2: Stage2Result
    created_at: datetime


class ThresholdQueryResponseItem(BaseModel):
    listing_id: int
    title: str
    score: float
    is_asset: bool
    suspicion_flag: Optional[bool] = None
    created_at: datetime


class ThresholdQueryResponse(BaseModel):
    stage: Literal["stage1", "stage2"]
    threshold: float
    count: int
    items: List[ThresholdQueryResponseItem]


class RecentListingItem(BaseModel):
    listing_id: int
    title: str
    is_asset: bool
    stage1_score: float
    similarity_score: float
    stage2_score: Optional[float] = None
    suspicion_flag: Optional[bool] = None
    created_at: datetime


class RecentListingsResponse(BaseModel):
    count: int
    items: List[RecentListingItem]


class MetadataResponse(BaseModel):
    model_paths: dict[str, Any]
    datasets: dict[str, Any]
    metrics: dict[str, Any]