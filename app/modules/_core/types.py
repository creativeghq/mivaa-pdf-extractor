"""Types shared by the module registry and module manifests."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

PriceTier = Literal["free", "pro", "enterprise"]


class ModuleManifest(BaseModel):
    """Metadata for a module. Must match the frontend manifest of the same slug."""

    slug: str
    name: str
    description: str
    category: str
    price_tier: PriceTier = "free"
    icon: str = ""
    version: str = "0.1.0"


class ModuleDefinition(BaseModel):
    """
    Shape every backend module's `__init__.py` must export as `definition`.
    Router is optional — modules that only expose services don't need one.
    """

    manifest: ModuleManifest
    router_path: Optional[str] = Field(
        default=None,
        description="Dotted import path of the module's FastAPI APIRouter (e.g. 'app.modules.greek_marketplaces.routes.router').",
    )
    tags: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
