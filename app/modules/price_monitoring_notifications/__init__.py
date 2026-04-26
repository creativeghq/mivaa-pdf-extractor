"""Price Monitoring Notifications module — credit-metered alert dispatcher."""

from app.modules._core.types import ModuleDefinition
from app.modules.price_monitoring_notifications.manifest import manifest
from app.modules.price_monitoring_notifications.service import (
    PriceAlertDispatcher,
    get_price_alert_dispatcher,
)

definition = ModuleDefinition(
    manifest=manifest,
    router_path=None,  # no public routes — dispatched internally from price_monitoring + tracked_queries
    tags=["price-monitoring-notifications"],
)

__all__ = [
    "PriceAlertDispatcher",
    "definition",
    "get_price_alert_dispatcher",
]
