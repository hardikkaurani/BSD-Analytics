"""Business analytics, executive KPIs, and fleet rebalancing engine."""

from bsd_analytics.analytics.insights import (
    generate_fleet_recommendations,
    get_executive_kpis,
)

__all__ = [
    "get_executive_kpis",
    "generate_fleet_recommendations",
]
