"""Weekly cost-per-tonne-km per route, plus the two baselines the brief defines exactly.

    cost_per_tonne_km = sum(freight_cost) / sum(quantity_tonnes * distance_km)   (per route-week)

    vs own history   = mean of the route's weekly values over the (up to) 8 prior weeks
    vs similar routes = mean weekly value of the *other* routes with the same route_type, same week
"""
import pandas as pd

from .config import CONFIG, Config

REQUIRED_COLUMNS = [
    "shipment_id", "origin", "destination", "route_type", "quantity_tonnes",
    "distance_km", "freight_cost_inr", "shipment_date",
]


def load_shipments(path) -> pd.DataFrame:
    """Read and validate the shipment file, adding route / week_of / tonne_km columns."""
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"shipment file is missing columns: {missing}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("shipment file has missing values in required columns")
    if (df["quantity_tonnes"] <= 0).any() or (df["distance_km"] <= 0).any():
        raise ValueError("quantity_tonnes and distance_km must be positive")
    if df["shipment_id"].duplicated().any():
        raise ValueError("duplicate shipment_id values")

    df = df.copy()
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    # Each route must have exactly one route_type, otherwise grouping is ambiguous.
    if (df.groupby("route")["route_type"].nunique() > 1).any():
        raise ValueError("a route has more than one route_type")
    # Monday of the shipment's Monday-Sunday week.
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.dayofweek, unit="D")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_metrics(shipments: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """One row per (route, week_of) with cost, both baselines and the peer-gap excess."""
    weekly = (
        shipments.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(n_shipments=("shipment_id", "count"),
             total_cost_inr=("freight_cost_inr", "sum"),
             total_tonne_km=("tonne_km", "sum"))
        .sort_values(["route", "week_of"])
        .reset_index(drop=True)
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost_inr"] / weekly["total_tonne_km"]

    # --- vs own history: strictly-prior weeks only (shift(1) = no look-ahead) -----------
    by_route = weekly.groupby("route")["cost_per_tonne_km"]
    prior = by_route.transform(lambda s: s.shift(1))
    weekly["_prior"] = prior
    weekly["own_avg"] = weekly.groupby("route")["_prior"].transform(
        lambda s: s.rolling(cfg.history_weeks, min_periods=1).mean())
    weekly["n_prior_weeks"] = weekly.groupby("route")["_prior"].transform(
        lambda s: s.rolling(cfg.history_weeks, min_periods=1).count()).astype(int)
    weekly["own_pct"] = (weekly["cost_per_tonne_km"] / weekly["own_avg"] - 1) * 100
    weekly = weekly.drop(columns="_prior")

    # --- vs similar routes: other routes, same route_type, same week -------------------
    grp = weekly.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    peer_sum, peer_n = grp.transform("sum"), grp.transform("count")
    weekly["n_peers"] = peer_n - 1
    weekly["peer_avg"] = ((peer_sum - weekly["cost_per_tonne_km"]) / (peer_n - 1)).where(peer_n > 1)
    weekly["peer_pct"] = (weekly["cost_per_tonne_km"] / weekly["peer_avg"] - 1) * 100

    # --- the route's *usual* gap to its peers (prior weeks only) and the excess over it --
    # Some routes are structurally dearer than their peers (e.g. Mumbai-Pune vs Delhi-Jaipur),
    # so a raw "+12% vs peers" is normal for them. The excess removes that standing gap.
    weekly["peer_gap_baseline"] = weekly.groupby("route")["peer_pct"].transform(
        lambda s: s.shift(1).expanding(min_periods=cfg.peer_baseline_min_weeks).median())
    weekly["peer_excess_pp"] = weekly["peer_pct"] - weekly["peer_gap_baseline"]
    return weekly
