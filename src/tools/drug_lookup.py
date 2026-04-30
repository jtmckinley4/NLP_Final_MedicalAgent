"""OpenFDA drug label lookup — no API key required."""

from __future__ import annotations

import httpx
from pydantic_ai import ModelRetry

from src.models.schemas import DrugInfoResult

_BASE_URL = "https://api.fda.gov/drug/label.json"
_TIMEOUT = 10.0
# Characters to keep from any single label field — raw FDA label text can be enormous
_MAX_FIELD_CHARS = 800


def _truncate(texts: list[str]) -> list[str]:
    """Join and truncate a label field's list of strings to a readable length."""
    joined = " ".join(texts).strip()
    if len(joined) <= _MAX_FIELD_CHARS:
        return [joined]
    return [joined[:_MAX_FIELD_CHARS] + "…"]


def _extract_field(label: dict, *keys: str) -> list[str]:
    """Return the first matching key from a label dict, truncated.

    openFDA label fields are lists of strings (one entry per label section).
    We try each key in order and return the first non-empty match.
    """
    for key in keys:
        value = label.get(key)
        if value and isinstance(value, list):
            return _truncate(value)
    return []

def _normalize_via_rxnorm(name: str) -> str:
    """Try to resolve to a canonical generic name via RxNorm. Falls back to input."""
    try:
        r = httpx.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params={"name": name, "search": 1},
            timeout=5.0,
        )
        if r.status_code == 200:
            rxcui = r.json().get("idGroup", {}).get("rxnormId", [None])[0]
            if rxcui:
                # Get canonical name for that RxCUI
                name_r = httpx.get(
                    f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json",
                    params={"propName": "RxNorm Name"},
                    timeout=5.0,
                )
                # parse and return canonical name...
    except Exception:
        pass
    return name  # fallback: use as-is


def lookup_drug_info_openfda(drug_name: str) -> DrugInfoResult:
    """Retrieve medication label information from the openFDA drug label API.

    Searches by generic name, then brand name, then substance name. Returns
    structured label data including uses, side effects, interactions, and
    warnings. Raises ModelRetry on transient errors so the agent can retry.

    Args:
        drug_name: The medication name to look up (generic or brand name).
    """
    cleaned = _normalize_via_rxnorm(drug_name.strip().lower())
    if not cleaned:
        raise ModelRetry("drug_name cannot be empty.")

    # Try multiple search strategies in order — generic name is most reliable
    search_strategies = [
        f'openfda.generic_name:"{cleaned}"',
        f'openfda.brand_name:"{cleaned}"',
        f'openfda.substance_name:"{cleaned}"',
    ]

    label: dict | None = None
    for search in search_strategies:
        try:
            response = httpx.get(
                _BASE_URL,
                params={"search": search, "limit": 1},
                timeout=_TIMEOUT,
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    label = results[0]
                    break
            elif response.status_code == 404:
                continue  # Try next strategy
            else:
                raise ModelRetry(
                    f"openFDA returned unexpected status {response.status_code}. "
                    "Try again shortly."
                )
        except httpx.TimeoutException:
            raise ModelRetry("openFDA request timed out. Try again.")
        except httpx.NetworkError as e:
            raise ModelRetry(f"Network error contacting openFDA: {e}. Try again.")

    if label is None:
        return DrugInfoResult(
            drug_name=cleaned,
            uses=[],
            common_side_effects=[],
            interactions=[],
            warnings=[],
            disclaimer=(
                f'"{cleaned}" was not found in the openFDA drug label database. '
                "It may be a very new drug, a non-US brand name, or a misspelling. "
                "Please consult a pharmacist for information about this medication."
            ),
        )

    # Prefer the canonical generic name from the label metadata
    openfda_meta = label.get("openfda", {})
    canonical_name = (
        openfda_meta.get("generic_name", [cleaned])[0].lower()
        if openfda_meta.get("generic_name")
        else cleaned
    )

    return DrugInfoResult(
        drug_name=canonical_name,
        uses=_extract_field(label, "indications_and_usage", "purpose"),
        common_side_effects=_extract_field(label, "adverse_reactions", "side_effects"),
        interactions=_extract_field(label, "drug_interactions"),
        warnings=_extract_field(label, "warnings", "warnings_and_cautions", "boxed_warning"),
        disclaimer=(
            "This information is sourced from the openFDA drug label database "
            "and reflects FDA-approved labeling. It is not a substitute for "
            "advice from a licensed pharmacist or clinician."
        ),
    )