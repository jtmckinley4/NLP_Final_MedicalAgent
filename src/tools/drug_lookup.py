"""OpenFDA lookup stub for medication specialist workflow."""

from __future__ import annotations

from pydantic_ai import ModelRetry

from src.models.schemas import DrugInfoResult


def lookup_drug_info_openfda(drug_name: str) -> DrugInfoResult:
    """Return deterministic placeholder drug information.

    TODO: Replace with a real OpenFDA API call once teammate integration lands.
    """
    cleaned = drug_name.strip().lower()
    if not cleaned:
        raise ModelRetry("drug_name cannot be empty.")

    if cleaned in {"ibuprofen", "advil"}:
        return DrugInfoResult(
            drug_name="ibuprofen",
            uses=["Pain relief", "Fever reduction"],
            common_side_effects=["Nausea", "Stomach upset"],
            interactions=["May increase bleeding risk with anticoagulants"],
            disclaimer="Stub data only. Not from live OpenFDA yet.",
        )
    if cleaned in {"acetaminophen", "paracetamol", "tylenol"}:
        return DrugInfoResult(
            drug_name="acetaminophen",
            uses=["Pain relief", "Fever reduction"],
            common_side_effects=["Rare rash", "Nausea"],
            interactions=["Liver toxicity risk with heavy alcohol use"],
            disclaimer="Stub data only. Not from live OpenFDA yet.",
        )

    return DrugInfoResult(
        drug_name=cleaned,
        uses=["No stub profile available"],
        common_side_effects=["Unknown in stub data"],
        interactions=["Unknown in stub data"],
        disclaimer="Stub data only. TODO: query OpenFDA API in teammate integration.",
    )

