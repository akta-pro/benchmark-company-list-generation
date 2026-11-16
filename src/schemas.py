"""Pydantic schemas for OpenRouter structured output, keyed by name so a query
in queries.yaml can select one via `structured_output: <name>`."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ListGenCompany(BaseModel):
    name: str = Field(description="Company name.")
    website: str = Field(
        description="Primary company website URL, including scheme (e.g. https://example.com)."
    )
    ownership_type: Literal["public", "private"] = Field(
        description="Whether the company is publicly traded or privately held."
    )
    hq_city: str = Field(description="City where the company is headquartered.")
    hq_country: str = Field(description="Country where the company is headquartered.")
    product_category: str = Field(
        description="Primary product or service category (e.g. 'Residential & commercial HVAC install/service')."
    )


class ListGen(BaseModel):
    """An object containing a list of companies with core firmographic fields."""

    companyList: list[ListGenCompany] = Field(description="Flat array of companies.")


class RelevanceVerdict(BaseModel):
    """What the relevance judge (src.validate) must return. The JSON schema sent
    to the judge is the query's validation_schema file; this validates the reply."""

    relevant: Literal["relevant", "not_relevant", "info_insufficient"]
    relevant_rationale: str

    @field_validator("relevant_rationale")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject a whitespace-only rationale; keep a valid one exactly as written."""
        if not value.strip():
            raise ValueError("relevant_rationale is blank")
        return value


SCHEMAS: dict[str, type[BaseModel]] = {
    "list_gen": ListGen,
}
