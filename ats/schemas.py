"""A single local input type used by the file parser; not included in Kaggle."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CVInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=40, max_length=30000)

    @field_validator("text")
    @classmethod
    def readable_text(cls, value):
        value = value.strip()
        if len(value) < 40 or "\x00" in value:
            raise ValueError("CV needs at least 40 readable characters and no null bytes.")
        return value
