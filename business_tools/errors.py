"""Shared structured errors for standalone business tools."""

from pydantic import BaseModel, ConfigDict


class ToolError(BaseModel):
    """Machine-readable error returned by a business tool."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
