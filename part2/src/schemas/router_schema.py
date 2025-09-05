from typing import List, Union, Any
from pydantic import BaseModel, Field

members = ["tara_support", "tara_troubleshoot", "tara_automate"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class NextSchema(BaseModel):
    title: str = "Next"
    anyOf: List[Union[dict, Any]] = Field(..., description="List of options")

class RouteSchema(BaseModel):
    title: str = "routeSchema"
    type: str = "object"
    properties: dict = Field(..., description="Properties of the route schema")
    required: List[str] = Field(..., description="Required properties")

class SupervisorOut(BaseModel):
    name: str = "route"
    description: str = "Select the next role."
    parameters: RouteSchema

next_schema = NextSchema(anyOf=[{"enum": options}])
route_schema = RouteSchema(properties={"next": next_schema.model_dump()}, required=["next"])
supervisor_schema = SupervisorOut(parameters=route_schema)