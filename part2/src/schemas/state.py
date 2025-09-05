from typing import Annotated, Sequence, List, Dict, Any, TypedDict
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    stage: Annotated[str, "The current stage or state of the agent"]
    user: Annotated[str, "Unique identifier for the user"]
    messages: Annotated[Sequence[BaseMessage], operator.add] #List of messages related to the agent's interactions
    scratchpad: Annotated[Dict[str, Any], "Scratchpad"]