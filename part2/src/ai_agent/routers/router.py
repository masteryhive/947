import json

def router(state):
    last_message = state["messages"][-1]
    next_stage = None  # Initialize next_stage to None
    next_stage = json.loads(last_message.content).get("next")

    pass