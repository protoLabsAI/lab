"""
5 multi-tool chaining tasks.
Each requires calling 2+ tools in sequence where later steps depend on earlier results.
"""

MULTI_TOOL_TASKS = [
    {
        "id": "mt_001",
        "goal": "Find my next calendar event, check the weather at that location, and draft a two-sentence travel note for me.",
        "category": "multi_tool",
        "expected_tools": ["get_next_event", "get_weather", "draft_text"],
        "min_steps": 3,
    },
    {
        "id": "mt_002",
        "goal": "Look up the last email from Alice, summarise it in one sentence, then add a follow-up task to my todo list.",
        "category": "multi_tool",
        "expected_tools": ["search_email", "draft_text", "create_task"],
        "min_steps": 3,
    },
    {
        "id": "mt_003",
        "goal": "Check if I have any meetings tomorrow afternoon, and if so, find the contact details for all attendees.",
        "category": "multi_tool",
        "expected_tools": ["list_events", "get_contact"],
        "min_steps": 2,
    },
    {
        "id": "mt_004",
        "goal": "Search my notes for anything about the Q2 budget, pull the relevant figure, and send a Slack message to the finance channel with that number.",
        "category": "multi_tool",
        "expected_tools": ["search_notes", "send_message"],
        "min_steps": 2,
    },
    {
        "id": "mt_005",
        "goal": "Find the most recent invoice in my email, extract the total amount, and create a calendar reminder to pay it three days before the due date.",
        "category": "multi_tool",
        "expected_tools": ["search_email", "create_event"],
        "min_steps": 3,
    },
]
