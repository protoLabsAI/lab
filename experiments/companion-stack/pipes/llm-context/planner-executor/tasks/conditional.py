"""
5 conditional branch tasks.
Each requires the model to make a decision based on a tool result before proceeding.
"""

CONDITIONAL_TASKS = [
    {
        "id": "cd_001",
        "goal": "Check if I have any free time tomorrow between 2pm and 5pm. If I do, schedule a 30-minute focus block. If not, find the next available 30-minute slot this week.",
        "category": "conditional",
        "expected_tools": ["list_events", "create_event"],
        "min_steps": 2,
    },
    {
        "id": "cd_002",
        "goal": "Look up the weather for this weekend. If it's going to rain, draft a message to my hiking group suggesting we reschedule. If it's clear, draft a message confirming we're on.",
        "category": "conditional",
        "expected_tools": ["get_weather", "draft_text"],
        "min_steps": 2,
    },
    {
        "id": "cd_003",
        "goal": "Check whether the task 'submit expense report' is already in my todo list. If it is, mark it complete. If it isn't, add it with a due date of this Friday.",
        "category": "conditional",
        "expected_tools": ["search_tasks", "complete_task", "create_task"],
        "min_steps": 2,
    },
    {
        "id": "cd_004",
        "goal": "Search my email for a reply from Bob about the contract. If there is one, summarise it. If there isn't, draft a follow-up email to Bob asking for an update.",
        "category": "conditional",
        "expected_tools": ["search_email", "draft_text"],
        "min_steps": 2,
    },
    {
        "id": "cd_005",
        "goal": "Check my calendar for any conflicts on Thursday at 3pm. If there's a conflict, propose two alternative times later that day. If there's no conflict, confirm Thursday 3pm works.",
        "category": "conditional",
        "expected_tools": ["list_events"],
        "min_steps": 2,
    },
]
