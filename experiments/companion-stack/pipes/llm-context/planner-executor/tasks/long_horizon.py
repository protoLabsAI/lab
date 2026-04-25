"""
5 long-horizon tasks.
Each requires 4+ steps and tests plan coherence across a longer execution chain.
"""

LONG_HORIZON_TASKS = [
    {
        "id": "lh_001",
        "goal": "Plan my travel day next Monday: find my first meeting, check the commute time from home, look up parking near the venue, draft a travel itinerary with departure time, and add it to my notes.",
        "category": "long_horizon",
        "expected_tools": ["list_events", "get_directions", "search_web", "draft_text", "create_note"],
        "min_steps": 4,
    },
    {
        "id": "lh_002",
        "goal": "Prepare for my weekly review: list all tasks completed this week, find any emails I haven't replied to, identify my three biggest priorities for next week, and draft a weekly summary note.",
        "category": "long_horizon",
        "expected_tools": ["list_tasks", "search_email", "draft_text", "create_note"],
        "min_steps": 4,
    },
    {
        "id": "lh_003",
        "goal": "Set up a new project called 'Website Redesign': create a project folder in notes, add five initial tasks with due dates spread over the next two weeks, draft a project brief, and send it to the team Slack channel.",
        "category": "long_horizon",
        "expected_tools": ["create_note", "create_task", "draft_text", "send_message"],
        "min_steps": 4,
    },
    {
        "id": "lh_004",
        "goal": "Research and book a team lunch: find a date when all four team members are free next week, search for well-reviewed Italian restaurants within 1km of the office, draft an invitation with the restaurant details, and send it to the team.",
        "category": "long_horizon",
        "expected_tools": ["list_events", "search_web", "draft_text", "send_message"],
        "min_steps": 4,
    },
    {
        "id": "lh_005",
        "goal": "Do a full inbox triage: find all unread emails from the last 3 days, categorise them as action-needed or FYI, draft replies for the action-needed ones, and create a task for each reply that needs sending.",
        "category": "long_horizon",
        "expected_tools": ["search_email", "draft_text", "create_task"],
        "min_steps": 4,
    },
]
