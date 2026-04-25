"""
5 error recovery tasks.
Each has a step that will return an error or empty result, testing whether
the planner correctly identifies the failure and adapts.
"""

ERROR_RECOVERY_TASKS = [
    {
        "id": "er_001",
        "goal": "Get the phone number for contact 'Dr. Martinez' and text them that I'll be 10 minutes late.",
        "category": "error_recovery",
        # get_contact will return "contact not found" — model should try search_email or ask user
        "inject_error": {"tool": "get_contact", "error": "Contact 'Dr. Martinez' not found."},
        "expected_tools": ["get_contact", "search_email"],
        "min_steps": 2,
    },
    {
        "id": "er_002",
        "goal": "Check the current stock price of ACME Corp and add it to my notes.",
        "category": "error_recovery",
        # web_search returns rate-limit error on first call
        "inject_error": {"tool": "web_search", "error": "Rate limit exceeded. Retry after 30s."},
        "expected_tools": ["web_search", "create_note"],
        "min_steps": 2,
    },
    {
        "id": "er_003",
        "goal": "Send the Q3 report draft to my manager via email.",
        "category": "error_recovery",
        # search_files returns no results — model should acknowledge and ask for clarification
        "inject_error": {"tool": "search_files", "error": "No file matching 'Q3 report' found."},
        "expected_tools": ["search_files", "send_email"],
        "min_steps": 2,
    },
    {
        "id": "er_004",
        "goal": "Reschedule my 2pm meeting tomorrow to 4pm.",
        "category": "error_recovery",
        # update_event fails — event is in a read-only shared calendar
        "inject_error": {"tool": "update_event", "error": "Permission denied: event is on a shared read-only calendar."},
        "expected_tools": ["list_events", "update_event"],
        "min_steps": 2,
    },
    {
        "id": "er_005",
        "goal": "Look up last month's total expenses from my finance notes and create a summary.",
        "category": "error_recovery",
        # search_notes returns partial/corrupt data
        "inject_error": {"tool": "search_notes", "error": "Partial result: note index corrupted. Only 3 of 12 records returned."},
        "expected_tools": ["search_notes", "draft_text"],
        "min_steps": 2,
    },
]
