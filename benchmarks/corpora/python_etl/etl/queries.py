"""Shared SQL used by several jobs."""

ACTIVE_USERS = """
    SELECT u.id, u.email
    FROM raw.users AS u
    JOIN raw.subscriptions AS s ON s.user_id = u.id
    WHERE s.active
"""
