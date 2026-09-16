"""Studio_server-side wrapper for the kiln_server synthetic-user `/generate`
endpoint. Exports the client and its typed exception hierarchy.
"""

from app.desktop.studio_server.synthetic_user.client import (
    SyntheticUserClient,
    SyntheticUserError,
    SyntheticUserRequestError,
    SyntheticUserServerError,
)

__all__ = [
    "SyntheticUserClient",
    "SyntheticUserError",
    "SyntheticUserRequestError",
    "SyntheticUserServerError",
]
