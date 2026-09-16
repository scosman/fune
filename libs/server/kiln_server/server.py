import argparse
import os
from importlib.metadata import version
from typing import Sequence

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from kiln_ai.utils.config import Config

from .custom_errors import connect_custom_errors
from .document_api import connect_document_api
from .feedback_api import connect_feedback_api
from .project_api import connect_project_api
from .prompt_api import connect_prompt_api
from .run_api import connect_run_api
from .spec_api import connect_spec_api
from .statistics_api import connect_statistics_api
from .task_api import connect_task_api
from .utils.agent_checks.policy import ALLOW_AGENT


def _get_version() -> str:
    """Get the version of the kiln-server package."""
    try:
        return version("kiln-server")
    except Exception:
        return "unknown"


tags_metadata = [
    {
        "name": "Projects",
        "description": "Create, read, update, delete, and import projects.",
    },
    {
        "name": "Tasks",
        "description": "Create and manage tasks within projects.",
    },
    {
        "name": "Prompts",
        "description": "Create and manage prompts for tasks.",
    },
    {
        "name": "Specs",
        "description": "Create and manage Kiln Specs: AI guided evaluation generation.",
    },
    {
        "name": "Runs",
        "description": "Execute tasks. View and manage the task run datastore.",
    },
    {
        "name": "Feedback",
        "description": "Create and list feedback on task runs.",
    },
    {
        "name": "Run Configs",
        "description": "Manage run configurations for tasks and evals.",
    },
    {
        "name": "Documents",
        "description": "Manage documents, extraction, chunking, embedding, vector stores, and RAG configurations.",
    },
    {
        "name": "Evals",
        "description": "Create and run evaluations for tasks.",
    },
    {
        "name": "Eval Inputs",
        "description": "Author and manage eval input datasets for tasks.",
    },
    {
        "name": "Statistics",
        "description": "Confidence intervals and significance tests on eval metrics.",
    },
    {
        "name": "Synthetic Data",
        "description": "Generate synthetic data for evals and fine-tuning.",
    },
    {
        "name": "Multiturn SDG",
        "description": "Generate multi-turn synthetic-user conversation datasets for eval.",
    },
    {
        "name": "Eval Builder",
        "description": "Eval builder V2 review pipeline: judge and claim/evidence generation over alignment traces.",
    },
    {
        "name": "Fine-tuning",
        "description": "Create and manage fine-tuning jobs.",
    },
    {
        "name": "Prompt Optimization",
        "description": "Run and monitor prompt optimization jobs.",
    },
    {
        "name": "Skills",
        "description": "Create and manage agent skills within projects.",
    },
    {
        "name": "Copilot",
        "description": "AI copilot for spec generation, refinement, and data generation.",
    },
    {
        "name": "Code Tools",
        "description": "Create, test, and manage user-authored Python code tools.",
    },
    {
        "name": "Tools & MCP",
        "description": "Manage tool servers and MCP connections.",
    },
    {
        "name": "Providers & Models",
        "description": "List and manage AI providers and models.",
    },
    {
        "name": "Git Sync",
        "description": "Git-based synchronization setup, configuration, and management.",
    },
    {
        "name": "Agent",
        "description": "Token-efficient overview endpoints designed for the Kiln chat agent.",
    },
    {
        "name": "Settings & Utilities",
        "description": "Server settings, connectivity checks, and utility endpoints.",
    },
    {
        "name": "Jobs",
        "description": "Run, monitor, and control background jobs, and stream their events.",
    },
]


def make_app(lifespan=None, extra_middleware: list[type] | None = None):
    app = FastAPI(
        title="Kiln AI API",
        summary="A REST API for Kiln AI.",
        description="This API is used to interact with all aspects of Kiln AI. For example, it can create and manage the data model (projects, tasks, prompts, evals, etc). It can also control the execution of the application including running tasks, evals, and more.",
        version=_get_version(),
        lifespan=lifespan,
        openapi_tags=tags_metadata,
    )

    @app.get(
        "/ping",
        summary="Ping Server",
        tags=["Settings & Utilities"],
        openapi_extra=ALLOW_AGENT,
    )
    def ping():
        """Ping the server to check connectivity."""
        return "pong"

    connect_project_api(app)
    connect_task_api(app)
    connect_prompt_api(app)
    connect_spec_api(app)
    connect_run_api(app)
    connect_feedback_api(app)
    connect_document_api(app)
    connect_statistics_api(app)
    connect_custom_errors(app)

    frontend_port = os.environ.get("KILN_FRONTEND_PORT", "5173")
    allowed_origins = [
        f"http://localhost:{frontend_port}",
        f"http://127.0.0.1:{frontend_port}",
        f"https://localhost:{frontend_port}",
        f"https://127.0.0.1:{frontend_port}",
    ]

    # Install any caller-provided middleware BEFORE CORS so it ends up inner to
    # CORS. CORS must remain the outermost middleware: Starlette makes the
    # last-added middleware outermost, and CORS only adds its headers to
    # responses that pass back out through it. Inner middleware that
    # short-circuits a response (e.g. GitSyncMiddleware returning a git-sync
    # error) would otherwise bypass CORS entirely, and the browser would block
    # the response with "origin not allowed".
    for middleware_class in extra_middleware or []:
        app.add_middleware(middleware_class)

    app.add_middleware(
        # Type issue https://github.com/astral-sh/ty/issues/1635
        CORSMiddleware,  # type: ignore[arg-type]
        allow_credentials=True,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Kiln AI REST Server.")
    parser.add_argument("--host", default=None, help="Host for network transports.")
    parser.add_argument(
        "--port", type=int, default=None, help="Port for network transports."
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Log level for the server when using network transports.",
    )
    parser.add_argument(
        "--auto-reload",
        action="store_true",
        help="Enable auto-reload for the server.",
    )
    return parser


app = make_app()


def main(argv: Sequence[str] | None = None) -> None:

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # --host/--port override the default from config.
    # Set as env vars so the config env_var lookup picks them up
    # in reloaded worker processes (in-memory config doesn't survive reload).
    if args.host is not None:
        os.environ["KILN_LOCAL_API_HOST"] = args.host
    if args.port is not None:
        os.environ["KILN_LOCAL_API_PORT"] = str(args.port)

    uvicorn.run(
        "kiln_server.server:app",
        host=Config.shared().kiln_local_api_host,
        port=Config.shared().kiln_local_api_port,
        reload=args.auto_reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
