import asyncio
import contextlib
import logging
import os
import threading
import time
import tkinter as tk
from contextlib import asynccontextmanager
from pathlib import Path

import kiln_ai.datamodel.strict_mode as datamodel_strict_mode
import kiln_server.server as kiln_server
import uvicorn
from fastapi import FastAPI
from kiln_ai.adapters.remote_config import (
    refresh_model_list_background,
    should_skip_remote_model_list,
)
from kiln_ai.utils.config import Config
from kiln_ai.utils.logging import setup_litellm_logging

from app.desktop.git_sync.background_sync import BackgroundSync
from app.desktop.git_sync.config import get_git_sync_config
from app.desktop.git_sync.git_sync_api import connect_git_sync_api
from app.desktop.git_sync.middleware import GitSyncMiddleware
from app.desktop.git_sync.registry import GitSyncRegistry
from app.desktop.log_config import log_config
from app.desktop.studio_server.agent_api import connect_agent_api
from app.desktop.studio_server.batch_plan_api import connect_batch_plan_api
from app.desktop.studio_server.chat import connect_chat_api
from app.desktop.studio_server.code_tool_api import connect_code_tool_api
from app.desktop.studio_server.copilot_api import connect_copilot_api
from app.desktop.studio_server.data_gen_api import connect_data_gen_api
from app.desktop.studio_server.dev_tools import connect_dev_tools
from app.desktop.studio_server.eval_api import connect_evals_api
from app.desktop.studio_server.eval_builder_api import connect_eval_builder_api
from app.desktop.studio_server.finetune_api import connect_fine_tune_api
from app.desktop.studio_server.import_api import connect_import_api
from app.desktop.studio_server.jobs.api import connect_jobs_api
from app.desktop.studio_server.jobs.registry import job_registry
from app.desktop.studio_server.multiturn_sdg_api import connect_multiturn_sdg_api
from app.desktop.studio_server.prompt_api import connect_prompt_api
from app.desktop.studio_server.prompt_optimization_job_api import (
    connect_prompt_optimization_job_api,
)
from app.desktop.studio_server.provider_api import connect_provider_api
from app.desktop.studio_server.repair_api import connect_repair_api
from app.desktop.studio_server.run_config_api import connect_run_config_api
from app.desktop.studio_server.settings_api import connect_settings
from app.desktop.studio_server.skill_api import connect_skill_api
from app.desktop.studio_server.tool_api import connect_tool_servers_api
from app.desktop.studio_server.webhost import connect_webhost

logger = logging.getLogger(__name__)


async def _start_background_syncs() -> None:
    """Start background sync for all auto-sync projects."""
    config = Config.shared()
    raw_projects = config.git_sync_projects
    if not raw_projects:
        return

    for project_path in raw_projects:
        project_config = get_git_sync_config(project_path)
        if project_config is None:
            continue
        if project_config["sync_mode"] != "auto":
            continue
        clone_path = project_config.get("clone_path")
        if clone_path is None:
            continue

        repo_path = Path(clone_path)
        if not repo_path.exists():
            logger.warning(
                "Clone path %s for project %s does not exist, skipping background sync",
                clone_path,
                project_path,
            )
            continue

        manager = GitSyncRegistry.get_or_create(
            repo_path=repo_path,
            remote_name=project_config["remote_name"],
            pat_token=project_config.get("pat_token"),
            oauth_token=project_config.get("oauth_token"),
            auth_mode=project_config["auth_mode"],
        )
        bg_sync = BackgroundSync(manager)
        GitSyncRegistry.register_background_sync(repo_path, bg_sync)
        await bg_sync.start()
        logger.info("Started background sync for project %s", project_path)


async def _stop_background_syncs() -> None:
    """Stop all background syncs and close all managers."""
    for bg_sync in GitSyncRegistry.all_background_syncs():
        await bg_sync.stop()
    for manager in GitSyncRegistry.all_managers():
        await manager.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # debug event loop, warning on hangs
    should_debug = os.environ.get("DEBUG_EVENT_LOOP", "false") == "true"
    if should_debug:
        loop = asyncio.get_event_loop()
        loop.set_debug(True)

    # Set datamodel strict mode on startup
    original_strict_mode = datamodel_strict_mode.strict_mode()
    datamodel_strict_mode.set_strict_mode(True)

    try:
        await _start_background_syncs()
        yield
    finally:
        # End open SSE subscriptions so a UI holding the jobs stream open can't
        # keep the worker alive (e.g. block a dev-server hot reload). Pure
        # observer teardown — jobs keep running. Note uvicorn only reaches
        # lifespan shutdown after its graceful-shutdown wait, so the dev server
        # also sets timeout_graceful_shutdown to bound that wait.
        job_registry.events.shutdown()
        try:
            await _stop_background_syncs()
        finally:
            datamodel_strict_mode.set_strict_mode(original_strict_mode)


def make_app(tk_root: tk.Tk | None = None):
    setup_litellm_logging()

    if not should_skip_remote_model_list():
        refresh_model_list_background()

    # GitSyncMiddleware must be installed INNER to CORS (via extra_middleware)
    # so that its short-circuit error responses still pass back out through
    # CORSMiddleware and receive CORS headers. Adding it here with
    # app.add_middleware() would make it outermost, bypassing CORS and causing
    # the browser to block git-sync error responses ("origin not allowed").
    app = kiln_server.make_app(lifespan=lifespan, extra_middleware=[GitSyncMiddleware])
    connect_provider_api(app)
    connect_prompt_api(app)
    connect_repair_api(app)
    connect_settings(app)
    connect_data_gen_api(app)
    connect_fine_tune_api(app)
    connect_evals_api(app)
    connect_run_config_api(app)
    connect_import_api(app, tk_root=tk_root)
    connect_tool_servers_api(app)
    connect_code_tool_api(app)
    connect_skill_api(app)
    connect_prompt_optimization_job_api(app)
    connect_copilot_api(app)
    connect_eval_builder_api(app)
    connect_multiturn_sdg_api(app)
    connect_batch_plan_api(app)
    connect_git_sync_api(app)
    connect_agent_api(app)
    connect_dev_tools(app)
    connect_chat_api(app)
    connect_jobs_api(app)
    # Important: webhost must be last, it handles all other URLs
    connect_webhost(app)
    return app


def server_config(port: int, host: str, tk_root: tk.Tk | None = None):
    return uvicorn.Config(
        make_app(tk_root=tk_root),
        host=host,
        port=port,
        use_colors=False,
        log_config=log_config(),
    )


class ThreadedServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        self.stopped = False
        thread = threading.Thread(target=self.run_safe, daemon=True)
        thread.start()
        try:
            while not self.started and not self.stopped:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

    def run_safe(self):
        try:
            self.run()
        finally:
            self.stopped = True

    def running(self):
        return self.started and not self.stopped


def run_studio():
    from kiln_ai.utils.config import Config

    uvicorn.run(
        kiln_server.app,
        host=Config.shared().kiln_local_api_host,
        port=Config.shared().kiln_local_api_port,
        log_level="warning",
    )


def run_studio_thread():
    thread = threading.Thread(target=run_studio)
    thread.start()
    return thread
