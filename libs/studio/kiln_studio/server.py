import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .provider_management import connect_provider_management
from .settings import connect_settings
from .webhost import connect_webhost


def make_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    def ping():
        return "pong"

    connect_provider_management(app)
    connect_settings(app)
    connect_webhost(app)

    return app


if __name__ == "__main__":
    app = make_app()
    uvicorn.run(app, host="127.0.0.1", port=8757)
