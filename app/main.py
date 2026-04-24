import asyncio
from contextlib import asynccontextmanager
from contextlib import suppress

from fastapi import FastAPI, HTTPException, Request
from telegram import Update

from app.api import router
from app.config import get_settings
from app.db import Base, engine
from app.services.modeling import get_or_train_model_bundle
from app.services.scheduler import start_scheduler, stop_scheduler
from app.telegram_bot import create_telegram_application, shutdown_telegram_application


settings = get_settings()
telegram_app = create_telegram_application()
background_train_task: asyncio.Task | None = None


async def configure_telegram_webhook() -> None:
    if telegram_app is None:
        return
    base_url = settings.public_base_url
    if not base_url:
        return
    webhook_url = f"{base_url}/telegram/webhook/{settings.telegram_webhook_secret}"
    await telegram_app.bot.set_webhook(webhook_url, allowed_updates=Update.ALL_TYPES)


async def warm_model_in_background() -> None:
    await asyncio.to_thread(
        get_or_train_model_bundle,
        settings.auto_train_model_on_start,
        settings.auto_refresh_data_on_start,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global background_train_task
    Base.metadata.create_all(bind=engine)
    start_scheduler()

    if telegram_app is not None:
        await telegram_app.initialize()
        await telegram_app.start()
        await configure_telegram_webhook()

    if settings.auto_train_model_on_start:
        background_train_task = asyncio.create_task(warm_model_in_background())

    try:
        yield
    finally:
        if background_train_task is not None:
            background_train_task.cancel()
            with suppress(asyncio.CancelledError):
                await background_train_task
        await shutdown_telegram_application(telegram_app)
        stop_scheduler()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)


@app.post("/telegram/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request) -> dict:
    if telegram_app is None:
        raise HTTPException(status_code=503, detail="Telegram bot token is not configured.")
    if secret != settings.telegram_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid webhook secret.")

    payload = await request.json()
    update = Update.de_json(payload, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}
