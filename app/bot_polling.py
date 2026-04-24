from app.db import Base, engine
from app.telegram_bot import create_telegram_application


def main() -> None:
    Base.metadata.create_all(bind=engine)
    application = create_telegram_application()
    if application is None:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
    application.run_polling()


if __name__ == "__main__":
    main()
