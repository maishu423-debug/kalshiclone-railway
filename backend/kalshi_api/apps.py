from django.apps import AppConfig


class KalshiApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "kalshi_api"

    def ready(self):
        import os

        # RUN_MAIN is set by Django's local autoreloader. Railway explicitly
        # enables the tasks with RUN_BACKGROUND_TASKS. Production must use one
        # Gunicorn worker/replica so that only one copy of each task is started.
        run_tasks = os.environ.get(
            "RUN_BACKGROUND_TASKS",
            os.environ.get("RUN_MAIN", "false"),
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not run_tasks:
            return

        from .forecast import start_background_refresh
        from .price_tracker import start_price_tracking
        from .temp_monitor import start_temp_monitor

        start_background_refresh()
        start_temp_monitor()
        start_price_tracking()
