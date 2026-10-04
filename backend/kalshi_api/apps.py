from django.apps import AppConfig


class KalshiApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "kalshi_api"

    def ready(self):
        import os

        # RUN_MAIN is set by Django's local autoreloader. Railway explicitly
        # enables the tasks with RUN_BACKGROUND_TASKS. Production must use one
        # Gunicorn worker/replica so that only one copy of each task is started.
        explicit = os.environ.get("RUN_BACKGROUND_TASKS")
        if explicit is not None:
            run_tasks = explicit.strip().lower() in {"1", "true", "yes", "on"}
        else:
            # Local convenience: `manage.py runserver` starts the tasks without any env var.
            # With the autoreloader only the child process (RUN_MAIN=true) runs them; with
            # --noreload there is just one process, so it runs them.
            import sys
            run_tasks = False
            if "runserver" in sys.argv:
                run_tasks = os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv
            elif os.environ.get("RUN_MAIN", "").strip().lower() == "true":
                run_tasks = True
        if not run_tasks:
            return

        from .forecast import start_background_refresh
        from .price_tracker import start_price_tracking
        from .temp_monitor import start_temp_monitor

        start_background_refresh()
        start_temp_monitor()
        start_price_tracking()
