import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

from kalshi_api.auth import load_local_env


load_local_env(BASE_DIR)


def env_bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


DEBUG = env_bool(
    "DJANGO_DEBUG",
    not bool(os.environ.get("RAILWAY_ENVIRONMENT_NAME")),
)

_dev_secret_key = "dev-only-kalshi-clone-secret-key"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", _dev_secret_key)
if not DEBUG and SECRET_KEY == _dev_secret_key:
    raise RuntimeError("DJANGO_SECRET_KEY must be configured in production.")

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get(
        "DJANGO_ALLOWED_HOSTS",
        "localhost,127.0.0.1,.railway.app,.railway.internal,healthcheck.railway.app",
    ).split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "kalshi_api.apps.KalshiApiConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
TEMPLATES = []
WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("DATABASE_PATH", str(BASE_DIR / "db.sqlite3")),
        "OPTIONS": {
            "timeout": 20,
        },
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
