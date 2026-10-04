from django.urls import path

from . import views


urlpatterns = [
    path("health/", views.health),
    path("miami-temperature",         views.miami_temperature),
    path("miami-temperature/",        views.miami_temperature),
    path("miami-temperature/stream",  views.miami_temperature_stream),
    path("miami-temperature/stream/", views.miami_temperature_stream),
    path("paper/state",               views.paper_state),
    path("paper/state/",              views.paper_state),
    path("paper/order",               views.paper_order),
    path("paper/order/",              views.paper_order),
    path("paper/reset",               views.paper_reset),
    path("paper/reset/",              views.paper_reset),
    # Algorithm
    path("algorithm/state",           views.algorithm_state),
    path("algorithm/state/",          views.algorithm_state),
    path("algorithm/trade",           views.algorithm_trade),
    path("algorithm/trade/",          views.algorithm_trade),
    path("algorithm/refresh",         views.algorithm_refresh),
    path("algorithm/refresh/",        views.algorithm_refresh),
    path("algorithm/cron-refresh",    views.algorithm_cron_refresh),
    path("algorithm/cron-refresh/",   views.algorithm_cron_refresh),
    path("algorithm/refresh-temp",    views.algorithm_refresh_temp),
    path("algorithm/refresh-temp/",   views.algorithm_refresh_temp),
    path("algorithm/forecast-history",  views.forecast_history),
    path("algorithm/forecast-history/", views.forecast_history),
    path("algorithm/trade-history",     views.trade_history),
    path("algorithm/trade-history/",    views.trade_history),
]
