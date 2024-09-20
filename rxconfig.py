import reflex as rx


config = rx.Config(
    app_name="root",
    loglevel="debug",
    frontend_port=30000,
    backend_port=38000,
    deploy_url="http://0.0.0.0:30000",
    telemetry_enabled=False,
)
