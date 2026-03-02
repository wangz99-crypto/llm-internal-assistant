# src/ui/demo_page.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

def mount_demo_ui(
    app: FastAPI,
    *,
    demo_mode: bool,
    demo_cookie_name: str,
    demo_admin_key: str,
) -> None:
    """
    Register the demo landing page (GET /) on the given FastAPI app.
    Keeps UI concerns out of src/app.py.
    """

    # Read HTML template from file
    template_path = Path(__file__).resolve().parent / "templates" / "demo.html"
    html = template_path.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        resp = HTMLResponse(html)

        # Set demo cookie (HttpOnly) so browser carries it automatically
        if demo_mode:
            # Render/Reverse-proxy friendly HTTPS detection
            xf_proto = (request.headers.get("x-forwarded-proto") or "").lower()
            is_https = (xf_proto == "https") or (getattr(request.url, "scheme", "") == "https")

            resp.set_cookie(
                key=demo_cookie_name,
                value=demo_admin_key,
                httponly=True,
                samesite="lax",
                secure=is_https,
                max_age=60 * 60 * 24 * 7,
            )
        return resp