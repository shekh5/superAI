"""
Minimal Agentic Tool-Calling Service
-------------------------------------
A FastAPI service that exposes an /agent endpoint and mounts the
reasoning chain documented in README.md under /chain.
"""

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from reasoning_chain.router import router as chain_router
from reasoning_chain.safe_math import evaluate_arithmetic
from reasoning_chain.workspace_config import workspace_settings

APP_VERSION = "1.1.0"

app = FastAPI(title="SuperAI Service", version=APP_VERSION)
app.include_router(chain_router, prefix="/chain")

if workspace_settings.enabled:
    from reasoning_chain.auth import admin_router, bootstrap_admin, require_identity
    from reasoning_chain.auth import router as auth_router

    workspace_settings.validate()
    app.include_router(auth_router)
    app.include_router(admin_router)

    @app.on_event("startup")
    def initialize_workspace():
        bootstrap_admin()

    @app.middleware("http")
    async def protect_workspace(request, call_next):
        public_paths = {
            "/",
            "/health",
            "/version",
            "/chat",
            "/dashboard",
            "/auth/login",
            "/auth/logout",
        }
        if request.url.path not in public_paths:
            require_identity(request)
        return await call_next(request)


# ---------- Tools the agent can call ----------

def tool_calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    return evaluate_arithmetic(expression)


def tool_get_time() -> str:
    """Return the current UTC time."""
    return datetime.now(timezone.utc).isoformat()


def tool_get_weather(city: str) -> str:
    """Gets current weather data. Uses WeatherAPI.com if WEATHER_API_KEY is in env."""
    api_key = os.environ.get("WEATHER_API_KEY")
    if api_key:
        try:
            safe_city = urllib.parse.quote(city)
            url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={safe_city}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                loc_name = data["location"]["name"]
                cond = data["current"]["condition"]["text"]
                temp = data["current"]["temp_c"]
                return f"{loc_name}: {cond}, {temp}°C"
        except Exception as e:
            return f"Error: Weather API call failed: {e}"

    fake_data = {
        "san francisco": "62F, foggy",
        "new york": "75F, sunny",
        "delhi": "34C, hazy",
        "banswara": "33C, partly cloudy",
    }
    return fake_data.get(city.lower(), f"No weather data for '{city}' (mock service)")


TOOLS = {
    "calculator": tool_calculator,
    "get_time": tool_get_time,
    "get_weather": tool_get_weather,
}


# ---------- API models ----------

class AgentRequest(BaseModel):
    tool: str
    argument: Optional[str] = None


class AgentResponse(BaseModel):
    tool: str
    result: str
    latency_ms: float


# ---------- Routes ----------

@app.get("/health")
def health():
    """Used by CD pipeline / container orchestrator for health checks."""
    return {"status": "ok", "time": tool_get_time()}


@app.get("/")
def root():
    return {"message": "SuperAI Service is running", "tools": list(TOOLS.keys())}


@app.get("/version")
def version():
    return {"version": APP_VERSION}



@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    """Serves the interactive Agent chat interface."""
    static_file_path = os.path.join(os.path.dirname(__file__), "static", "chat.html")
    try:
        with open(static_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat UI source file not found")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_ui():
    """Serves the interactive Agent observability dashboard."""
    static_file_path = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")
    try:
        with open(static_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard source file not found")


@app.post("/agent", response_model=AgentResponse)
def run_agent(req: AgentRequest):
    if req.tool not in TOOLS:
        available = list(TOOLS.keys())
        detail = f"Unknown tool '{req.tool}'. Available: {available}"
        raise HTTPException(status_code=400, detail=detail)

    start = time.perf_counter()
    try:
        if req.tool == "get_time":
            result = tool_get_time()
        elif req.tool == "calculator":
            if not req.argument:
                raise HTTPException(status_code=400, detail="calculator requires 'argument'")
            result = tool_calculator(req.argument)
        elif req.tool == "get_weather":
            if not req.argument:
                detail = "get_weather requires 'argument' (city)"
                raise HTTPException(status_code=400, detail=detail)
            result = tool_get_weather(req.argument)
        else:
            raise HTTPException(status_code=400, detail="Tool not implemented")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000
    return AgentResponse(tool=req.tool, result=result, latency_ms=round(latency_ms, 3))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
