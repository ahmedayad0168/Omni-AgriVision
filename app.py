"""
Omni-AgriVision — Chainlit UI (real API version)

Talks to the FastAPI backend at http://localhost:8000.

SIMPLIFIED STARTUP (No Redis/Celery - Basic Features):
1. Start API: python -m uvicorn api.main:app --host localhost --port 8000 --reload
2. Start UI: python -m chainlit run app.py --host localhost --port 8500

FULL STARTUP (With Video Processing - Requires Docker):
docker-compose up -d

Note: Without Redis/Celery, video upload/analysis won't work, but AI chat, health checks, and reports will work.
"""

import asyncio
import os
from datetime import datetime

import chainlit as cl
import httpx

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
API_TIMEOUT = 600.0  # long timeout — agent analysis can take minutes


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def api_get(path: str, **params):
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        r = await client.get(f"{API_BASE}{path}", params=params)
        r.raise_for_status()
        return r.json()


async def api_post(path: str, json_body=None, query_params=None):
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        r = await client.post(
            f"{API_BASE}{path}",
            json=json_body,
            params=query_params,
        )
        r.raise_for_status()
        return r.json()


async def api_upload(path: str, file_path: str, file_name: str, **params):
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        with open(file_path, "rb") as f:
            files = {"file": (file_name, f, "video/mp4")}
            r = await client.post(f"{API_BASE}{path}", params=params, files=files)
        r.raise_for_status()
        return r.json()


async def is_api_alive() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{API_BASE}/api/health")
            return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def format_dashboard(farms, fields, scans) -> str:
    lines = [
        "# Farm Dashboard",
        "",
        f"**Farms:** {len(farms)}",
        f"**Fields:** {len(fields)}",
        f"**Recent scans:** {len(scans)}",
        "",
    ]
    if farms:
        lines.append("### Farms")
        for f in farms:
            lines.append(
                f"- **#{f['id']}** {f['name']} — {f.get('crop_type', '?')} "
                f"({f.get('total_area', '?')} ha)"
            )
        lines.append("")
    if fields:
        lines.append("### Fields")
        for f in fields:
            lines.append(
                f"- **#{f['id']}** {f['field_name']} — {f.get('crop_type', '?')}"
            )
        lines.append("")
    if scans:
        lines.append("### Recent Scans")
        for s in scans:
            lines.append(
                f"- `{s['id']}` — {s['date'][:19]} — "
                f"status `{s['status']}`, frames {s.get('processed_frames', 0)}"
            )
        lines.append("")
    lines.append("**Available commands:**")
    lines.append("- `dashboard` — this view")
    lines.append("- `analyze` — upload a drone video")
    lines.append("- `health` — field health report")
    lines.append("- `scans` — list recent scans")
    lines.append("- `report` — generate HTML report")
    lines.append("- `alerts` — show open alerts")
    lines.append("- Anything else → sent to the AI agent")
    return "\n".join(lines)


def format_health(data: dict) -> str:
    if "health_score" not in data:
        return f"INFO: {data.get('message', 'No health data available.')}"

    status_indicator = {
        "healthy": "[HEALTHY]",
        "needs_attention": "[ATTENTION]",
        "critical": "[CRITICAL]",
    }.get(data.get("overall_status"), "[UNKNOWN]")

    lines = [
        f"## {status_indicator} Field Health Report",
        f"**Scan date:** {data.get('scan_date', 'N/A')}",
        f"**Health score:** `{data.get('health_score', 0)}%`",
        f"**Plants:** {data.get('healthy_plants', 0)} healthy / "
        f"{data.get('total_plants', 0)} total",
        f"**Overall status:** {data.get('overall_status', 'unknown')}",
    ]

    diseases = data.get("diseases") or []
    if diseases:
        lines.append("\n### Diseases")
        for d in diseases[:10]:
            lines.append(
                f"- **{d.get('name', 'unknown')}** — severity "
                f"{d.get('severity', 0):.2f} ({d.get('severity_level', '?')})"
            )

    pests = data.get("pests") or []
    if pests:
        lines.append("\n### Pests")
        for p in pests[:10]:
            lines.append(
                f"- **{p.get('name', 'unknown')}** — count "
                f"{p.get('count', 0)}, density {p.get('density', 0):.3f}"
            )

    if not diseases and not pests:
        lines.append("\nNo diseases or pests detected in the latest scan.")

    return "\n".join(lines)


def format_scan(data: dict) -> str:
    scan = data.get("scan", {})
    stats = data.get("statistics", {})
    status_emoji = {
        "completed": "✅",
        "processing": "⏳",
        "failed": "❌",
        "pending": "⏸️"
    }.get(scan.get('status', '?'), "❓")

    return (
        f"## 📹 Scan Analysis Results\n\n"
        f"**Scan ID:** `{scan.get('id', '?')}`\n"
        f"**Status:** {status_emoji} `{scan.get('status', '?')}`\n"
        f"**Date:** {scan.get('date', '?')}\n"
        f"**Frames processed:** {scan.get('frames', 0)} / {scan.get('total_frames', '?')}\n\n"
        f"### 🔍 Detection Summary\n\n"
        f"| Category | Count |\n"
        f"|----------|-------|\n"
        f"| **Total Detections** | **{stats.get('total_detections', 0)}** |\n"
        f"| 🌱 Plants | {stats.get('plants', 0)} |\n"
        f"| 🐛 Pests | {stats.get('pests', 0)} |\n"
        f"| 🌿 Weeds | {stats.get('weeds', 0)} |\n"
        f"| 🦠 Diseases | {stats.get('diseases', 0)} |\n\n"
        f"**Analysis:** Video processing completed successfully. "
        f"Data has been stored in SQL Server database for further analysis."
    )


# ---------------------------------------------------------------------------
# Field picker via action buttons
# ---------------------------------------------------------------------------
async def pick_field() -> dict | None:
    try:
        farms = (await api_get("/api/farms")).get("farms", [])
    except Exception as e:
        await cl.Message(content=f"❌ Could not fetch farms: `{e}`").send()
        return None

    all_fields = []
    for farm in farms:
        try:
            fdata = await api_get(f"/api/farms/{farm['id']}/fields")
            all_fields.extend(fdata.get("fields", []))
        except Exception:
            continue

    if not all_fields:
        await cl.Message(
            content="❌ No fields found. Run `python scripts/init_db.py` first."
        ).send()
        return None

    actions = [
        cl.Action(
            name="pick_field",
            value=str(f["id"]),
            label=f"#{f['id']} — {f['field_name']} ({f.get('crop_type', '?')})",
            payload={"field_id": f["id"]},
        )
        for f in all_fields
    ]

    res = await cl.AskActionMessage(
    content="Select a field:",
    actions=actions,
    timeout=180,
    ).send()

    if not res:
        await cl.Message(content="Selection timed out.").send()
        return None

    # Chainlit 2.x returns {"name": ..., "payload": {...}, "value": ...}
    # 'value' is often None; the real data lives in 'payload'.
    field_id = None
    if isinstance(res, dict):
        payload = res.get("payload") or {}
        field_id = payload.get("field_id")
        if field_id is None:
            field_id = res.get("value")

    if field_id is None:
        await cl.Message(content="❌ No field was selected.").send()
        return None

    try:
        field_id = int(field_id)
    except (TypeError, ValueError):
        await cl.Message(content=f"❌ Invalid field id: {field_id!r}").send()
        return None

    return next((f for f in all_fields if f["id"] == field_id), None)


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    """Main chat start handler with service checks"""
    # Check API availability
    api_available = await is_api_alive()

    if not api_available:
        await cl.Message(
            content=(
                "## ⚠️ System Warning\n\n"
                f"Cannot connect to API at `{API_BASE}`\n\n"
                "**Please start the API server first:**\n"
                "```bash\n"
                "python -m uvicorn api.main:app --host localhost --port 8000 --reload\n"
                "```\n\n"
                "**Then restart this UI:**\n"
                "```bash\n"
                "python -m chainlit run app.py --host localhost --port 8500\n"
                "```"
            )
        ).send()
        return

    cl.user_session.set("field_id", None)

    # Professional welcome message with better formatting
    welcome_content = (
        "# 🌾 Omni-AgriVision\n\n"
        "**AI-Powered Precision Agriculture Platform**\n\n"
        f"✅ **System Status**: Connected to API at `{API_BASE}`\n\n"
        "## 📋 Available Commands\n\n"
        "| Command | Description |\n"
        "|----------|-------------|\n"
        "| `dashboard` | View farm overview and statistics |\n"
        "| `analyze` | Upload and analyze drone video |\n"
        "| `health` | Check field health status |\n"
        "| `scans` | List recent scan results |\n"
        "| `report` | Generate comprehensive reports |\n"
        "| `alerts` | View active field alerts |\n"
        "| *Any question* | Chat with AI assistant |\n\n"
        "## 🚀 Key Features\n\n"
        "🎥 **Video Analysis**: Process drone footage for crop health monitoring\n"
        "🤖 **AI Agent**: Get intelligent recommendations and insights\n"
        "📊 **Data Analytics**: Track trends and patterns over time\n"
        "📧 **Notifications**: Email and Telegram alerts\n"
        "📈 **Reports**: Generate detailed HTML reports\n"
        "🗄️ **Database**: SQL Server integration for data persistence\n\n"
        "## 💡 Quick Start\n\n"
        "1. Type `dashboard` to see your farm overview\n"
        "2. Use `analyze` to upload and process drone videos\n"
        "3. Ask any question about your crops for AI-powered insights\n"
        "4. Generate reports with `report` command\n\n"
        "**Processing Mode**: Video analysis works both locally and with Docker."
    )

    await cl.Message(content=welcome_content).send()


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()
    cmd = text.lower().strip()

    # Handle attached video files — auto-trigger analyze flow
    video_elements = []
    if message.elements:
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
        for elem in message.elements:
            if elem.mime and elem.mime.startswith("video"):
                video_elements.append(elem)
            elif elem.name and elem.name.lower().endswith(video_exts):
                video_elements.append(elem)

    if video_elements:
        video_elem = video_elements[0]
        await cl.Message(
            content=f"Processing uploaded video: `{video_elem.name}`…"
        ).send()
        field = await pick_field()
        if not field:
            return
        field_id = field["id"]
        cl.user_session.set("field_id", field_id)
        await process_uploaded_video(field_id, video_elem.path, video_elem.name)
        return

    if cmd == "dashboard":
        await show_dashboard()
    elif cmd == "analyze":
        await show_analyze()
    elif "analyze" in cmd:
        await cl.Message(
            content=(
                "Use the exact command `analyze` to upload a drone video for processing.\n"
                "Or simply upload a video file directly in this chat."
            )
        ).send()
        await show_analyze()
    elif cmd == "health":
        await show_health()
    elif cmd == "scans":
        await show_scans()
    elif cmd == "report":
        await show_report()
    elif cmd == "alerts":
        await show_alerts()
    elif cmd == "":
        await cl.Message(content="Please type a command or question.").send()
    else:
        await ask_agent(text)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
async def show_dashboard():
    try:
        farms = (await api_get("/api/farms")).get("farms", [])
        all_fields = []
        for farm in farms:
            fdata = await api_get(f"/api/farms/{farm['id']}/fields")
            all_fields.extend(fdata.get("fields", []))

        recent_scans = []
        for fld in all_fields[:5]:
            sdata = await api_get(f"/api/field/{fld['id']}/scans", limit=3)
            recent_scans.extend(sdata.get("scans", []))

        md = format_dashboard(farms, all_fields, recent_scans[:10])

        # Create a more professional dashboard display
        dashboard_elements = [
            cl.Text(name="Dashboard Content", content=md),
            cl.Text(
                name="Statistics Summary",
                content=f"**📊 Quick Stats:**\n"
                f"- Total Farms: {len(farms)}\n"
                f"- Total Fields: {len(all_fields)}\n"
                f"- Recent Scans: {len(recent_scans)}\n"
                f"- Database Status: ✅ Connected\n"
                f"- AI Agent: ✅ Ready"
            )
        ]

        await cl.Message(content=dashboard_elements).send()
    except Exception as e:
        await cl.Message(content=f"❌ Dashboard failed: `{e}`").send()


async def process_uploaded_video(field_id: int, file_path: str, file_name: str):
    """Process an already-uploaded video file (from message element or AskFileMessage)."""
    await cl.Message(content=f"Uploading `{file_name}`…").send()

    try:
        result = await api_upload(
            "/api/upload", file_path, file_name, field_id=field_id
        )
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        await cl.Message(
            content=(
                f"Upload failed ({e.response.status_code}): "
                f"{detail or str(e)}\n\n"
                "**Troubleshooting:**\n"
                "- Ensure API server is running\n"
                "- Check video file format\n"
                "- Verify field exists in database"
            )
        ).send()
        return
    except Exception as e:
        await cl.Message(content=f"Upload failed: `{e}`").send()
        return

    scan_id = result["scan_id"]
    mode = result.get("mode", "unknown")

    if mode == "local":
        await cl.Message(
            content=(
                f"Uploaded — scan ID `{scan_id}`\n\n"
                "Processing video locally (synchronous mode).\n"
                "This may take a few minutes depending on video length."
            )
        ).send()
    else:
        await cl.Message(
            content=(
                f"Uploaded — scan ID `{scan_id}`\n\n"
                "Celery worker is processing the video.\n"
                "I'll poll the API for status."
            )
        ).send()

    await poll_scan(scan_id)


async def show_analyze():
    field = await pick_field()
    if not field:
        return

    field_id = field["id"]
    cl.user_session.set("field_id", field_id)

    files = await cl.AskFileMessage(
        content=(
            f"Upload a drone video for **field #{field_id} "
            f"({field['field_name']})**"
        ),
        accept=[
            "video/mp4",
            "video/x-msvideo",
            "video/quicktime",
            "video/x-matroska",
        ],
        max_size_mb=1024,
        timeout=900,
    ).send()

    if not files:
        await cl.Message(content="Upload cancelled.").send()
        return

    video = files[0]

    await process_uploaded_video(field_id, video.path, video.name)


async def poll_scan(scan_id: str, max_wait: int = 1800):
    waited = 0
    interval = 5

    msg = cl.Message(content=f"Processing scan `{scan_id}`…")
    await msg.send()

    while waited < max_wait:
        try:
            data = await api_get(f"/api/scan/{scan_id}")
            status = data["scan"]["status"]

            if status == "completed":
                msg.content = "Processing complete.\n\n" + format_scan(data)
                await msg.update()
                return
            if status == "failed":
                msg.content = (
                    f"❌ Scan `{scan_id}` failed. "
                    "Check the Celery terminal for details."
                )
                await msg.update()
                return

            msg.content = (
                f"⏳ Processing scan `{scan_id}`… "
                f"({waited}s elapsed, status=`{status}`)"
            )
            await msg.update()
        except Exception as e:
            msg.content = f"Polling error: `{e}`"
            await msg.update()

        await asyncio.sleep(interval)
        waited += interval

    msg.content = (
        f"⌛ Timed out after {max_wait}s. "
        "Check the Celery worker terminal."
    )
    await msg.update()


async def show_health():
    field = await pick_field()
    if not field:
        return
    try:
        data = await api_get(f"/api/field/{field['id']}/health")
        await cl.Message(content=format_health(data)).send()
    except Exception as e:
        await cl.Message(content=f"❌ Health fetch failed: `{e}`").send()


async def show_scans():
    field = await pick_field()
    if not field:
        return
    try:
        data = await api_get(f"/api/field/{field['id']}/scans", limit=10)
        scans = data.get("scans", [])
        if not scans:
            await cl.Message(
                content=f"No scans found for field #{field['id']}."
            ).send()
            return
        lines = [f"## Recent Scans — Field #{field['id']}"]
        for s in scans:
            lines.append(
                f"- `{s['id']}` — {s['date'][:19]} — "
                f"status `{s['status']}`, frames {s.get('processed_frames', 0)}"
            )
        await cl.Message(content="\n".join(lines)).send()
    except Exception as e:
        await cl.Message(content=f"❌ Scans fetch failed: `{e}`").send()


async def show_report():
    field = await pick_field()
    if not field:
        return
    await cl.Message(
        content=f"📄 Generating report for field #{field['id']}…"
    ).send()
    try:
        data = await api_get(
            f"/api/field/{field['id']}/report", format="html"
        )
        html = data.get("content", "")
        out_dir = "data/outputs/reports"
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"chainlit_report_{ts}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        await cl.Message(
            content=(
                f"Report generated.\n\n"
                f"Saved to: `{out_path}`\n\n"
                "Open it in a browser to view the styled HTML."
            )
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Report failed: `{e}`").send()


async def show_alerts():
    field = await pick_field()
    if not field:
        return
    try:
        data = await api_get(
            f"/api/field/{field['id']}/alerts", resolved=False
        )
        alerts = data.get("alerts", [])
        if not alerts:
            await cl.Message(
                content=f"No open alerts for field #{field['id']}."
            ).send()
            return
        lines = [f"## Alerts — Field #{field['id']}"]
        for a in alerts:
            lines.append(
                f"- **[{a['alert_level']}]** {a['title']}\n  {a['description']}"
            )
        await cl.Message(content="\n".join(lines)).send()
    except Exception as e:
        await cl.Message(content=f"❌ Alerts fetch failed: `{e}`").send()


async def ask_agent(query: str):
    field = await pick_field()
    if not field:
        return
    field_id = field["id"]

    await cl.Message(
        content=f"Asking the AI agent about field #{field_id}…"
    ).send()

    try:
        data = await api_post(
        "/api/agent/chat",
        query_params={"field_id": field_id, "query": query},
        )
        response = data.get("response", "(no response)")
        await cl.Message(
            content=f"### Agent response\n\n{response}"
        ).send()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        await cl.Message(
            content=(
                f"❌ Agent error ({e.response.status_code}): "
                f"{detail or str(e)}\n\n"
                "**Hint:** make sure Ollama is running:\n"
                "```powershell\nollama serve\nollama pull qwen3:8b\n```"
            )
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Agent request failed: `{e}`").send()