"""
webhook.py — Receives Grafana alerts and triggers GitHub Actions retraining workflow.

Grafana contact point points to POST /webhook/retrain.
On receiving a firing alert with action=retrain, calls GitHub Actions API
to dispatch the retrain-redeploy workflow.
"""
import logging
import os

import requests
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()

GITHUB_REPO = os.getenv("GITHUB_REPO", "AryanM27/photoprism-proj03")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
WORKFLOW_FILE = "retrain-redeploy.yml"
GITHUB_REF = "main"


def _trigger_github_workflow(trigger_reason: str = "drift") -> bool:
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set — cannot trigger retraining workflow")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "ref": GITHUB_REF,
        "inputs": {
            "trigger_reason": trigger_reason,
            "semantic_path": "artifacts/semantic/openclip_enhanced_real_v2/training_summary.txt",
            "aesthetic_path": "artifacts/aesthetic/mobilenet_v3_large_fusion_real_v2/training_summary.txt",
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 204:
            logger.info("Successfully triggered retraining workflow via GitHub Actions")
            return True
        else:
            logger.error("GitHub Actions trigger failed: %s %s", resp.status_code, resp.text)
            return False
    except Exception as exc:
        logger.error("Failed to call GitHub Actions API: %s", exc)
        return False


@router.post("/webhook/retrain")
async def retrain_webhook(request: Request):
    """
    Grafana webhook contact point endpoint.
    Fires GitHub Actions retraining workflow when drift alert is received.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    alerts = body.get("alerts", [])
    firing_alerts = [a for a in alerts if a.get("status") == "firing"]

    if not firing_alerts:
        logger.info("Webhook received but no firing alerts — ignoring")
        return {"status": "ignored", "reason": "no firing alerts"}

    # Check if any alert has action=retrain label
    retrain_alerts = [
        a for a in firing_alerts
        if a.get("labels", {}).get("action") == "retrain"
    ]

    if not retrain_alerts:
        logger.info("Firing alerts present but none with action=retrain — ignoring")
        return {"status": "ignored", "reason": "no retrain action label"}

    alert_names = [a.get("labels", {}).get("alertname", "unknown") for a in retrain_alerts]
    logger.warning("Drift/retrain alert(s) fired: %s — triggering retraining workflow", alert_names)

    success = _trigger_github_workflow(trigger_reason="drift")

    if success:
        return {"status": "triggered", "alerts": alert_names}
    else:
        raise HTTPException(status_code=500, detail="Failed to trigger GitHub Actions workflow")
