"""
loop.py — Continuous production-like traffic simulator.

Runs until interrupted (Ctrl-C / SIGTERM).  Each tick picks a
weighted random action, executes it, then sleeps for a log-normally
distributed think time.

Target: ~30 ticks/min  (mean think time ~2 s, log-normal sigma = 0.5)
"""

import logging
import math
import os
import random
import signal
import time

from .actions import ACTION_FNS, pick_action
from .bootstrap import bootstrap, teardown
from .client import PhotoprismClient
from .s3_source import S3ImageSource

logger = logging.getLogger(__name__)

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    logger.info("Signal %s received — finishing current tick then stopping", signum)
    _STOP = True


def _think_time(mean_seconds: float = 2.0, sigma: float = 0.5) -> float:
    """Log-normal think time — mean ~mean_seconds, heavy tail."""
    mu = math.log(mean_seconds) - sigma**2 / 2
    return max(0.1, random.lognormvariate(mu, sigma))


def run_forever(
    photoprism_url: str,
    username: str,
    password: str,
    s3_bucket: str,
    s3_prefix: str,
    mean_think_seconds: float = 2.0,
) -> None:
    global _STOP
    _STOP = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    s3_source = S3ImageSource(bucket=s3_bucket, prefix=s3_prefix)
    client = PhotoprismClient(base_url=photoprism_url, username=username, password=password)

    state = bootstrap(client)
    logger.info(
        "Simulation started: user=%s url=%s bucket=%s prefix=%s",
        username,
        photoprism_url,
        s3_bucket,
        s3_prefix,
    )

    tick = 0
    while not _STOP:
        tick += 1
        action = pick_action()
        logger.info("[tick %d] action=%s", tick, action)

        fn = ACTION_FNS[action]
        try:
            if action == "upload":
                fn(client, state, s3_source)
            else:
                fn(client, state)
        except Exception as exc:
            logger.error("[tick %d] %s raised: %s", tick, action, exc)

        think = _think_time(mean_think_seconds)
        logger.debug("[tick %d] sleeping %.2fs", tick, think)
        time.sleep(think)

    teardown(client)
    logger.info(
        "Simulation stopped after %d ticks — searches=%d uploads=%d likes=%d browses=%d",
        tick,
        state.searches_done,
        state.uploads_done,
        state.likes_done,
        state.browses_done,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Photoprism production-like data generator")
    parser.add_argument("--url", default=os.getenv("PHOTOPRISM_URL", "http://localhost:2342"))
    parser.add_argument("--username", default=os.getenv("PHOTOPRISM_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("PHOTOPRISM_PASSWORD"))
    parser.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", "training-module-proj03"))
    parser.add_argument("--s3-prefix", default=os.getenv("S3_PREFIX"))
    parser.add_argument(
        "--mean-think",
        type=float,
        default=float(os.getenv("MEAN_THINK_SECONDS", "2.0")),
    )
    args = parser.parse_args()

    if not args.password:
        parser.error("PHOTOPRISM_PASSWORD env var or --password is required")
    if not args.s3_prefix:
        parser.error("S3_PREFIX env var or --s3-prefix is required")

    run_forever(
        photoprism_url=args.url,
        username=args.username,
        password=args.password,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        mean_think_seconds=args.mean_think,
    )
