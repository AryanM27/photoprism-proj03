"""
loop.py — Continuous production-like traffic simulator.

Runs until interrupted (Ctrl-C / SIGTERM).  Each tick picks a
weighted random action, executes it, then sleeps for a log-normally
distributed think time.

Target: ~30 ticks/min  (mean think time ~2 s, log-normal sigma = 0.5)

Multiple users are supported via NUM_USERS env var. Each user runs in
its own thread sharing the same Photoprism credentials but with an
independent session and SimState.
"""

import logging
import math
import os
import random
import signal
import time
import threading

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


def _run_user(
    user_index: int,
    photoprism_url: str,
    username: str,
    password: str,
    s3_source: S3ImageSource,
    mean_think_seconds: float,
) -> None:
    """Run a single simulated user loop in its own thread."""
    client = PhotoprismClient(base_url=photoprism_url, username=username, password=password)
    state = bootstrap(client)
    state.user_id = f"datagen_user_{user_index}"
    logger.info("User %d started (session user=%s)", user_index, username)

    tick = 0
    while not _STOP:
        tick += 1
        action = pick_action()
        logger.debug("[user %d tick %d] action=%s", user_index, tick, action)

        fn = ACTION_FNS[action]
        try:
            if action == "upload":
                fn(client, state, s3_source)
            else:
                fn(client, state)
        except Exception as exc:
            logger.error("[user %d tick %d] %s raised: %s", user_index, tick, action, exc)

        think = _think_time(mean_think_seconds)
        time.sleep(think)

    teardown(client)
    logger.info(
        "User %d stopped after %d ticks — searches=%d uploads=%d likes=%d browses=%d",
        user_index, tick,
        state.searches_done, state.uploads_done, state.likes_done, state.browses_done,
    )


def run_forever(
    photoprism_url: str,
    username: str,
    password: str,
    s3_bucket: str,
    s3_prefix: str,
    mean_think_seconds: float = 2.0,
    num_users: int = 1,
) -> None:
    global _STOP
    _STOP = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    s3_source = S3ImageSource(bucket=s3_bucket, prefix=s3_prefix)
    logger.info("Starting %d simulated user(s)", num_users)

    threads = []
    for i in range(1, num_users + 1):
        t = threading.Thread(
            target=_run_user,
            args=(i, photoprism_url, username, password, s3_source, mean_think_seconds),
            daemon=True,
            name=f"datagen-user-{i}",
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


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
    parser.add_argument(
        "--num-users",
        type=int,
        default=int(os.getenv("NUM_USERS", "1")),
        help="Number of concurrent simulated users",
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
        num_users=args.num_users,
    )
