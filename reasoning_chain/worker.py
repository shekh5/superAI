"""RQ worker entry point for durable document ingestion."""

import os

import redis
from rq import Queue, Worker


def main() -> None:
    connection = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    worker = Worker([Queue("document-ingestion", connection=connection)], connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
