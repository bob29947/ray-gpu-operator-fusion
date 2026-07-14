"""Actor-local cuDF, RMM, and KvikIO state shared by GPU transforms."""

from __future__ import annotations

import numbers
import os
from contextlib import suppress
from typing import Any, Optional

_GIB = 1024**3
_KVIKIO_THREADS = 32
_KVIKIO_TASK_SIZE = 16 * 1024**2


def is_positive_integral(value: Any) -> bool:
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and value > 0
    )


def require_config(config: Any, expected: type, kind: str) -> Any:
    if not isinstance(config, expected):
        raise TypeError(f"{kind} requires {expected.__name__}")
    return config


def live_rmm_pool_maximum(free_memory: int) -> int:
    """Reserve 30% and at least 2 GiB outside the managed RMM pool."""

    maximum = min(int(free_memory * 0.70), int(free_memory) - 2 * _GIB)
    return max(0, maximum // 256 * 256)


def refresh_aws_credentials(
    region: str,
    session: Any = None,
    credentials: Any = None,
) -> tuple[Any, Any]:
    """Refresh ambient credentials into the variables consumed by KvikIO."""

    if os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3"):
        raise RuntimeError("Native S3 execution does not support endpoint overrides")
    if session is None:
        import botocore.session

        session = botocore.session.get_session()
    if credentials is None:
        credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("No ambient AWS credentials are available")
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        raise RuntimeError("The ambient AWS credential provider is incomplete")

    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)
    os.environ["AWS_DEFAULT_REGION"] = region
    os.environ["AWS_REGION"] = region
    return session, credentials


def configure_kvikio(region: str, session: Any, credentials: Any) -> None:
    refresh_aws_credentials(region, session, credentials)
    os.environ["KVIKIO_NTHREADS"] = str(_KVIKIO_THREADS)
    os.environ["KVIKIO_TASK_SIZE"] = str(_KVIKIO_TASK_SIZE)

    import kvikio.defaults as defaults
    from kvikio.remote_file import is_remote_file_available

    if not is_remote_file_available():
        raise RuntimeError("KvikIO remote I/O is unavailable")
    defaults.set({"num_threads": _KVIKIO_THREADS, "task_size": _KVIKIO_TASK_SIZE})
    applied = (int(defaults.get("num_threads")), int(defaults.get("task_size")))
    if applied != (_KVIKIO_THREADS, _KVIKIO_TASK_SIZE):
        raise RuntimeError("KvikIO rejected its actor-local configuration")


class CudfRuntimeContext:
    """Mutable state owned by one GPU worker, never by an operator spec."""

    def __init__(self) -> None:
        self.cudf: Any = None
        self.cupy: Any = None
        self._rmm_upstream: Any = None
        self._rmm_pool: Any = None
        self._source_kind: Optional[str] = None
        self._region: Optional[str] = None
        self._aws_session: Any = None
        self._aws_credentials: Any = None
        self._s3_client: Any = None

    def ensure_cudf(
        self,
        *,
        source_kind: Optional[str] = None,
        region: Optional[str] = None,
    ) -> "CudfRuntimeContext":
        if self.cudf is None:
            import rmm

            free_memory, _ = rmm.mr.available_device_memory()
            maximum = live_rmm_pool_maximum(free_memory)
            if maximum <= 0:
                raise RuntimeError(
                    "Parquet/cuDF execution requires more than 2 GiB of free "
                    "GPU memory"
                )
            self._rmm_upstream = rmm.mr.get_current_device_resource()
            self._rmm_pool = rmm.mr.PoolMemoryResource(
                self._rmm_upstream,
                initial_pool_size=min(_GIB, maximum),
                maximum_pool_size=maximum,
            )
            rmm.mr.set_current_device_resource(self._rmm_pool)

            import cupy
            from rmm.allocators.cupy import rmm_cupy_allocator

            cupy.cuda.set_allocator(rmm_cupy_allocator)
            import cudf

            self.cudf = cudf
            self.cupy = cupy

        if source_kind == "s3":
            self._ensure_s3(region)
        elif source_kind not in (None, "local"):
            raise ValueError(f"Unsupported direct cuDF source kind: {source_kind!r}")
        return self

    def _ensure_s3(self, region: Optional[str]) -> None:
        if not isinstance(region, str) or not region:
            raise RuntimeError("Native S3 execution requires a nonempty region")
        if self._source_kind not in (None, "s3"):
            raise RuntimeError("A GPU worker cannot change direct-read source kind")
        if self._region not in (None, region):
            raise RuntimeError("A GPU worker cannot change its direct-read S3 region")
        if self._aws_session is None:
            client = None
            try:
                session, credentials = refresh_aws_credentials(region)
                client = session.create_client("s3", region_name=region)
                configure_kvikio(region, session, credentials)
                self.cudf.set_option("kvikio_remote_io", True)
                if not self.cudf.get_option("kvikio_remote_io"):
                    raise RuntimeError("cuDF rejected KvikIO remote I/O")
            except Exception:
                if client is not None:
                    with suppress(Exception):
                        client.close()
                raise
            self._aws_session = session
            self._aws_credentials = credentials
            self._s3_client = client
        self._source_kind = "s3"
        self._region = region

    @property
    def s3_client(self) -> Any:
        if self._s3_client is None:
            raise RuntimeError("S3 runtime was not initialized")
        return self._s3_client

    def refresh_s3_for_task(self) -> None:
        if self._source_kind != "s3":
            return
        session, credentials = refresh_aws_credentials(
            self._region, self._aws_session, self._aws_credentials
        )
        client = session.create_client("s3", region_name=self._region)
        previous_client = self._s3_client
        self._aws_session = session
        self._aws_credentials = credentials
        self._s3_client = client
        with suppress(Exception):
            previous_client.close()


class CudfTransformRuntime:
    config_type: type
    kind: str

    def __init__(self, config: Any, context: CudfRuntimeContext) -> None:
        self._config = require_config(config, self.config_type, self.kind)
        self._context = context

    def initialize(self) -> None:
        self._context.ensure_cudf()
