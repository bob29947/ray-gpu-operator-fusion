"""Ray-compatible exception handling for GPU user-defined functions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Callable, NoReturn, TypeVar

from ray.data.context import DataContext
from ray.data.exceptions import UserCodeException
from ray.util.rpdb import _is_ray_debugger_post_mortem_enabled


T = TypeVar("T")


def raise_user_code_exception(error: Exception) -> NoReturn:
    """Match the stock map UDF exception policy.

    Keep the original exception for debugger/original-exception mode. Otherwise,
    replace the noisy internal stack with Ray Data's public ``UserCodeException``.
    """

    context = DataContext.get_current()
    if _is_ray_debugger_post_mortem_enabled() or context.raise_original_map_exception:
        raise error
    raise UserCodeException("UDF failed to process a data block.") from error


def invoke_user_code(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Invoke user code and apply Ray's public exception policy."""

    try:
        return fn(*args, **kwargs)
    except Exception as error:
        raise_user_code_exception(error)


def iterate_user_code(values: Iterable[T]) -> Iterator[T]:
    """Iterate a lazy user result while wrapping failures raised after the call."""

    try:
        iterator = iter(values)
    except Exception as error:
        raise_user_code_exception(error)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception as error:
            raise_user_code_exception(error)
