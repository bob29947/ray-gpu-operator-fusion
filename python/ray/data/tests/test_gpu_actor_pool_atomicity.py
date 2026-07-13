"""Characterization of stock ActorPool publication used by GPU stages."""

import os

import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data.context import DataContext


def test_actor_stream_retry_does_not_commit_partial_or_duplicate_output(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
):
    """Actor death reuses streaming indexes instead of publishing a new prefix."""

    @ray.remote(num_cpus=0)
    class AttemptCounter:
        def __init__(self):
            self._attempt = 0

        def next(self):
            self._attempt += 1
            return self._attempt

    class KillAfterFirstOutputOnce:
        def __init__(self, attempt_counter):
            self._attempt_counter = attempt_counter

        def __call__(self, batch):
            yield {"id": [int(batch["id"][0])]}
            if ray.get(self._attempt_counter.next.remote()) == 1:
                os._exit(23)
            yield {"id": [int(batch["id"][1])]}

    attempt_counter = AttemptCounter.remote()
    result = (
        ray.data.from_items([{"id": 0}, {"id": 1}], override_num_blocks=1)
        .map_batches(
            KillAfterFirstOutputOnce,
            compute=ActorPoolStrategy(size=1),
            batch_size=None,
            fn_constructor_args=(attempt_counter,),
        )
        .take_all()
    )

    assert result == [{"id": 0}, {"id": 1}]


def test_actor_stream_max_errored_blocks_drops_partial_task_output(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
):
    """A tolerated streaming-task failure drops the task's complete output."""

    class YieldThenFail:
        def __call__(self, batch):
            yield {"id": [int(batch["id"][0])]}
            raise ValueError("expected failure after output")

    context = DataContext.get_current()
    context.max_errored_blocks = 1
    result = (
        ray.data.from_items([{"id": 0}, {"id": 1}], override_num_blocks=1)
        .map_batches(
            YieldThenFail,
            compute=ActorPoolStrategy(size=1),
            batch_size=None,
        )
        .take_all()
    )

    assert result == []
