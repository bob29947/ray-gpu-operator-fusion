"""GPU tokenizer and sequence builder used by the fusion benchmark pipeline.

This module deliberately has no Ray, cuDF, or CuPy imports at module import time.
Ray workers import GPU libraries only after a GPU resource has been assigned.
The implementation is self-contained, but follows the optimized integer-modulo
token layout used by the TFM v3 prototype.
"""

from __future__ import annotations

from typing import Final

import numpy as np


PROJECTION: Final[tuple[str, ...]] = (
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant State",
    "Zip",
    "MCC",
)
GROUP_KEYS: Final[tuple[str, str]] = ("User", "Card")
TOKEN_COLUMNS: Final[tuple[str, ...]] = tuple(f"token_{index:02d}" for index in range(12))

MERCHANT_HASH_SIZE: Final[int] = 2_000
SEQUENCE_LENGTH: Final[int] = 4_096
SEQUENCE_CHUNK_SIZE: Final[int] = 315
PAD_TOKEN_ID: Final[int] = 0
BOS_TOKEN_ID: Final[int] = 1
EOS_TOKEN_ID: Final[int] = 2
SEP_TOKEN_ID: Final[int] = 3
UNK_TOKEN_ID: Final[int] = 4

KNOWN_MCCS: Final[tuple[int, ...]] = (
    -1, 1711, 3000, 3001, 3005, 3006, 3007, 3008, 3009, 3058, 3066,
    3075, 3132, 3144, 3174, 3256, 3260, 3359, 3387, 3389, 3390, 3393,
    3395, 3405, 3504, 3509, 3596, 3640, 3684, 3722, 3730, 3771, 3775,
    3780, 4111, 4112, 4121, 4131, 4214, 4411, 4511, 4722, 4784, 4814,
    4829, 4899, 4900, 5045, 5094, 5192, 5193, 5211, 5251, 5261, 5300,
    5310, 5311, 5411, 5499, 5533, 5541, 5621, 5651, 5655, 5661, 5712,
    5719, 5722, 5732, 5733, 5812, 5813, 5814, 5815, 5816, 5912, 5921,
    5932, 5941, 5942, 5947, 5970, 5977, 6300, 7011, 7210, 7230, 7276,
    7349, 7393, 7531, 7538, 7542, 7549, 7801, 7802, 7832, 7922, 7995,
    7996, 8011, 8021, 8041, 8043, 8049, 8062, 8099, 8111, 8931, 9402,
)
INDUSTRY_RANGES: Final[tuple[tuple[int, int, str], ...]] = (
    (0, 1499, "AGRICULTURAL"),
    (1500, 2999, "CONTRACTED"),
    (3000, 3299, "AIRLINES"),
    (3300, 3499, "CAR_RENTAL"),
    (3500, 3999, "LODGING"),
    (4000, 4799, "TRANSPORTATION"),
    (4800, 4999, "UTILITIES"),
    (5000, 5599, "RETAIL"),
    (5600, 5699, "CLOTHING"),
    (5700, 7299, "MISC_STORES"),
    (7300, 7999, "BUSINESS"),
    (8000, 8999, "PROFESSIONAL"),
    (9000, 9999, "GOVERNMENT"),
)
CHIP_MAPPING: Final[dict[str, str]] = {
    "SWIPE TRANSACTION": "SWIPE",
    "CHIP TRANSACTION": "CHIP",
    "ONLINE TRANSACTION": "ONLINE",
}
ALL_STATES: Final[tuple[str, ...]] = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP", "XX", "ONLINE",
)


def integer_mod_merchant_bucket(values: np.ndarray, size: int = MERCHANT_HASH_SIZE) -> np.ndarray:
    """Pure reference for the data-independent merchant mapping."""
    if size < 1:
        raise ValueError("merchant hash size must be positive")
    values = np.asarray(values, dtype=np.int64)
    return np.abs(values) % size


def _vocab_layout(merchant_hash_size: int) -> dict[str, int]:
    cat_labels = sorted({label for _, _, label in INDUSTRY_RANGES}) + ["GENERAL"]
    mcc_labels = sorted({str(value) for value in KNOWN_MCCS}) + ["-1"]
    chip_labels = sorted(set(CHIP_MAPPING.values())) + ["UNK"]
    state_labels = sorted(set(ALL_STATES)) + ["XX"]
    sizes = {
        "amt": 7,
        "merch": merchant_hash_size,
        "cat": len(dict.fromkeys(cat_labels)),
        "mcc": len(dict.fromkeys(mcc_labels)),
        "hour": 24,
        "dow": 7,
        "month": 12,
        "card": 10,
        "chip": len(dict.fromkeys(chip_labels)),
        "zip3": 1_000,
        "state": len(dict.fromkeys(state_labels)),
        "cust": 3_000,
    }
    offset = UNK_TOKEN_ID + 1
    offsets: dict[str, int] = {}
    for name, size in sizes.items():
        offsets[name] = offset
        offset += size
    offsets["vocab_size"] = offset
    return offsets


class OptimalCudfTokenizer:
    """Row-preserving, data-independent, GPU transaction tokenizer."""

    def __init__(self, merchant_hash_size: int = MERCHANT_HASH_SIZE):
        import cudf
        import cupy as cp

        self.cudf = cudf
        self.cp = cp
        self.merchant_hash_size = int(merchant_hash_size)
        if self.merchant_hash_size < 1:
            raise ValueError("merchant_hash_size must be positive")
        self.offset = _vocab_layout(self.merchant_hash_size)
        if self.offset["vocab_size"] > np.iinfo(np.uint16).max:
            raise ValueError("token vocabulary does not fit in uint16")

        cat_labels = sorted({label for _, _, label in INDUSTRY_RANGES}) + ["GENERAL"]
        cat_index = {label: index for index, label in enumerate(dict.fromkeys(cat_labels))}
        self.cat_default = cat_index["GENERAL"]
        cat_lookup = np.full(10_000, self.cat_default, dtype=np.int32)
        for low, high, label in INDUSTRY_RANGES:
            cat_lookup[max(0, low) : min(10_000, high + 1)] = cat_index[label]
        self.cat_lookup = cp.asarray(cat_lookup)

        mcc_labels = sorted({str(value) for value in KNOWN_MCCS}) + ["-1"]
        mcc_index = {
            int(value): index for index, value in enumerate(dict.fromkeys(mcc_labels))
        }
        self.mcc_default = mcc_index[-1]
        mcc_lookup = np.full(10_000, self.mcc_default, dtype=np.int32)
        for value, index in mcc_index.items():
            if 0 <= value < len(mcc_lookup):
                mcc_lookup[value] = index
        self.mcc_lookup = cp.asarray(mcc_lookup)

        chip_labels = sorted(set(CHIP_MAPPING.values())) + ["UNK"]
        chip_index = {label: index for index, label in enumerate(dict.fromkeys(chip_labels))}
        self.chip_default = chip_index["UNK"]
        self.chip_index = {
            raw: chip_index.get(label, self.chip_default)
            for raw, label in CHIP_MAPPING.items()
        }

        state_labels = sorted(set(ALL_STATES)) + ["XX"]
        self.state_index = {
            label: index for index, label in enumerate(dict.fromkeys(state_labels))
        }
        self.state_default = self.state_index["XX"]
        self.month_offsets = cp.asarray(
            [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=cp.int32
        )

    def __call__(self, batch):
        cudf = self.cudf
        cp = self.cp
        frame = batch if isinstance(batch, cudf.DataFrame) else cudf.DataFrame(batch)
        input_rows = len(frame)

        user = frame["User"].astype("int64")
        card = frame["Card"].astype("int16")
        safe_card = card.clip(0, 9)
        year = frame["Year"].astype("int32")
        month = frame["Month"].astype("int32").clip(1, 12)
        day = frame["Day"].astype("int32").clip(1, 31)

        token_ids = cp.empty((input_rows, len(TOKEN_COLUMNS)), dtype=cp.uint16)

        amount = frame["Amount"].astype(str).str.lstrip("$").astype("float32")
        amount_bin = (
            (amount >= 10).astype("int16")
            + (amount >= 50).astype("int16")
            + (amount >= 100).astype("int16")
            + (amount >= 500).astype("int16")
            + (amount >= 1_000).astype("int16")
            + (amount >= 5_000).astype("int16")
        )
        token_ids[:, 0] = self.offset["amt"] + amount_bin.to_cupy()

        merchant = frame["Merchant Name"].fillna(0).astype("int64")
        token_ids[:, 1] = self.offset["merch"] + (
            merchant.abs() % self.merchant_hash_size
        ).to_cupy()

        mcc = frame["MCC"].fillna(-1).astype("int64").to_cupy()
        valid_mcc = (mcc >= 0) & (mcc < len(self.mcc_lookup))
        safe_mcc = cp.clip(mcc, 0, len(self.mcc_lookup) - 1)
        token_ids[:, 2] = self.offset["cat"] + cp.where(
            valid_mcc, self.cat_lookup[safe_mcc], self.cat_default
        )
        token_ids[:, 3] = self.offset["mcc"] + cp.where(
            valid_mcc, self.mcc_lookup[safe_mcc], self.mcc_default
        )

        time_text = frame["Time"].fillna("00:00").astype(str)
        hour = time_text.str.slice(0, 2).astype("int16").clip(0, 23)
        minute = time_text.str.slice(3, 5).astype("int16").clip(0, 59)
        hour_cp = hour.to_cupy()
        token_ids[:, 4] = self.offset["hour"] + hour_cp

        year_cp = year.to_cupy()
        month_cp = month.to_cupy()
        day_cp = day.to_cupy()
        adjusted_year = year_cp - (month_cp < 3)
        sunday_zero = (
            adjusted_year
            + adjusted_year // 4
            - adjusted_year // 100
            + adjusted_year // 400
            + self.month_offsets[month_cp - 1]
            + day_cp
        ) % 7
        token_ids[:, 5] = self.offset["dow"] + ((sunday_zero + 6) % 7)
        token_ids[:, 6] = self.offset["month"] + month_cp
        token_ids[:, 7] = self.offset["card"] + safe_card.to_cupy()

        chip = frame["Use Chip"].fillna("").astype(str).str.upper()
        chip_index = chip.map(self.chip_index).fillna(self.chip_default).astype("int16")
        token_ids[:, 8] = self.offset["chip"] + chip_index.to_cupy()

        zip_code = frame["Zip"].fillna(0).astype("int64").clip(0, 99_999).to_cupy()
        zip3 = cp.where(
            zip_code >= 10_000,
            zip_code // 100,
            cp.where(zip_code >= 1_000, zip_code // 10, zip_code),
        )
        token_ids[:, 9] = self.offset["zip3"] + zip3

        state = frame["Merchant State"].fillna("XX").astype(str).str.upper().str.strip()
        state = state.where(state != "", "XX")
        state_index = state.map(self.state_index).fillna(self.state_default).astype("int16")
        token_ids[:, 10] = self.offset["state"] + state_index.to_cupy()
        token_ids[:, 11] = self.offset["cust"] + user.clip(0, 2_999).to_cupy()

        txn_order = (
            ((((year_cp.astype(cp.int64) * 13 + month_cp) * 32 + day_cp) * 24 + hour_cp)
            * 60)
            + minute.to_cupy()
        )
        output = cudf.DataFrame({"User": user, "Card": card, "txn_order": txn_order})
        for index, name in enumerate(TOKEN_COLUMNS):
            output[name] = token_ids[:, index]
        if len(output) != input_rows:
            raise AssertionError("row-preserving tokenizer changed the row count")
        return output


_SEQUENCE_SCATTER_KERNEL = None
_PARTITION_SEQUENCE_SCATTER_KERNEL = None
_SEQUENCE_SCATTER_SOURCE = r'''extern "C" __global__
void scatter_sorted_tokens(
    const unsigned short* tokens,
    const int* order,
    long long row_stride,
    long long column_stride,
    unsigned short* output,
    int rows,
    int sequence_length,
    int chunk_size,
    unsigned short bos,
    unsigned short eos,
    unsigned short sep
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;

    int source_row = order[row];
    int sequence = row / chunk_size;
    int position = row - sequence * chunk_size;
    long long output_offset =
        ((long long)sequence) * sequence_length + 1 +
        ((long long)position) * 13;

    #pragma unroll
    for (int token = 0; token < 12; ++token) {
        output[output_offset + token] = tokens[
            ((long long)source_row) * row_stride +
            ((long long)token) * column_stride
        ];
    }
    output[output_offset + 12] =
        (position == chunk_size - 1 || row == rows - 1) ? eos : sep;
    if (position == 0) {
        output[((long long)sequence) * sequence_length] = bos;
    }
}
'''

_PARTITION_SEQUENCE_SCATTER_SOURCE = r'''extern "C" __global__
void scatter_segmented_tokens(
    const unsigned short* tokens,
    long long row_stride,
    long long column_stride,
    const int* boundaries,
    const long long* sequence_offsets,
    unsigned short* output,
    int groups,
    int sequence_length,
    int chunk_size,
    int initialize_final_tail,
    unsigned short pad,
    unsigned short bos,
    unsigned short eos,
    unsigned short sep
) {
    int group = blockIdx.x;
    if (group >= groups) return;

    int begin = boundaries[group];
    int end = boundaries[group + 1];
    int count = end - begin;
    long long first_sequence = sequence_offsets[group];

    if (initialize_final_tail) {
        int final_rows = (count - 1) % chunk_size + 1;
        int final_used = 1 + final_rows * 13;
        long long final_sequence = sequence_offsets[group + 1] - 1;
        long long final_offset = final_sequence * sequence_length;
        for (int position = final_used + threadIdx.x;
             position < sequence_length;
             position += blockDim.x) {
            output[final_offset + position] = pad;
        }
        __syncthreads();
    }

    for (int row = begin + threadIdx.x; row < end; row += blockDim.x) {
        int local_row = row - begin;
        int local_sequence = local_row / chunk_size;
        int position = local_row - local_sequence * chunk_size;
        long long sequence = first_sequence + local_sequence;
        long long output_offset =
            sequence * sequence_length + 1 + ((long long)position) * 13;

        #pragma unroll
        for (int token = 0; token < 12; ++token) {
            output[output_offset + token] = tokens[
                ((long long)row) * row_stride +
                ((long long)token) * column_stride
            ];
        }
        output[output_offset + 12] =
            (position == chunk_size - 1 || row == end - 1) ? eos : sep;
        if (position == 0) {
            output[sequence * sequence_length] = bos;
        }
    }
}
'''


def _get_sequence_scatter_kernel(cp):
    global _SEQUENCE_SCATTER_KERNEL
    if _SEQUENCE_SCATTER_KERNEL is None:
        _SEQUENCE_SCATTER_KERNEL = cp.RawKernel(
            _SEQUENCE_SCATTER_SOURCE, "scatter_sorted_tokens"
        )
    return _SEQUENCE_SCATTER_KERNEL


def _get_partition_sequence_scatter_kernel(cp):
    global _PARTITION_SEQUENCE_SCATTER_KERNEL
    if _PARTITION_SEQUENCE_SCATTER_KERNEL is None:
        _PARTITION_SEQUENCE_SCATTER_KERNEL = cp.RawKernel(
            _PARTITION_SEQUENCE_SCATTER_SOURCE, "scatter_segmented_tokens"
        )
    return _PARTITION_SEQUENCE_SCATTER_KERNEL


def build_sequences_gpu(
    group,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    chunk_size: int = SEQUENCE_CHUNK_SIZE,
):
    """Build chronological fixed-width sequences for one ``User/Card`` group."""
    import cupy as cp

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if 1 + chunk_size * (len(TOKEN_COLUMNS) + 1) > sequence_length:
        raise ValueError("chunk_size does not fit in sequence_length")

    rows = len(group)
    if rows == 0:
        return {"input_ids": np.zeros((0, sequence_length), dtype=np.uint16)}

    unsorted_tokens = group[list(TOKEN_COLUMNS)].to_cupy().astype(
        cp.uint16, copy=False
    )
    # Exchange implementations don't promise the same incoming order for rows
    # with an equal timestamp. Make the UDF's ordering structural: time first,
    # followed by token_00 .. token_11. Rows equal on every key are identical
    # transactions for sequence construction and can be interchanged safely.
    if str(group["txn_order"].dtype) == "int64" and all(
        str(group[column].dtype) == "uint16" for column in TOKEN_COLUMNS
    ):
        # libcudf sorts the existing typed columns directly. The previous
        # CuPy path stacked thirteen keys into one int64 matrix first, which
        # expanded every uint16 token key fourfold and dominated small-group
        # execution time.
        order = group.argsort(by=["txn_order", *TOKEN_COLUMNS])
    else:
        # Preserve the callable's historical coercion behavior for compatible
        # non-native inputs. The tokenizer's declared output uses the fast
        # int64/uint16 schema above.
        transaction_order = group["txn_order"].astype("int64").to_cupy()
        sort_keys = cp.stack(
            [
                unsorted_tokens[:, index]
                for index in reversed(range(len(TOKEN_COLUMNS)))
            ]
            + [transaction_order],
            axis=0,
        )
        order = cp.lexsort(sort_keys).astype(cp.int32, copy=False)

    sequence_count = (rows + chunk_size - 1) // chunk_size
    packed_length = 1 + chunk_size * (len(TOKEN_COLUMNS) + 1)
    if packed_length == sequence_length:
        # Every complete sequence is overwritten by the scatter. Initialize
        # only the unused tail of the final partial sequence.
        output = cp.empty((sequence_count, sequence_length), dtype=cp.uint16)
        final_rows = (rows - 1) % chunk_size + 1
        final_used = 1 + final_rows * (len(TOKEN_COLUMNS) + 1)
        if final_used < sequence_length:
            output[-1, final_used:] = PAD_TOKEN_ID
    else:
        # Non-default sequence widths contain padding after every chunk.
        output = cp.full(
            (sequence_count, sequence_length), PAD_TOKEN_ID, dtype=cp.uint16
        )

    scatter = _get_sequence_scatter_kernel(cp)
    blocks = (rows + 255) // 256
    scatter(
        (blocks,),
        (256,),
        (
            unsorted_tokens,
            order,
            np.int64(unsorted_tokens.strides[0] // unsorted_tokens.itemsize),
            np.int64(unsorted_tokens.strides[1] // unsorted_tokens.itemsize),
            output,
            np.int32(rows),
            np.int32(sequence_length),
            np.int32(chunk_size),
            np.uint16(BOS_TOKEN_ID),
            np.uint16(EOS_TOKEN_ID),
            np.uint16(SEP_TOKEN_ID),
        ),
    )
    return {"input_ids": cp.asnumpy(output)}


def build_sequences_partition_gpu(
    batch,
    context,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    chunk_size: int = SEQUENCE_CHUNK_SIZE,
):
    """Apply ``build_sequences_gpu`` to a sorted partition in one GPU call."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if 1 + chunk_size * (len(TOKEN_COLUMNS) + 1) > sequence_length:
        raise ValueError("chunk_size does not fit in sequence_length")

    boundaries = np.asarray(context.input_group_boundaries, dtype=np.int64)
    if boundaries.ndim != 1 or not len(boundaries):
        raise ValueError("partition group boundaries must be a nonempty vector")
    if boundaries[0] != 0 or boundaries[-1] != len(batch):
        raise ValueError(
            "partition group boundaries must span the complete input batch"
        )
    if len(boundaries) > 1 and np.any(boundaries[1:] <= boundaries[:-1]):
        raise ValueError("partition group boundaries must be strictly increasing")
    num_groups = len(boundaries) - 1
    if int(context.num_groups) != num_groups:
        raise ValueError(
            "partition context group count does not match its boundaries"
        )
    native_schema = str(batch["txn_order"].dtype) == "int64" and all(
        str(batch[column].dtype) == "uint16" for column in TOKEN_COLUMNS
    )
    if not native_schema:
        raise TypeError(
            "build_sequences_partition_gpu requires tokenizer output with "
            "txn_order=int64 and every token column=uint16; refusing to run "
            "the optimized partition UDF on an incompatible schema"
        )
    if num_groups == 0:
        return {"input_ids": np.zeros((0, sequence_length), dtype=np.uint16)}
    if len(batch) > np.iinfo(np.int32).max:
        raise ValueError("partition execution supports at most INT32_MAX input rows")

    import cudf
    import cupy as cp
    import pylibcudf as plc

    values, value_metadata = batch.loc[
        :, list(TOKEN_COLUMNS)
    ].to_pylibcudf(copy=False)
    keys, _ = batch.loc[
        :, ["txn_order", *TOKEN_COLUMNS]
    ].to_pylibcudf(copy=False)
    device_boundaries = cp.asarray(boundaries, dtype=cp.int32)
    sorted_values = plc.sorting.segmented_sort_by_key(
        values,
        keys,
        plc.Column.from_array(device_boundaries),
        [plc.types.Order.ASCENDING] * (len(TOKEN_COLUMNS) + 1),
        [plc.types.NullOrder.AFTER] * (len(TOKEN_COLUMNS) + 1),
    )
    sorted_tokens = cudf.DataFrame.from_pylibcudf(
        sorted_values, value_metadata
    ).to_cupy().astype(cp.uint16, copy=False)

    counts = np.diff(boundaries)
    sequence_counts = (counts + chunk_size - 1) // chunk_size
    sequence_offsets = np.empty(num_groups + 1, dtype=np.int64)
    sequence_offsets[0] = 0
    np.cumsum(sequence_counts, out=sequence_offsets[1:])
    output_rows = int(sequence_offsets[-1])
    packed_length = 1 + chunk_size * (len(TOKEN_COLUMNS) + 1)
    if packed_length == sequence_length:
        output = cp.empty((output_rows, sequence_length), dtype=cp.uint16)
        initialize_final_tail = 1
    else:
        output = cp.full(
            (output_rows, sequence_length), PAD_TOKEN_ID, dtype=cp.uint16
        )
        initialize_final_tail = 0

    scatter = _get_partition_sequence_scatter_kernel(cp)
    scatter(
        (num_groups,),
        (256,),
        (
            sorted_tokens,
            np.int64(sorted_tokens.strides[0] // sorted_tokens.itemsize),
            np.int64(sorted_tokens.strides[1] // sorted_tokens.itemsize),
            device_boundaries,
            cp.asarray(sequence_offsets),
            output,
            np.int32(num_groups),
            np.int32(sequence_length),
            np.int32(chunk_size),
            np.int32(initialize_final_tail),
            np.uint16(PAD_TOKEN_ID),
            np.uint16(BOS_TOKEN_ID),
            np.uint16(EOS_TOKEN_ID),
            np.uint16(SEP_TOKEN_ID),
        ),
    )
    return {"input_ids": cp.asnumpy(output)}


def build_sequences_reference(
    token_rows: np.ndarray,
    transaction_order: np.ndarray,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    chunk_size: int = SEQUENCE_CHUNK_SIZE,
) -> np.ndarray:
    """CPU reference used only by dependency-light contract tests."""
    token_rows = np.asarray(token_rows, dtype=np.uint16)
    transaction_order = np.asarray(transaction_order, dtype=np.int64)
    if token_rows.ndim != 2 or token_rows.shape[1] != len(TOKEN_COLUMNS):
        raise ValueError("token_rows must have shape (rows, 12)")
    if len(token_rows) != len(transaction_order):
        raise ValueError("token rows and transaction order lengths differ")
    sort_keys = tuple(
        token_rows[:, index] for index in reversed(range(len(TOKEN_COLUMNS)))
    ) + (transaction_order,)
    order = np.lexsort(sort_keys)
    token_rows = token_rows[order]
    outputs = []
    for start in range(0, len(token_rows), chunk_size):
        sequence = [BOS_TOKEN_ID]
        chunk = token_rows[start : start + chunk_size]
        for index, row in enumerate(chunk):
            sequence.extend(int(value) for value in row)
            sequence.append(EOS_TOKEN_ID if index == len(chunk) - 1 else SEP_TOKEN_ID)
        array = np.full(sequence_length, PAD_TOKEN_ID, dtype=np.uint16)
        array[: len(sequence)] = sequence
        outputs.append(array)
    if not outputs:
        return np.zeros((0, sequence_length), dtype=np.uint16)
    return np.stack(outputs)
