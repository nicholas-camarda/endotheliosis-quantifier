"""Local Label Studio brush RLE helpers.

This module vendors the minimal brush RLE encode/decode contract used by the
project so the production CLI does not depend on `label_studio_converter`
import side effects.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class _InputStream:
    def __init__(self, data: str) -> None:
        self.data = data
        self.index = 0

    def read(self, size: int) -> int:
        chunk = self.data[self.index : self.index + size]
        self.index += size
        return int(chunk or '0', 2)


def _access_bit(data: list[int], num: int) -> int:
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def _bytes_to_bit_string(data: list[int]) -> str:
    return ''.join(str(_access_bit(data, i)) for i in range(len(data) * 8))


def _bits_to_bytes(bit_string: str, width: int = 8) -> list[int]:
    return [
        int(bit_string[index : index + width], 2)
        for index in range(0, len(bit_string), width)
    ]


def _base_rle_encode(inarray: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(inarray)
    length = len(array)
    if length == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=array.dtype),
        )
    unequal = array[1:] != array[:-1]
    indices = np.append(np.where(unequal), length - 1)
    run_lengths = np.diff(np.append(-1, indices))
    positions = np.cumsum(np.append(0, run_lengths))[:-1]
    return run_lengths, positions, array[indices]


def encode_rle(
    arr: Iterable[int], wordsize: int = 8, rle_sizes: list[int] | None = None
) -> list[int]:
    """Encode a flattened RGBA array into Label Studio brush RLE."""
    sizes = rle_sizes or [3, 4, 8, 16]
    num_values = len(arr)
    base = f'{num_values:032b}{wordsize - 1:05b}' + ''.join(
        f'{size - 1:04b}' for size in sizes
    )
    out = ''
    run_lengths, _positions, values = _base_rle_encode(arr)
    for run_length, value in zip(run_lengths, values):
        if run_length == 1:
            out += '0'
            out += '00'
            out += '000'
            out += f'{int(value):08b}'
            continue

        if run_length <= 8:
            out += '1'
            out += '00'
            out += f'{run_length - 1:03b}'
            out += f'{int(value):08b}'
            continue

        if run_length <= 16:
            out += '1'
            out += '01'
            out += f'{run_length - 1:04b}'
            out += f'{int(value):08b}'
            continue

        if run_length <= 256:
            out += '1'
            out += '10'
            out += f'{run_length - 1:08b}'
            out += f'{int(value):08b}'
            continue

        remaining = int(run_length)
        while remaining > 2**16:
            out += '1'
            out += '11'
            out += f'{2**16 - 1:016b}'
            out += f'{int(value):08b}'
            remaining -= 2**16
        out += '1'
        out += '11'
        out += f'{remaining - 1:016b}'
        out += f'{int(value):08b}'

    # Match Label Studio's upstream encoder exactly. When the bitstream length is
    # already byte-aligned, it still appends one full zero byte.
    padding = 8 - len(base + out) % 8
    total = base + out + ('0' * padding)
    return _bits_to_bytes(total)


def mask2rle(mask: np.ndarray) -> list[int]:
    """Convert a 2D uint8 or int mask into Label Studio brush RLE."""
    if len(mask.shape) != 2:
        raise ValueError('mask must be 2D np.array')
    if mask.dtype != np.uint8 and mask.dtype != int:
        raise ValueError('mask must be uint8 or int')
    flat = np.repeat(mask.ravel(), 4)
    return encode_rle(flat)


def decode_rle(rle: Iterable[int]) -> np.ndarray:
    """Decode Label Studio brush RLE into a flat uint8 RGBA array."""
    values = [int(item) for item in rle]
    stream = _InputStream(_bytes_to_bit_string(values))
    num_values = stream.read(32)
    word_size = stream.read(5) + 1
    rle_sizes = [stream.read(4) + 1 for _ in range(4)]

    index = 0
    out = np.zeros(num_values, dtype=np.uint8)
    while index < num_values:
        is_series = stream.read(1)
        next_index = index + 1 + stream.read(rle_sizes[stream.read(2)])
        if is_series:
            out[index:next_index] = stream.read(word_size)
            index = next_index
            continue
        while index < next_index:
            out[index] = stream.read(word_size)
            index += 1
    return out
