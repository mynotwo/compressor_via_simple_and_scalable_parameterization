"""Integer binary arithmetic coder (Witten-Neal-Cleary style, 32-bit state).

The coder is model-agnostic: each call receives a cumulative-frequency
interval ``[cum_lo, cum_hi)`` out of ``total``. ``total`` must stay below
2**30 so the interval never collapses.
"""

from __future__ import annotations

PRECISION = 32
FULL = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
MAX_TOTAL = 1 << (PRECISION - 2)


class _BitWriter:
    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, bit: int) -> None:
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.out.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def getvalue(self) -> bytes:
        if self.nbits:
            return bytes(self.out) + bytes([self.acc << (8 - self.nbits)])
        return bytes(self.out)


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = FULL
        self.pending = 0
        self._bits = _BitWriter()

    def _emit(self, bit: int) -> None:
        self._bits.write(bit)
        inv = bit ^ 1
        for _ in range(self.pending):
            self._bits.write(inv)
        self.pending = 0

    def encode(self, cum_lo: int, cum_hi: int, total: int) -> None:
        if not 0 <= cum_lo < cum_hi <= total <= MAX_TOTAL:
            raise ValueError(f"bad interval [{cum_lo},{cum_hi}) of {total}")
        rng = self.high - self.low + 1
        self.high = self.low + rng * cum_hi // total - 1
        self.low = self.low + rng * cum_lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return self._bits.getvalue()


class ArithmeticDecoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0  # bit position
        self.low = 0
        self.high = FULL
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self._read()

    def _read(self) -> int:
        byte_i = self.pos >> 3
        if byte_i >= len(self.data):
            self.pos += 1
            return 0
        bit = (self.data[byte_i] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def target(self, total: int) -> int:
        """Scaled code value; the symbol is the one whose interval contains it."""
        rng = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // rng

    def consume(self, cum_lo: int, cum_hi: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + rng * cum_hi // total - 1
        self.low = self.low + rng * cum_lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._read()
