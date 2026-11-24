/* Same 32-bit inclusive arithmetic intervals as pico/nn_codec.py.
 * Interval products deliberately use uint64_t; inference only needs int32_t. */
#include "nn.h"
#define HALF UINT32_C(0x80000000)
#define QUARTER UINT32_C(0x40000000)

typedef struct { uint8_t *out; size_t capacity, length; unsigned used; uint8_t byte; } Writer;
static int bit_write(Writer *w, unsigned bit) {
    w->byte = (uint8_t)((w->byte << 1) | bit);
    if (++w->used == 8) {
        if (w->length == w->capacity) return -1;
        w->out[w->length++] = w->byte;
        w->used = 0; w->byte = 0;
    }
    return 0;
}
static int emit(Writer *w, unsigned bit, size_t *pending) {
    if (bit_write(w, bit)) return -1;
    while (*pending) { if (bit_write(w, bit ^ 1)) return -1; --*pending; }
    return 0;
}

int nn_compress(NNState *s, const uint8_t *input, size_t length,
                uint8_t *output, size_t capacity) {
    Writer w = {output, capacity, 0, 0, 0};
    uint32_t low = 0, high = UINT32_MAX;
    size_t pending = 0;
    nn_reset(s);
    for (size_t i = 0; i < length; ++i) {
        const uint32_t *cdf = nn_predict(s);
        unsigned symbol = input[i];
        uint64_t width = (uint64_t)high - low + 1;
        high = low + (uint32_t)((width * cdf[symbol + 1]) >> 16) - 1;
        low += (uint32_t)((width * cdf[symbol]) >> 16);
        for (;;) {
            unsigned bit;
            if (high < HALF) bit = 0;
            else if (low >= HALF) { bit = 1; low -= HALF; high -= HALF; }
            else if (low >= QUARTER && high < 3u * QUARTER) {
                ++pending; low -= QUARTER; high -= QUARTER;
                low <<= 1; high = (high << 1) | 1; continue;
            } else break;
            if (emit(&w, bit, &pending)) return -1;
            low <<= 1; high = (high << 1) | 1;
        }
        nn_advance(s, input[i]);
    }
    ++pending;
    if (emit(&w, low < QUARTER ? 0 : 1, &pending)) return -1;
    if (w.used) {
        if (w.length == capacity) return -1;
        w.out[w.length++] = (uint8_t)(w.byte << (8 - w.used));
    }
    return (int)w.length;
}

static unsigned bit_read(const uint8_t *input, size_t size, size_t *position) {
    size_t p = (*position)++;
    return p / 8 < size ? (input[p / 8] >> (7 - p % 8)) & 1 : 0;
}

int nn_decompress(NNState *s, const uint8_t *input, size_t payload_length,
                  size_t length, uint8_t *output, size_t capacity) {
    if (length > capacity) return -1;
    uint32_t low = 0, high = UINT32_MAX, code = 0;
    size_t position = 0;
    nn_reset(s);
    for (unsigned i = 0; i < 32; ++i) code = (code << 1) | bit_read(input, payload_length, &position);
    for (size_t i = 0; i < length; ++i) {
        const uint32_t *cdf = nn_predict(s);
        uint64_t width = (uint64_t)high - low + 1;
        if (code < low || code > high) return -2;
        uint32_t value = (uint32_t)(((((uint64_t)code - low + 1) << 16) - 1) / width);
        unsigned left = 0, right = 256;
        while (left + 1 < right) {
            unsigned middle = (left + right) / 2;
            if (cdf[middle] <= value) left = middle; else right = middle;
        }
        unsigned symbol = left;
        high = low + (uint32_t)((width * cdf[symbol + 1]) >> 16) - 1;
        low += (uint32_t)((width * cdf[symbol]) >> 16);
        for (;;) {
            if (high < HALF) { }
            else if (low >= HALF) { low -= HALF; high -= HALF; code -= HALF; }
            else if (low >= QUARTER && high < 3u * QUARTER) {
                low -= QUARTER; high -= QUARTER; code -= QUARTER;
            } else break;
            low <<= 1; high = (high << 1) | 1;
            code = (code << 1) | bit_read(input, payload_length, &position);
        }
        output[i] = (uint8_t)symbol;
        nn_advance(s, (uint8_t)symbol);
    }
    return (int)length;
}
