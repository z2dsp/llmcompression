#include "platform.h"
#include "nn.h"

#define MAX_MESSAGE 512
#define MAX_PAYLOAD 1100
static NNState state;
static uint8_t message[MAX_MESSAGE], payload[MAX_PAYLOAD];
static char line[2 * MAX_PAYLOAD + 40];
extern uint32_t _sdata, _ebss, _estack;

static int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int unhex(const char *text, uint8_t *out, unsigned capacity) {
    unsigned n = 0;
    while (*text) {
        int a = nibble(*text++);
        if (!*text) return -1;
        int b = nibble(*text++);
        if (a < 0 || b < 0 || n == capacity) return -1;
        out[n++] = (uint8_t)((a << 4) | b);
    }
    return (int)n;
}
static void hex(const uint8_t *data, unsigned count) {
    const char digits[] = "0123456789abcdef";
    for (unsigned i = 0; i < count; ++i) {
        uart_putc(digits[data[i] >> 4]); uart_putc(digits[data[i] & 15]);
    }
}
static void field(uint32_t value) { uart_putc(' '); uart_uint(value); }
static uint32_t decimal(char **p) {
    uint32_t result = 0;
    for (unsigned digits = 0; **p >= '0' && **p <= '9'; ++digits) {
        if (digits >= 6) return UINT32_MAX;
        result = result * 10 + (*(*p)++ - '0');
    }
    return result;
}
static uint32_t stack_used(void) {
    uint32_t *p = &_ebss;
    while (p < &_estack && *p == 0xA55A3CC3) ++p;
    return (uint32_t)((uintptr_t)&_estack - (uintptr_t)p);
}
static void command(void) {
    if (line[0] == 'I' && !line[1]) {
        uart_text("I"); field(SystemCoreClock); field(sizeof(state));
        field((uintptr_t)&_ebss - (uintptr_t)&_sdata);
        field(stack_used()); field(NN_MODEL_BYTES);
        uart_text(" " NN_MODEL_SHA " DSP\n");
    } else if (line[0] == 'P' && line[1] == ' ') {
        int n = unhex(line + 2, message, MAX_MESSAGE);
        if (n != NN_CONTEXT * 2) { uart_text("ERR history\n"); return; }
        for (unsigned i = 0; i < NN_CONTEXT; ++i) {
            state.history[i] = ((uint16_t)message[2 * i] << 8) | message[2 * i + 1];
            if (state.history[i] > 256) { uart_text("ERR symbol\n"); return; }
        }
        uint32_t start = DWT->CYCCNT;
        const uint32_t *cdf = nn_predict(&state);
        uint32_t cycles = DWT->CYCCNT - start;
        uart_text("P"); field(cycles); uart_putc(' ');
        hex((const uint8_t *)cdf, 257 * 4); uart_putc('\n');
    } else if (line[0] == 'B' && line[1] == ' ') {
        char *p = line + 2;
        uint32_t count = decimal(&p);
        if (*p || !count || count > 1000) { uart_text("ERR count\n"); return; }
        nn_reset(&state); nn_predict(&state);
        uint32_t start = DWT->CYCCNT;
        for (unsigned i = 0; i < count; ++i) nn_predict(&state);
        uint32_t cycles = DWT->CYCCNT - start;
        uart_text("B"); field(count); field(cycles); uart_putc('\n');
    } else if (line[0] == 'E' && line[1] == ' ') {
        int n = unhex(line + 2, message, MAX_MESSAGE);
        if (n < 0) { uart_text("ERR hex\n"); return; }
        uint32_t start = DWT->CYCCNT;
        int size = nn_compress(&state, message, (unsigned)n, payload, sizeof(payload));
        uint32_t cycles = DWT->CYCCNT - start;
        if (size < 0) { uart_text("ERR capacity\n"); return; }
        uart_text("E"); field(cycles); field((uint32_t)size); uart_putc(' ');
        hex(payload, (unsigned)size); uart_putc('\n');
    } else if (line[0] == 'D' && line[1] == ' ') {
        char *p = line + 2;
        uint32_t length = decimal(&p);
        if (*p != ' ' || length > MAX_MESSAGE) { uart_text("ERR length\n"); return; }
        int size = unhex(p + 1, payload, sizeof(payload));
        if (size < 0) { uart_text("ERR hex\n"); return; }
        uint32_t start = DWT->CYCCNT;
        int result = nn_decompress(&state, payload, (unsigned)size, length, message, sizeof(message));
        uint32_t cycles = DWT->CYCCNT - start;
        if (result < 0) { uart_text("ERR decode\n"); return; }
        uart_text("D"); field(cycles); field(length); uart_putc(' ');
        hex(message, length); uart_putc('\n');
    } else uart_text("ERR command\n");
}
int main(void) {
    /* Paint unused RAM for a conservative stack high-water estimate. */
    uintptr_t limit = __get_MSP() - 512;
    for (uint32_t *p = &_ebss; (uintptr_t)p < limit; ++p) *p = 0xA55A3CC3;
    platform_init();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uart_text("PY32 NN READY\n");
    for (;;) {
        unsigned n = 0; int overflow = 0;
        for (;;) {
            char ch = uart_getc();
            if (ch == '\n') break;
            if (ch == '\r') continue;
            if (n + 1 < sizeof(line)) line[n++] = ch; else overflow = 1;
        }
        line[n] = 0;
        if (overflow) uart_text("ERR line\n"); else command();
    }
}
