/* Bit-exact MCPNN001/002/003 integer inference. No allocator or floating point. */
#include "nn.h"
#include <limits.h>

static int32_t dot(const int8_t *weights, const int16_t *inputs, unsigned count) {
    int32_t sum = 0;
    unsigned i = 0;
#if defined(__ARM_FEATURE_DSP) && !defined(NN_SCALAR)
    for (; i + 1 < count; i += 2) {
        uint32_t w = (uint16_t)(int16_t)weights[i]
                   | ((uint32_t)(uint16_t)(int16_t)weights[i + 1] << 16);
        uint32_t x = (uint16_t)inputs[i] | ((uint32_t)(uint16_t)inputs[i + 1] << 16);
        __asm__("smlad %0, %1, %2, %0" : "+r"(sum) : "r"(w), "r"(x));
    }
#endif
    for (; i < count; ++i) sum += weights[i] * inputs[i];
    return sum;
}

void nn_reset(NNState *s) {
    for (unsigned i = 0; i < NN_CONTEXT; ++i) s->history[i] = 256;
}

void nn_advance(NNState *s, uint8_t symbol) {
    for (unsigned i = 1; i < NN_CONTEXT; ++i) s->history[i - 1] = s->history[i];
    s->history[NN_CONTEXT - 1] = symbol;
}

const uint32_t *nn_predict(NNState *s) {
    for (unsigned p = 0; p < NN_CONTEXT; ++p)
        for (unsigned j = 0; j < NN_EMBEDDING; ++j)
            s->features[p * NN_EMBEDDING + j] = nn_embedding[s->history[p] * NN_EMBEDDING + j];
    for (unsigned i = 0; i < NN_HIDDEN; ++i) {
        int32_t a = dot(nn_w1 + i * NN_CONTEXT * NN_EMBEDDING, s->features,
                        NN_CONTEXT * NN_EMBEDDING);
        int32_t index = ((a * NN_MUL1) >> NN_SHIFT1) + nn_b1[i];
        if (index < -2048) index = -2048;
        if (index > 2048) index = 2048;
        s->hidden[i] = nn_tanh[index + 2048];
    }
    const int16_t *output_hidden = s->hidden;
#if NN_HIDDEN_LAYERS >= 2
    for (unsigned i = 0; i < NN_HIDDEN; ++i) {
        int32_t a = dot(nn_w_mid + i * NN_HIDDEN, s->hidden, NN_HIDDEN);
        int32_t index = (((a >> NN_PRE_MID) * NN_MUL_MID) >> NN_SHIFT_MID) + nn_b_mid[i];
        if (index < -2048) index = -2048;
        if (index > 2048) index = 2048;
        s->hidden2[i] = nn_tanh[index + 2048];
    }
    output_hidden = s->hidden2;
#endif
#if NN_HIDDEN_LAYERS == 3
    /* Layer one is no longer needed: reuse its buffer for layer three. */
    for (unsigned i = 0; i < NN_HIDDEN; ++i) {
        int32_t a = dot(nn_w_mid2 + i * NN_HIDDEN, s->hidden2, NN_HIDDEN);
        int32_t index = (((a >> NN_PRE_MID2) * NN_MUL_MID2) >> NN_SHIFT_MID2) + nn_b_mid2[i];
        if (index < -2048) index = -2048;
        if (index > 2048) index = 2048;
        s->hidden[i] = nn_tanh[index + 2048];
    }
    output_hidden = s->hidden;
#endif
    int32_t maximum = INT32_MIN;
    for (unsigned i = 0; i < 256; ++i) {
        int32_t a = dot(nn_w2 + i * NN_HIDDEN, output_hidden, NN_HIDDEN);
        int32_t value = (((a >> NN_PRE2) * NN_MUL2) >> NN_SHIFT2) + nn_b2[i];
        s->values.logits[i] = value;
        if (value > maximum) maximum = value;
    }
    uint32_t mass = 0;
    for (unsigned i = 0; i < 256; ++i) {
        uint32_t distance = (uint32_t)(maximum - s->values.logits[i]);
        uint32_t score = distance <= 4096 ? nn_exp[distance] : 0;
        s->values.cdf[i] = score;
        mass += score;
    }
    uint32_t allocated = 0;
    for (unsigned i = 0; i < 256; ++i) {
        uint32_t numerator = s->values.cdf[i] * 65280u;
        uint32_t frequency = 1 + numerator / mass;
        s->values.cdf[i] = frequency;
        s->remainders[i] = numerator % mass;
        s->order[i] = (uint16_t)i;
        allocated += frequency;
    }
    /* Stable descending residual sort: equal residuals retain byte order. */
    for (unsigned i = 1; i < 256; ++i) {
        uint16_t symbol = s->order[i];
        unsigned j = i;
        while (j && s->remainders[s->order[j - 1]] < s->remainders[symbol]) {
            s->order[j] = s->order[j - 1];
            --j;
        }
        s->order[j] = symbol;
    }
    for (unsigned i = 0; i < 65536u - allocated; ++i) ++s->values.cdf[s->order[i]];
    /* Prefix sum in place, preserving each overwritten frequency in a register. */
    uint32_t previous = s->values.cdf[0], cumulative = 0;
    s->values.cdf[0] = 0;
    for (unsigned i = 0; i < 256; ++i) {
        uint32_t next = i < 255 ? s->values.cdf[i + 1] : 0;
        cumulative += previous;
        s->values.cdf[i + 1] = cumulative;
        previous = next;
    }
    return s->values.cdf;
}
