#ifndef NN_H
#define NN_H
#include <stddef.h>
#include <stdint.h>
#include "model.h"

/* All weights/tables are const flash objects. This is the only inference RAM. */
typedef struct {
    int16_t features[NN_CONTEXT * NN_EMBEDDING];
    int16_t hidden[NN_HIDDEN];
#if NN_HIDDEN_LAYERS >= 2
    int16_t hidden2[NN_HIDDEN];
#endif
    union { int32_t logits[257]; uint32_t cdf[257]; } values;
    uint32_t remainders[256];
    uint16_t order[256];
    uint16_t history[NN_CONTEXT];
} NNState;

void nn_reset(NNState *state);
void nn_advance(NNState *state, uint8_t symbol);
const uint32_t *nn_predict(NNState *state);
int nn_compress(NNState *state, const uint8_t *input, size_t length,
                uint8_t *output, size_t capacity);
int nn_decompress(NNState *state, const uint8_t *input, size_t payload_length,
                  size_t length, uint8_t *output, size_t capacity);
#endif
