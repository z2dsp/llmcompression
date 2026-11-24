#ifndef PLATFORM_H
#define PLATFORM_H
#include <stdint.h>
#include "py32f403xD.h"
void platform_init(void);
void uart_putc(char ch);
char uart_getc(void);
void uart_text(const char *text);
void uart_uint(uint32_t value);
#endif
