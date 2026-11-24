#include "platform.h"
#include <stddef.h>

void platform_init(void) {
    /* Internal 8 MHz oscillator x18: no external crystal assumption. */
    RCC->CR |= RCC_CR_HSION;
    while (!(RCC->CR & RCC_CR_HSIRDY)) { }
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_HSI;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_HSI) { }
    RCC->CR &= ~RCC_CR_PLLON;
    while (RCC->CR & RCC_CR_PLLRDY) { }
    FLASH->ACR = (FLASH->ACR & ~FLASH_ACR_LATENCY) | 6;
    RCC->CFGR = RCC_CFGR_PLLMULL18 | RCC_CFGR_PPRE1_DIV2 | RCC_CFGR_PPRE2_2;
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY)) { }
    RCC->CFGR |= RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) { }
    SystemCoreClock = 144000000;

    RCC->AHB2ENR |= RCC_AHB2ENR_IOPAEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
    (void)RCC->AHB2ENR;
    GPIOA->MODER = (GPIOA->MODER & ~((3u << 18) | (3u << 20))) | (2u << 18) | (2u << 20);
    GPIOA->AFR[1] = (GPIOA->AFR[1] & ~((15u << 4) | (15u << 8))) | (2u << 4) | (2u << 8);
    GPIOA->OTYPER &= ~((1u << 9) | (1u << 10));
    GPIOA->OSPEEDR |= (3u << 18) | (3u << 20);
    GPIOA->PUPDR = (GPIOA->PUPDR & ~((3u << 18) | (3u << 20))) | (1u << 20);
    USART1->CR1 = 0; USART1->CR2 = 0; USART1->CR3 = 0;
    USART1->BRR = 72000000u / 115200u;
    USART1->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void uart_putc(char ch) { while (!(USART1->SR & USART_SR_TXE)) { } USART1->DR = (uint8_t)ch; }
char uart_getc(void) { while (!(USART1->SR & USART_SR_RXNE)) { } return (char)USART1->DR; }
void uart_text(const char *s) { while (*s) uart_putc(*s++); }
void uart_uint(uint32_t value) {
    char digits[10]; unsigned n = 0;
    do { digits[n++] = (char)('0' + value % 10); value /= 10; } while (value);
    while (n) uart_putc(digits[--n]);
}
void *memcpy(void *dest, const void *src, size_t n) {
    uint8_t *d = dest; const uint8_t *s = src;
    while (n--) *d++ = *s++;
    return dest;
}
void *memset(void *dest, int value, size_t n) {
    uint8_t *d = dest; while (n--) *d++ = (uint8_t)value; return dest;
}
