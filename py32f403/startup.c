#include <stdint.h>
extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;
void SystemInit(void);
int main(void);
void Default_Handler(void) { for (;;) __asm__("wfi"); }
void Reset_Handler(void) {
    uint32_t *src = &_sidata;
    for (uint32_t *p = &_sdata; p < &_edata;) *p++ = *src++;
    for (uint32_t *p = &_sbss; p < &_ebss;) *p++ = 0;
    SystemInit();
    main();
    Default_Handler();
}
__attribute__((section(".isr_vector"), used))
const uintptr_t vectors[16 + 60] = {
    [0] = (uintptr_t)&_estack,
    [1] = (uintptr_t)Reset_Handler,
    [2 ... 75] = (uintptr_t)Default_Handler
};
