/* RAM-resident page programmer; sequences follow Puya HAL flash driver.
 * No mass erase, option programming, or read-protection changes. */
#include "py32f403xD.h"

typedef struct {
    uint32_t address, pages, source;
    volatile uint32_t status, completed;
} Command;
static int wait_flash(void) {
    uint32_t timeout = 2000000;
    while (FLASH->SR & FLASH_SR_BSY) if (!--timeout) return -1;
    if (FLASH->SR & FLASH_SR_WRPERR) return -2;
    FLASH->SR = FLASH_SR_EOP | FLASH_SR_WRPERR;
    return 0;
}
__attribute__((section(".entry"), used))
void flash_entry(Command *command) {
    __disable_irq();
    command->status = 0xFFFFFFFF; command->completed = 0;
    if (command->address < 0x08000000 || command->address % 256 || !command->pages
        || command->pages > 16 || command->address + command->pages * 256 > 0x08060000
        || command->source != 0x20004000) { command->status = 10; goto done; }
    /* Use the reset HSI clock before any erase/program operation. */
    RCC->CR |= RCC_CR_HSION;
    while (!(RCC->CR & RCC_CR_HSIRDY)) { }
    RCC->CFGR &= ~RCC_CFGR_SW;
    while (RCC->CFGR & RCC_CFGR_SWS) { }
    FLASH->KEYR = FLASH_KEY1; FLASH->KEYR = FLASH_KEY2;
    if (FLASH->CR & FLASH_CR_LOCK) { command->status = 11; goto done; }
    for (unsigned page = 0; page < command->pages; ++page) {
        volatile uint32_t *dest = (uint32_t *)(command->address + page * 256);
        const uint32_t *src = (const uint32_t *)(command->source + page * 256);
        if (wait_flash()) { command->status = 12; goto lock; }
        FLASH->CR = (FLASH->CR & ~(FLASH_CR_PG | FLASH_CR_PGSTRT)) | FLASH_CR_PER;
        dest[0] = 0xFF;
        if (wait_flash()) { command->status = 13; goto lock; }
        FLASH->CR &= ~FLASH_CR_PER;
        FLASH->CR |= FLASH_CR_PG;
        for (unsigned i = 0; i < 64; ++i) {
            if (i == 63) FLASH->CR |= FLASH_CR_PGSTRT;
            dest[i] = src[i];
        }
        if (wait_flash()) { command->status = 14; goto lock; }
        FLASH->CR &= ~(FLASH_CR_PG | FLASH_CR_PGSTRT);
        ++command->completed;
    }
    command->status = 0;
lock:
    FLASH->CR |= FLASH_CR_LOCK;
done:
    __BKPT(0);
    for (;;) { }
}
