ENTRY(flash_entry)
MEMORY { RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 12K }
SECTIONS {
 .text : { KEEP(*(.entry)) *(.text*) *(.rodata*) } > RAM
 /DISCARD/ : { *(.ARM.exidx*) *(.comment*) *(.note*) }
}
