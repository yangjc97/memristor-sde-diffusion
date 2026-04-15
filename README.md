# memristor-sde-diffusion

Code for "Resistive memory-based neural differential equation solver for score-based diffusion model"

## Abstract

While AI-Generated Content (AIGC) strives to replicate human imagination, current models like score-based diffusion remain slow and energy-intensive. This inefficiency stems from conventional digital computers, where physically separated storage and processing units cause data-transfer bottlenecks, and discrete operations disrupt naturally continuous generation dynamics. Here we show a brain-inspired, analog in-memory computing system that overcomes these limitations. By employing resistive memory, our system seamlessly integrates storage and computation to act as a time-continuous neural differential equation solver. We experimentally validate our solution with 180 nm resistive memory in-memory computing macros. Demonstrating equivalent generative quality to the software baseline, our system achieved enhancements in generative speed for both unconditional and conditional generation tasks, by factors of 69.0 and 116.5, respectively. Moreover, it accomplished reductions in energy consumption of 31.5% and 52.0%. Our approach expands the horizon for hardware solutions in edge computing for generative AI applications.

## Requirements

The codes are tested on Ubuntu 20.04, CUDA 11.1 with the following packages:

```shell
torch == 2.1.2
scipy == 1.10.0
numpy == 1.23.5
pandas == 1.5.3
```

## MCU code

The MCU code is in the `MCU_Project` folder. The code is written in C and can be compiled in `Keil`. The code is tested on the STM32F407 board.
