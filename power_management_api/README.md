# API Definition

## Power Log

### `start_power_log(period: float, path: str) -> int` and `stop_power_log() -> int`

#### Purpose

Track the real-time power consumption of the SoC. Generate a log in real-time, so that the energy consumption in a certain time can be analyzed.

#### Implementation

The power goes through the power line `VDD_IN` can be read from a virtual fs. When `start_power_log` is called, it split a sub-process recording power in a certain period and append log to a specified file. When `stop_power_log` is called, the sub-process is stopped.

#### Possible Problem

1. If the system enters sleep mode, it can no longer trace the power.
2. If the recording frequency is too high, it consumes much CPU resource.

## Sleep Mode

### `sleep_with_timer(second: int) -> int`

#### Purpose

When the board enters sleep mode, the CPU and GPU are totally shutdown, leading to a minimum power consumption. However, the system need a outside signal to wake up, which may be not suitable for out application. The only internal signal that can wake the system up is the timer. Thus, when entering sleep mode, a timer should be set accordingly.

#### Implementation

The function sets up an RTC timer and call `systemctl` to suspend the system. It returns 0 on success.

## CPU Power Management

### `get_CPU_info`

#### Purpose

The SoC has 2 Denver cores and 4 A57 cores. Before modify CPU parameters, user should know the current information of CPUs.

#### Implementation

If no argument is given, this function will return a list of structure that contains all the CPU info, including CPU number, current frequency, minimum frequency, maximum frequency, possible minimum frequency, possible maximum frequency, current power state, possible power states, and so on. If arguments are specified, the function will only return corresponding information. All the information is get by reading the virtual fs.

### `set_CPU_state` 

#### Purpose

Change the CPU power state, including sleep state.

#### Implementation

The core number and target power state should be specified. If power state is not allowed, it will report an error. The action is finished by writing to the virtual fs.

### `set_CPU_min_freq` and `set_CPU_max_freq`

#### Purpose

Control the CPU frequency.

#### Implementation

The core number and target frequency should be specified. If frequency is not allowed, it will report an error. The action is finished by writing to the virtual fs.

## GPU Power Management

### `get_GPU_info`

#### Purpose

Before modify GPU parameters, user should know the current information of GPUs.

#### Implementation

If no argument is given, this function will return a structure that contains all the GPU info, including current frequency, minimum frequency, maximum frequency, possible frequencies and so on. If arguments are specified, the function will only return corresponding information. All the information is get by reading the virtual fs.

### `set_GPU_min_freq` and `set_GPU_max_freq`

#### Purpose

Control the GPU frequency.

#### Implementation

The target frequency should be specified. If frequency is not allowed, it will report an error. The action is finished by writing to the virtual fs.

## Peripheral Power Management

This part is more tricky and may be not easy to manage. It should also consume less power than the CPU and GPU. I will work on this after finishing previous parts.