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

### `cpu_info` Dictionary

CPU-related information returned by `get_cpu_info` is a list of dictionaries. In each dictionary, there are following fields:

```json
{
  "cpu_idx": 0,
  "cpu_type": "A57",
  "cpu_online": true,
  "min_freq": 345600,
  "max_freq": 2035200,
  "cur_freq": 345000,
  "available_freq": [345600, 499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200],
  "cur_gov": "schedutil", 
  "available_gov": ["interactive", "conservative", "ondemand", "userspace", "powersave", "performance", "schedutil"]
}
```

There are two types of CPU, "A57" and "Denver". It's possible that CPU current frequency is different from available ones, because the available frequencies are used for scaling. If a CPU is not online, then current frequency will be zero and the available frequencies will be an empty list.

### `get_cpu_info(cpu_idx=None, cpu_type=True, cpu_online=True, min_freq=True, max_freq=True, cur_freq=True, available_freq=True, cur_gov=True, available_gov=True) -> List[Dict]`

#### Purpose

The SoC has 2 Denver cores and 4 A57 cores. Before modify CPU parameters, user should know the current information of CPUs.

#### Implementation

With no arguments, this function reads a series of virtual files and return a complete list of all possible info as show in the dictionary above. If it receives arguments, it will only return corresponding values. CPU index other than 0 to 5 will be ignored.

### `cpu_target` Dictionary

CPU-related parameters taken in by `set_cpu_state` is a list of dictionaries. In each dictionary, there can be following fields:

```json
{
  "cpu_idx": 0,
  "cpu_online": true,
  "min_freq": 345600,
  "max_freq": 2035200,
  "governor": "schdeutil"
}
```

### `set_cpu_state(cpu_targets: List[Dict]) -> int` 

#### Purpose

Change CPU power state, frequencies, and governor.

#### Implementation

This function takes in a list of dict and sets the cpu parameters accordingly. For each dict, "cpu_idx" is a necessary field, and the possible three options are "cpu_online", "min_freq", and "max_freq".

If the "cpu_online" field is False, or "cpu_online" is not specified and cpu is offline, the frequency fields and governor field will be ignored.

There are many possible errors and here's a list error code:

* -1: "cpu_idx" field is missing. 
* -2: The last online cpu is being offline.
* -3: The specified minimum frequency is not in available frequency list.
* -4: The specified maximum frequency is not in available frequency list.
* -5: Only minimum frequency is specified, and it is greater than current maximum frequency.
* -6: Only maximum frequency is specified, and it is smaller than current minimum frequency.
* -7: The specified maximum frequency is smaller than the specified minimum frequency.
* -8: The specified governor is not in available governor list.

If multiple dicts in the list have errors, the error code will be concatenated together. For example, if the first dict doesn't specify "cpu_idx", and the third dict gives a wrong minimum frequency, the return value will be -103. If only the third dict gives a wrong minimum frequency, the return value will be -3.

## GPU Power Management

### `gpu_info` Dictionary

GPU-related information returned by `get_gpu_info` is a dictionary. In the dictionary, there are following fields:

```json
{
  "min_freq": 114750000,
  "max_freq": 1122000000,
  "cur_freq": 114750000,
  "available_freq": [114750000, 216750000, 318750000, 420750000, 522750000, 624750000, 726750000, 854250000, 930750000, 1032750000, 1122000000, 1236750000, 1300500000],
  "cur_gov": "nvhost_podgov",
  "available_gov": ["wmark_active", "wmark_simple", "nvhost_podgov", "userspace", "performance", "simple_ondemand"]
}
```

### `get_gpu_info(min_freq=True, max_freq=True, cur_freq=True, available_freq=True, cur_gov=True, available_gov=True) -> Dict`

#### Purpose

Before modify GPU parameters, user should know the current information of GPUs.

#### Implementation

With no arguments, this function reads a series of virtual files and return all possible info. If it receives arguments, it will only return corresponding values.

### `gpu_target` Dictionary

GPU-related parameters taken in by `set_gpu_state` is a dictionary. In the dictionary, there can be following fields:

```json
{
  "min_freq": 114750000,
  "max_freq": 1122000000,
  "governor": "nvhost_podgov"
}
```

### `set_gpu_state(gpu_target: Dict) -> int`

#### Purpose

Change GPU frequencies and governor.

#### Implementation

**Note: Governor cannot be set currently. There may be some errors, and the code is commented.**

This function takes in a dict and sets the gpu parameters accordingly. The possible three options are "min_freq", "max_freq", and "governor".

There are many possible errors and here's a list error code:

* -1: The specified minimum frequency is not in available frequency list.
* -2: The specified maximum frequency is not in available frequency list.
* -3: Only minimum frequency is specified, and it is greater than current maximum frequency.
* -4: Only maximum frequency is specified, and it is smaller than current minimum frequency.
* -5: The specified maximum frequency is smaller than the specified minimum frequency.
* -6: The specified governor is not in available governor list.

## Peripheral Power Management

This part is more tricky and may be not easy to manage. It should also consume less power than the CPU and GPU. We may implement this part when we are available.