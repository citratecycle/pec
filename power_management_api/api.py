import os
import time
from multiprocessing import Process, Value
import subprocess

_local_dict = {}


def _trace_power_log(period: float, path: str, stop: Value) -> int:
    """
    This is a private function and shouldn't be called from outside.

    :param float period: period to sleep
    :param str path: path to log output file
    :param Value stop: value shared by parent and child processes
    :return: 0 on success, negative numbers on errors
    """
    power_trace_path = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
    log = open(path, "a")
    while stop.value != 1:
        power_trace = open(power_trace_path, "r")
        power = power_trace.read()
        power_trace.close()
        log.write("ms: {0}, power: {1}".format(int(time.time() * 1000), power))
        time.sleep(period)
    log.close()
    return 0


def start_power_log(period: float, path: str) -> int:
    """
    This function should be called with super-user privilege.
    This function **shouldn't** be called when a recording has already been started.

    It starts a child process recording input power to the SoC in specified period and
    outputting the result to a specified file.

    :param float period: period to sleep
    :param str path: **absolute** path to log output file
    :return: 0 on success, negative numbers on errors
    """
    if "power_log_subprocess" in _local_dict.keys():
        return -1
    stop = Value("i", 0)
    _local_dict["power_log_stop"] = stop
    p = Process(target=_trace_power_log, args=(period, path, stop))
    _local_dict["power_log_subprocess"] = p
    p.start()
    return 0


def stop_power_log() -> int:
    """
    This function **shouldn't** be called when there isn't a recording.
    Warning: Only call it when time is not critical, since it is blocking
    and will wait the subprocess to finish execution, which at most costs
    ``period`` of time.

    It stops the child process recording power.

    :return: 0 on success, negative numbers on errors
    """
    if "power_log_subprocess" not in _local_dict.keys():
        return -1
    _local_dict["power_log_stop"].value = 1
    _local_dict["power_log_subprocess"].join()
    _local_dict.pop("power_log_stop")
    _local_dict.pop("power_log_subprocess")
    return 0


def sleep_with_timer(second: int) -> int:
    """
    This function should be called with super-user privilege.
    Both setting up the timer and suspending the system need
    the privilege to function correctly.

    :param int second: Number of seconds you want to sleep for
    :return: 0 on success
    :rtype: int
    """
    if second < 0:
        return -1
    rtc_path = "/sys/class/rtc/rtc0/wakealarm"
    curr_t = int(time.time())
    timer = open(rtc_path, "w")
    timer.write(str(curr_t + second))
    timer.close()
    sleep = subprocess.call(["systemctl", "suspend"])
    if sleep != 0:
        return -1
    return 0
