import os
import subprocess


def sleep_with_timer(second: int) -> int:
    """
    This function should be called with super-user privilege.
    Both setting up the timer and suspending the system need
    the previlege to function correctly.

    :param int second: Number of seconds you want to sleep for
    :return: 0 on success
    :rtype: int
    """
    if second < 0:
        return -1
    rtc_path = "/sys/class/rtc/rtc0/wakealarm"
    import time
    curr_t = int(time.time())
    timer = open(rtc_path, "w")
    timer.write(str(curr_t + second))
    timer.close()
    sleep = subprocess.call(["systemctl", "suspend"])
    if sleep != 0:
        return -1
    return 0
