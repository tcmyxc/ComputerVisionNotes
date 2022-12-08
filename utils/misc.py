"""
散乱的工具类
"""

import datetime


def print_args(args):
    """优雅地打印命令行参数"""

    print("")
    print("-" * 20, "args", "-" * 20)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 18, "args end", "-" * 18, flush=True)


def get_current_time():
    '''get current time'''
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}-{utc_plus_8_time.minute:0>2d}-{utc_plus_8_time.second:0>2d}"
    return f"{ymd}_{hms}"


def print_time(time_elapsed, epoch=False):
    """打印程序执行时长"""
    time_hour = time_elapsed // 3600
    time_minite = (time_elapsed % 3600) // 60
    time_second = time_elapsed % 60
    if epoch:
        print(f"\nCurrent epoch take time: {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")
    else:
        print(f"\nAll complete in {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")


def print_yml_cfg(cfg):
    """打印从yml文件加载的配置"""

    print("")
    print("-" * 20, "yml cfg", "-" * 20)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 18, "yml cfg end", "-" * 18, flush=True)
