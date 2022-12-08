from mmengine.config import Config

cfg = Config.fromfile("configs/dataset.yml")
print(type(cfg))  # 是个对象
print(cfg)

# 像使用普通字典或者 Python 类一样来使用 cfg 变量
print(cfg["cifar-10"])

cifar_10 = cfg["cifar-10"]
print(cifar_10.model)

print("-" * 42)
cfg = Config.fromfile("configs/dataset.py")
print(cfg)

# 像使用普通字典或者 Python 类一样来使用 cfg 变量
print(cfg["cifar_10"])
print(cfg.cifar_10)


