"""自定义网络模块插件包。

ultralytics 的 parse_model 通过 ``globals()[模块名]`` 解析模型 yaml 里的层类型，
因此自定义模块必须在构建模型（以及反序列化含该模块的权重）之前注入其命名空间。

已实现模块：
    CBAM      —— 通道 + 空间注意力（Woo et al., ECCV 2018）
    CoordAtt  —— 坐标注意力（Hou et al., CVPR 2021）
"""

from . import cbam  # noqa: F401
from . import coordatt  # noqa: F401

_MODULES = {
    "CBAM": cbam.CBAM,
    "CoordAtt": coordatt.CoordAtt,
}

_REGISTERED: set[str] = set()


def register_attention(verbose: bool = True) -> bool:
    """把所有自定义模块注册进 ultralytics 的模块命名空间。

    可重复调用（幂等）；只有首次注册时打印提示，避免训练日志被刷屏。
    """
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    newly = []
    for name, cls in _MODULES.items():
        setattr(tasks, name, cls)
        setattr(modules, name, cls)
        if name not in _REGISTERED:
            _REGISTERED.add(name)
            newly.append(name)

    if newly and verbose:
        print(f"[plugins] 已注册自定义模块: {', '.join(newly)}")
    return True


__all__ = ["register_attention", "cbam", "coordatt"]
