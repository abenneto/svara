"""包级别的基本检查。"""

import svara


def test_version_is_exposed() -> None:
    assert isinstance(svara.__version__, str)
    assert svara.__version__.count(".") >= 2
