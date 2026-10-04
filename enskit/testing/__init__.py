r"""Conformance checks for code written against EnsKit's layers.

Call a check on a piece you wrote, an update rule or a conditional map, to
verify it against the contract of the layer it plugs into. Each check builds
its own small fixture, raises ``AssertionError`` naming the obligation that
failed, and returns ``None`` when every obligation holds.

================================ ==============================================
function                         checks
================================ ==============================================
:func:`check_update_rule`        a :class:`~enskit.kalman.UpdateRule`: result
                                 structure, dtype, value-free builds, keys,
                                 exactness on a linear-Gaussian problem,
                                 ``jit`` and ``vmap``
:func:`check_conditional_map`    a :class:`~enskit.distribution.ConditionalMap`:
                                 values, result structure, pointwise action,
                                 keys, agreement with
                                 :meth:`~enskit.distribution.Gaussian.condition`
================================ ==============================================

Notes
-----
This module may import every layer and is imported by none, so a check can
never become something the layers depend on. The obligations are those of the
"Kalman contract" and "Distribution contract" pages of the documentation.
"""

from ._conditional_map import check_conditional_map
from ._update_rule import check_update_rule

__all__ = ["check_update_rule", "check_conditional_map"]

for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
