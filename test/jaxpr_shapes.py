"""Shape census of one traced jaxpr, for the streamed-layout regressions.

The walk is iterative and id-deduped. jax 0.12 nests jaxprs inside eqn params
deep enough that the earlier recursive version raised RecursionError, and the
same eqn param can repeat a jaxpr, so plain recursion could also crawl the same
subgraph twice.
"""


def traced_shapes(traced):
    """Every outvar shape appearing in the jaxpr tree of `traced`."""
    shapes = []
    stack = [traced]
    seen = set()
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if hasattr(value, "jaxpr"):
            stack.append(value.jaxpr)
        elif hasattr(value, "eqns"):
            for eqn in value.eqns:
                for var in eqn.outvars:
                    aval = getattr(var, "aval", None)
                    if aval is not None and hasattr(aval, "shape"):
                        shapes.append(tuple(aval.shape))
                stack.extend(eqn.params.values())
        elif isinstance(value, (tuple, list)):
            stack.extend(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
    return shapes
