from __future__ import annotations


def log_implicit_gpu_status(state) -> None:
    try:
        import implicit
    except ModuleNotFoundError:
        return

    has_cuda = bool(getattr(getattr(implicit, "gpu", None), "HAS_CUDA", False))
    state.logger.info(f"implicit GPU support available: {has_cuda}")

    for rec in state.recommenders_base.items():
        scorer = getattr(rec, "scorer", None)
        if scorer is None or not hasattr(scorer, "_construct"):
            continue
        if not scorer.__class__.__module__.startswith("lenskit.implicit"):
            continue

        extra = getattr(getattr(rec, "lk_config", None), "__pydantic_extra__", None) or {}
        explicit_use_gpu = "use_gpu" in extra
        use_gpu = extra.get("use_gpu")
        if use_gpu is None:
            use_gpu = has_cuda

        if explicit_use_gpu and use_gpu and not has_cuda:
            state.logger.warning(
                f"Implicit GPU requested for {rec.name} but implicit reports no CUDA support"
            )

        try:
            delegate = scorer._construct()
            backend = delegate.__class__.__module__
        except Exception as exc:
            state.logger.warning(f"Implicit GPU check failed for {rec.name}: {exc}")
            continue

        state.logger.info(
            f"Implicit backend for {rec.name}: use_gpu={use_gpu}, backend={backend}"
        )
        if use_gpu and not backend.startswith("implicit.gpu"):
            state.logger.warning(
                f"Implicit GPU requested for {rec.name} but backend is {backend}"
            )
