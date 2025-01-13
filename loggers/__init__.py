
def str_to_cls(logger_str):
    if logger_str == 'WandbLogger':
        import loggers.WandBLogger as WandBLogger
        return WandBLogger
    elif logger_str == 'StdLogger':
        import loggers.StdLogger as StdLogger
        return StdLogger
    elif logger_str == 'TensorboardLogger':
        import loggers.TensorboardLogger as TensorboardLogger
        return TensorboardLogger