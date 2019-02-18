import logging

from dask import delayed  # noqa: F401

log = logging.getLogger('owl.daemon.pipeline')
log = logging.LoggerAdapter(log, {'component': 'DASK PIPELINE'})


def main(*, datalen: int):
    pass
