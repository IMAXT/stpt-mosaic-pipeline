import logging
import time
from functools import wraps

log = logging.getLogger('owl.daemon.pipeline')


def retry(exceptions, tries=4, delay=3, backoff=2):
    """Retry calling the decorated function using an exponential backoff.

    Parameters
    ----------
    exceptions
        The exception to check, May be a tuple of exceptions.
    tries
        Number of times to try (not retry) before giving up.
    delay
        Initial delay between retries in seconds.
    backoff
        Backoff multiplier (e.g. value of 2 will double the delay
        each retry).
    """

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = '{}, Retrying in {} seconds...'.format(e, mdelay)
                    log.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry
