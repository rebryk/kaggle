import resource
import uuid


def get_uuid() -> str:
    """
    Generate UUID.

    :return: generated UUID
    """
    return str(uuid.uuid4())


def remove_resource_limits():
    """Remove the limit of maximum number of open file descriptors for the current process."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, hard))
