from .http import httpstore


class s3store(httpstore):
    """Argo s3 file system

    Inherits from :class:`httpstore` but rely on :class:`s3fs.S3FileSystem` through
    the fsspec 's3' protocol specification.

    By default, this store will use AWS credentials available in the environment.

    If you want to force an anonymous session, you should use the `anon=True` option.

    In order to avoid a *no credentials found error*, you can use:

    >>> from argopy.utils import has_aws_credentials
    >>> fs = s3store(anon=not has_aws_credentials())

    """

    protocol = "s3"
