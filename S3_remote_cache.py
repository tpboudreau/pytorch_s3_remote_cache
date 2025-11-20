
import os
import logging
from typing import Optional
from typing_extensions import override
from torch._inductor.remote_cache import RemoteCache, RemoteCacheBackend, RemoteCacheJsonSerde, JsonDataTy

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class S3RemoteCacheBackend(RemoteCacheBackend[bytes]):
    """
    An S3 implementation of a remote/distributed cache.
    """

    _s3_client = None
    _s3_region: str = None
    _s3_bucket: str = None

    def __init__(self, cache_id: str) -> None:
        super().__init__()
        if not boto3:
            raise RuntimeError("boto3 not available but required for remote cache")

        self._s3_region = os.environ.get("TORCHINDUCTOR_REMOTE_CACHE_S3_REGION")
        if self._s3_region is None:
            log.warning("unable to determine region required for S3 remote cache; disabling remote cache")
            return

        self._s3_bucket = os.environ.get("TORCHINDUCTOR_REMOTE_CACHE_S3_BUCKET")
        if self._s3_bucket is None:
            log.warning("unable to determine bucket required for S3 remote cache; disabling remote cache")
            return

        self._s3_client = boto3.client('s3', region_name=self._s3_region)
        try:
            bucket_config = {}
            if self._s3_region != 'us-east-1':
                bucket_config['CreateBucketConfiguration'] = {'LocationConstraint': self._s3_region}
            self._s3_client.create_bucket(Bucket=self._s3_bucket, **bucket_config)
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == "BucketAlreadyOwnedByYou":
                pass
            else:
                self._s3_client = None

    @override
    def _get(self, key: str) -> Optional[bytes]:
        if not self._s3_client:
            return None

        data = None
        try:
            response = self._s3_client.get_object(Bucket=self._s3_bucket, Key=key)
            stream = response.get("Body")
            data = stream.read()

        except ClientError as e:
            code = e.response['Error']['Code']
            if code == "NoSuchKey":
                return None
            self._s3_client = None

        assert data is None or isinstance(data, bytes)
        return data

    @override
    def _put(self, key: str, data: bytes) -> None:
        if not self._s3_client:
            return

        try:
            response = self._s3_client.put_object(Body=data, Bucket=self._s3_bucket, Key=key, IfNoneMatch='*')
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == "PreconditionFailed":
                return
            self._s3_client = None

        return


class S3RemoteCache(RemoteCache[JsonDataTy]):
    def __init__(self, cache_id: str) -> None:
        backend = S3RemoteCacheBackend(cache_id)
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)
        version = 1
        self._key_fmt = f"pt2:{cache_id}::{{key}}:c{version}"

    def _get_key(self, key: str) -> str:
        return self._key_fmt.format(key=key)

    @override
    def _get(self, key: str, sample: Optional[type[object]]) -> Optional[bytes]:
        key = self._get_key(key)
        return super()._get(key, sample)

    @override
    def _put(self, key: str, value: bytes, sample: Optional[type[object]]) -> None:
        key = self._get_key(key)
        super()._put(key, value, sample)

