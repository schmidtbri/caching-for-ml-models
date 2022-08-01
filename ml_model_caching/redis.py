import os
from typing import List, Optional
from ml_base.decorator import MLModelDecorator


class RedisCachingDecorator(MLModelDecorator):
    """Decorator for caching around an MLModel instance."""

    def __init__(self, host: str, port: str, username: str, password: str, database: str,
                 prefix: str, hashing_fields: Optional[List[str]] = None) -> None:
        # if password has ${}, then replace with environment variable
        if password[0:2] == "${" and password[-1] == "}":
            password = os.environ[password[2:-1]]
        super().__init__(host=host, port=port, username=username, password=password,
                         database=database, prefix=prefix, hashing_fields=hashing_fields)

    def predict(self, data):
        pass

# Adding a hit rate measurement through the log …
# Add a serder time measurement through the log …
# Adding access time measurement through the log …
