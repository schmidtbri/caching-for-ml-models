from typing import Optional, List
import msgpack
import snappy
import json
import redis
from ml_base.decorator import MLModelDecorator


class RedisCachingDecorator(MLModelDecorator):
    """Decorator for caching around an MLModel instance."""

    def __init__(self, host: str, port: int, database: int, prefix: Optional[str] = None,
                 hashing_fields: Optional[List[str]] = None, serder: str = "JSON",
                 use_compression: bool = False,
                 reduced_precision: bool = False,
                 number_of_places: Optional[int] = None
                 ) -> None:

        if serder not in ["JSON", "MessagePack"]:
            raise ValueError("Serder option not supported.")

        if reduced_precision is True and number_of_places is None:
            raise ValueError("number_of_places must be provided when reduced_precision is True.")

        if number_of_places is None and reduced_precision is True:
            print(number_of_places, reduced_precision)
            raise ValueError("reduced_precision must be True when number_of_places is provided.")

        super().__init__(host=host, port=port, database=database, prefix=prefix,
                         hashing_fields=hashing_fields, serder=serder,
                         use_compression=use_compression,
                         reduced_precision=reduced_precision,
                         number_of_places=number_of_places)

        self.__dict__["_redis_client"] = redis.Redis(host=host, port=port, db=database)

    def predict(self, data):
        if self._configuration["prefix"] is not None:
            prefix = "{}/{}/{}/".format(self._configuration["prefix"],
                                        self._model.qualified_name,
                                        self._model.version)
        else:
            prefix = "{}/{}/".format(self._model.qualified_name,
                                     self._model.version)

        # reducing the precision of the numerical fields, if it is enabled
        if self._configuration["reduced_precision"] is True:
            for field_name, field_attributes in self._model.input_schema.schema()["properties"].items():
                if "type" in field_attributes.keys() and field_attributes["type"] == "number":
                    field_value = getattr(data, field_name)
                    setattr(data, field_name, round(field_value, self._configuration["number_of_places"]))

        # select hashing fields from input
        if self._configuration["hashing_fields"] is not None:
            data_dict = {key: data.dict()[key] for key in self._configuration["hashing_fields"]}
        else:
            data_dict = data.dict()

        # creating a key for the prediction inputs provided
        frozen_data = frozenset(data_dict.keys()), frozenset(data_dict.values())
        key = prefix + str(hash(frozen_data))

        # check if the prediction is in the cache
        prediction = self.__dict__["_redis_client"].get(key)

        # if the prediction is present in the cache
        if prediction is not None:

            # optionally decompressing the bytes
            if self._configuration["use_compression"]:
                decompressed_prediction = snappy.decompress(prediction)
            else:
                decompressed_prediction = prediction

            # deserializing to bytes
            if self._configuration["serder"] == "JSON":
                deserialized_prediction = json.loads(decompressed_prediction.decode())
            elif self._configuration["serder"] == "MessagePack":
                deserialized_prediction = msgpack.loads(decompressed_prediction)
            else:
                raise ValueError("Serder option not supported.")

            # creating the output instance
            prediction = self._model.output_schema(**deserialized_prediction)

            return prediction

        # if the prediction is not present in the cache
        else:
            # making a prediction with the model
            prediction = self._model.predict(data)

            # serializing to bytes
            if self._configuration["serder"] == "JSON":
                serialized_prediction = str.encode(json.dumps(prediction.dict()))
            elif self._configuration["serder"] == "MessagePack":
                serialized_prediction = msgpack.dumps(prediction.dict())
            else:
                raise ValueError("Serder option not supported.")

            # optionally compressing the bytes
            if self._configuration["use_compression"]:
                serialized_prediction = snappy.compress(serialized_prediction)

            # saving the prediction to the cache
            self.__dict__["_redis_client"].set(key, serialized_prediction)

            return prediction
