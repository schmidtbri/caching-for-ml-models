from typing import Optional
from pydantic import create_model
from uuid import uuid4
from ml_base import MLModelDecorator


class PredictionIDDecorator(MLModelDecorator):

    @property
    def description(self) -> str:
        decorator_description = " This model also has an optional input called 'prediction_id' that accepts an UUID " \
                                "string to uniquely identify the prediction returned. If the prediction id is not " \
                                "provided, a UUID is generated and returned in a field called 'prediction_id' in the " \
                                "model output."
        return self._model.description + decorator_description

    @property
    def input_schema(self):
        input_schema = self._model.input_schema
        new_input_schema = create_model(
            input_schema.__name__,
            prediction_id=(Optional[str], None),
            __base__=input_schema,
        )
        return new_input_schema

    @property
    def output_schema(self):
        output_schema = self._model.output_schema
        new_output_schema = create_model(
            output_schema.__name__,
            prediction_id=(str, ...),
            __base__=output_schema,
        )
        return new_output_schema

    def predict(self, data):
        if hasattr(data, "prediction_id") and data.prediction_id is not None:
            prediction_id = data.prediction_id
        else:
            prediction_id = str(uuid4())

        prediction = self._model.predict(data=data)
        wrapped_prediction = self.output_schema(prediction_id=prediction_id, **prediction.dict())
        return wrapped_prediction
