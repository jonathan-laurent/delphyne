# TODO

- In case of error 422 when Pydantic does not properly decode the request on the FastAPI side, we end up with an `end-of-stream` error. We should be able to detect the 422 status instead of trying to decipher the malformed stream.