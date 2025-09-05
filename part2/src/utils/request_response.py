from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class ApiResponse(JSONResponse):
    def __init__(self, message: str, data: dict = None, code: int = 200, **kwargs):
        content = {"code": code, "message": message, "data": data}
        content = jsonable_encoder(content)
        super().__init__(content=content, **kwargs)