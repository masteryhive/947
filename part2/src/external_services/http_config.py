import httpx
from src.config.logger import Logger

logger = Logger(__name__)

class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ApiClient":
        # create a brand-new client
        self.http_client = httpx.AsyncClient(verify=False)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # close it exactly once
        await self.http_client.aclose()
    
    async def get(self, endpoint: str, params=None, headers=None) -> httpx.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"GET {url}")
        return await self.http_client.get(url, params=params, headers=headers)

    async def post(self, endpoint, payload=None, data=None, files=None, headers=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"Consuming resource from service at {url}")
        if files or data and not payload:
            response = await self.http_client.post(url, files=files, data=data, headers=headers)
        else:
            response = await self.http_client.post(url, json=payload, headers=headers)
        return response

    async def put(self, endpoint, payload=None, headers=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = await self.http_client.put(url, json=payload, headers=headers)
        return response
    
    async def delete(self, endpoint, headers=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = await self.http_client.delete(url, headers=headers)
        return response