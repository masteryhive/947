import asyncio
import asyncpg
from typing import List, Dict, Any, Union, Optional
from tqdm import tqdm
import numpy as np
import uuid
from datetime import datetime as dt
from src.schemas.search_schema import MatchAnyOrInterval
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed
import json
from src.config.settings import settings
from src.schemas.search_schema import MatchAnyOrInterval
from src.schemas.document_schemas import Document, RAGPayload
from src.config.config_helper import Configuration
from src.config.logger import Logger

logger = Logger(__name__)

config = Configuration().get_config('vdb')

class PostgresVectorDB:
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        collection_name: str = settings.COLLECTION,
        max_attempts: int = config['max_attempts'],
        wait_time_seconds: int = config['wait_time_seconds'],
        batch_size: int = settings.BATCH_SIZE,
        content_payload_key: str = config['content_payload_key'],
        metadata_payload_key: str = config['metadata_payload_key'],
        score_key: str = config['score_key'],
        limit: int = int(settings.DEFAULT_QUERY_LIMIT),
    ):
        self.__model_name = model_name
        self.__collection_name = collection_name
        self.__sentence_model = SentenceTransformer(self.__model_name, device=config["device"], trust_remote_code=True)
        self.limit = limit
        self.__batch_size = batch_size
        self.__max_attempts = max_attempts
        self.__wait_time_seconds = wait_time_seconds
        self.__content_payload_key = content_payload_key
        self.__metadata_payload_key = metadata_payload_key
        self.__score_key = score_key
        
        # Database connection parameters
        self.db_config = {
            'host': settings.POSTGRES_HOST,
            'port': int(settings.POSTGRES_PORT),
            'database': settings.POSTGRES_DB,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD,
        }
        
        self.pool = None

    async def _get_connection_pool(self):
        """Get or create connection pool"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
        return self.pool

    def encode(self, docs: List[str]) -> np.ndarray:
        """
        Encode a list of documents in batches using the SentenceTransformer model.
        """
        @retry(stop=stop_after_attempt(self.__max_attempts), wait=wait_fixed(self.__wait_time_seconds))
        def encode_batch(batch_docs: List[str]) -> np.ndarray:
            try:
                return self.__sentence_model.encode([doc.page_content for doc in batch_docs])
            except Exception:
                return self.__sentence_model.encode(batch_docs)
        
        embeddings = []
        try:
            for i in tqdm(range(0, len(docs), self.__batch_size)):
                batch_docs = docs[i:i+self.__batch_size]
                batch_embeddings = encode_batch(batch_docs)
                embeddings.append(batch_embeddings)

            if embeddings:
                embeddings = np.concatenate(embeddings)
            else:
                raise ValueError("No embeddings were generated.")

            if self.__sentence_model.get_sentence_embedding_dimension() == embeddings.shape[1]:
                return embeddings
            else:
                raise logger.error(f"The embeddings have an incorrect dimension of {embeddings.shape[1]}.")
        except Exception as ex:
            raise logger.error(f"Attempt failed. Retrying Batch... Error: {str(ex)}")

    def generate_points(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate a list of points by encoding the documents and combining embeddings with metadata.
        """
        # Extract page_content for encoding
        content_list = [doc.get('page_content', '') for doc in docs]
        embeddings = self.encode(content_list)
        logger.info("Embedding Completed")

        # Combine the embeddings with the metadata
        points_list = [
            {
                "id": doc.get('metadata')["id"],
                "embedding": content_embedding.tolist(),
                "page_content": doc.get('page_content'),
                "document_metadata": doc.get('metadata'),
                "document_type": doc.get('metadata', {}).get('document_type', ''),
                "user_id": doc.get('metadata', {}).get('user_id', '')
            }
            for (doc, content_embedding) in zip(docs, embeddings)
        ]

        logger.info("Generating points")
        return points_list

    async def get_or_create_collection(self):
        """Create the vector table if it doesn't exist"""
        pool = await self._get_connection_pool()
        
        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table with vector column
            vector_dim = self.__sentence_model.get_sentence_embedding_dimension()
            
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.__collection_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                embedding vector({vector_dim}),
                page_content TEXT,
                document_metadata JSONB,
                document_type VARCHAR(50),
                user_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await conn.execute(create_table_query)
            
            # Create indexes for better performance
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.__collection_name}_embedding
                ON {self.__collection_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.__collection_name}_document_metadata
                ON {self.__collection_name} USING gin (document_metadata);
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.__collection_name}_user_id
                ON {self.__collection_name} USING btree (user_id);
            """)

            
            logger.info(f"Collection '{self.__collection_name}' ready for use.")

    async def exists(self) -> bool:
        """Check if the collection (table) exists"""
        pool = await self._get_connection_pool()
        
        async with pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                );
            """, self.__collection_name)
            return result

    async def delete_collection(self):
        """Drop the collection table"""
        pool = await self._get_connection_pool()
        
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self.__collection_name}")
            logger.info(f"Deleted '{self.__collection_name}' table.")

    async def delete_one(self, ids: List[str]):
        """Delete one or more items from the vector database"""
        if not isinstance(ids, list):
            ids = [ids]
            logger.warning(f"IDs '{ids}' must be in a list to be able to delete them.")
        
        pool = await self._get_connection_pool()
        
        async with pool.acquire() as conn:
            # Convert string IDs to UUIDs if necessary
            uuid_ids = []
            for id_val in ids:
                if isinstance(id_val, str):
                    uuid_ids.append(uuid.UUID(id_val))
                else:
                    uuid_ids.append(id_val)
            
            await conn.execute(
                f"DELETE FROM {self.__collection_name} WHERE id = ANY($1)",
                uuid_ids
            )
            logger.info(f"Deleted '{ids}' from '{self.__collection_name}' table.")

    async def batch_update(self, update_operations: List[Dict[str, Any]]) -> None:
        """
        Perform batch updates on the collection.
        """
        pool = await self._get_connection_pool()

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for operation in update_operations:
                        if operation['type'] == 'update_vectors':
                            # Update vectors for specific points
                            for point_id, vector in operation['data']:
                                await conn.execute(
                                    f"UPDATE {self.__collection_name} SET embedding = $1::vector, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                                    str(vector), uuid.UUID(str(point_id))
                                )
                        elif operation['type'] == 'set_payload':
                            # Update metadata for specific points
                            for point_id in operation['points']:
                                await conn.execute(
                                    f"UPDATE {self.__collection_name} SET document_metadata = document_metadata || $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                                    json.dumps(operation['payload']), uuid.UUID(str(point_id))
                                )
                        else:
                            logger.warning(f"Unsupported operation type: {operation['type']}")

            logger.info("Batch update completed successfully.")

        except Exception as ex:
            logger.error(f"Batch update failed due to error: {str(ex)}")
            raise

    async def upsert_points(self, points_list: List[Dict[str, Any]], is_batch: bool) -> None:
        """
        Upsert a list of points into the PostgreSQL table.
        """
        @retry(stop=stop_after_attempt(self.__max_attempts), wait=wait_fixed(self.__wait_time_seconds))
        async def upsert_batch(batch_data: List[Dict[str, Any]]) -> None:
            pool = await self._get_connection_pool()

            async with pool.acquire() as conn:
                async with conn.transaction():
                    upsert_query = f"""
                    INSERT INTO {self.__collection_name} (
                        id, embedding, page_content, document_metadata,
                        document_type, user_id, created_at, updated_at
                    )
                    VALUES ($1, $2::vector, $3, $4, $5, $6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        page_content = EXCLUDED.page_content,
                        document_metadata = EXCLUDED.document_metadata,
                        document_type = EXCLUDED.document_type,
                        user_id = EXCLUDED.user_id,
                        updated_at = CURRENT_TIMESTAMP
                    """

                    for point in batch_data:
                        embedding = point.get('embedding', point.get('vector'))
                        document_metadata = point.get('document_metadata', point.get('metadata', {}))
                        document_type = document_metadata.get('document_type', point.get('document_type', ''))
                        user_id = document_metadata.get('user_id', point.get('user_id', ''))

                        # Convert embedding to string format for PostgreSQL vector type
                        if isinstance(embedding, list):
                            embedding_str = str(embedding)
                        else:
                            embedding_str = str(embedding.tolist())

                        await conn.execute(
                            upsert_query,
                            uuid.UUID(str(point['id'])),
                            embedding_str,
                            point.get('page_content', point.get('content', '')),
                            json.dumps(document_metadata),
                            document_type,
                            user_id
                        )

        @retry(stop=stop_after_attempt(self.__max_attempts), wait=wait_fixed(self.__wait_time_seconds))
        async def upsert_single(points_list: List[Dict[str, Any]]) -> None:
            pool = await self._get_connection_pool()

            async with pool.acquire() as conn:
                async with conn.transaction():
                    upsert_query = f"""
                    INSERT INTO {self.__collection_name} (
                        id, embedding, page_content, document_metadata,
                        document_type, user_id, created_at, updated_at
                    )
                    VALUES ($1, $2::vector, $3, $4, $5, $6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        page_content = EXCLUDED.page_content,
                        document_metadata = EXCLUDED.document_metadata,
                        document_type = EXCLUDED.document_type,
                        user_id = EXCLUDED.user_id,
                        updated_at = CURRENT_TIMESTAMP
                    """

                    for point in points_list:
                        embedding = point.get('embedding', point.get('vector'))
                        document_metadata = point.get('document_metadata', point.get('metadata', {}))
                        document_type = document_metadata.get('document_type', point.get('document_type', ''))
                        user_id = document_metadata.get('user_id', point.get('user_id', ''))

                        # Convert embedding to string format for PostgreSQL vector type
                        if isinstance(embedding, list):
                            embedding_str = str(embedding)
                        else:
                            embedding_str = str(embedding.tolist())

                        await conn.execute(
                            upsert_query,
                            uuid.UUID(str(point['id'])),
                            embedding_str,
                            point.get('page_content', point.get('content', '')),
                            json.dumps(document_metadata),
                            document_type,
                            user_id
                        )
        
        if is_batch:
            for i in tqdm(range(0, len(points_list), self.__batch_size), desc="Upserting batches"):
                batch_data = points_list[i:i+self.__batch_size]
                await upsert_batch(batch_data)
            logger.info("Bulk records inserted successfully.")
        else:
            await upsert_single(points_list)
            logger.info("Records inserted successfully.")

    async def get(self, ids: Optional[Union[str, uuid.UUID, List[Union[str, uuid.UUID]]]] = None, retries: int = 3, delay: int = 2) -> Optional[List[dict]]:
        """Retrieve documents by IDs"""
        if ids is None:
            logger.warning("No ID provided. Nothing to retrieve.")
            return None
        if not isinstance(ids, list):
            ids = [ids]
        # Convert to UUIDs
        uuid_ids = []
        for id_val in ids:
            if isinstance(id_val, str):
                uuid_ids.append(uuid.UUID(id_val))
            else:
                uuid_ids.append(id_val)
        for attempt in range(retries):
            try:
                pool = await self._get_connection_pool()
                async with pool.acquire() as conn:
                    query = f"""
                    SELECT id, page_content, document_metadata, document_type, user_id
                    FROM {self.__collection_name}
                    WHERE id = ANY($1)
                    """

                    rows = await conn.fetch(query, uuid_ids)

                    results = []
                    for row in rows:
                        metadata = json.loads(row['document_metadata'])
                        # Add document_type and user_id to metadata if they're not already there
                        if 'document_type' not in metadata:
                            metadata['document_type'] = row['document_type']
                        if 'user_id' not in metadata:
                            metadata['user_id'] = row['user_id']

                        results.append({
                            'id': str(row['id']),
                            'payload': {
                                self.__content_payload_key: row['page_content'],
                                self.__metadata_payload_key: metadata
                            }
                        })

                    return results

            except Exception as ex:
                logger.error(f"Error retrieving data on attempt {attempt + 1}: {ex}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries exceeded. Failed to retrieve data.")
                    return None

    def refine(self, filters: Dict[str, MatchAnyOrInterval] = None) -> str:
        """
        Build WHERE clause for PostgreSQL based on filters.
        """
        if filters is None:
            return ""

        conditions = []

        for field, value in filters.items():
            if value.any is not None:
                # Handle array matching - PostgreSQL JSONB array contains
                placeholders = ', '.join([f"'{val}'" for val in value.any])
                conditions.append(f"(document_metadata->>'{field}')::text = ANY(ARRAY[{placeholders}])")

            elif any([value.gt, value.gte, value.lt, value.lte]):
                # Handle range conditions for dates
                field_path = f"(document_metadata->>'{field}')::timestamp"

                if value.gt:
                    conditions.append(f"{field_path} > '{value.gt}'")
                if value.gte:
                    conditions.append(f"{field_path} >= '{value.gte}'")
                if value.lt:
                    conditions.append(f"{field_path} < '{value.lt}'")
                if value.lte:
                    conditions.append(f"{field_path} <= '{value.lte}'")

        if conditions:
            return " AND " + " AND ".join(conditions)
        return ""

    async def search(self, query: str, limit: Optional[int] = None, filters: Dict[str, MatchAnyOrInterval] = None) -> List[Document]:
        """
        Perform vector similarity search using PostgreSQL + pgvector.
        """
        if limit is None:
            limit = self.limit

        # Encode the query
        query_vector = self.__sentence_model.encode(query)

        # Build filter conditions
        filter_clause = self.refine(filters)

        logger.info(f"Query: {query}")
        logger.info(f"Filters: {filter_clause}")
        try:
            pool = await self._get_connection_pool()
            async with pool.acquire() as conn:
                # PostgreSQL query with cosine similarity
                search_query = f"""
                SELECT
                    id,
                    page_content,
                    document_metadata,
                    document_type,
                    user_id,
                    1 - (embedding <=> $1::vector) as similarity_score
                FROM {self.__collection_name}
                WHERE 1=1 {filter_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """

                rows = await conn.fetch(search_query, str(query_vector.tolist()), limit)

                # Convert results to Document objects
                results = [
                    Document(
                        page_content=row['page_content'],
                        metadata={
                            **json.loads(row['document_metadata']),
                            'document_type': row['document_type'],
                            'user_id': row['user_id']
                        },
                        score={self.__score_key: float(row['similarity_score'])},
                    )
                    for row in rows
                ]

                return results

        except Exception as ex:
            logger.error(f"Search failed: {str(ex)}")
            # Try to create collection if it doesn't exist
            await self.get_or_create_collection()
            return []

    async def insert(self, docs: List[Dict[str, Any]], is_batch: bool = False):
        """Insert documents into the PostgreSQL vector table"""
        await self.get_or_create_collection()
        points_list = self.generate_points(docs)
        await self.upsert_points(points_list, is_batch)

    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed.")

    async def get_collections(self):
        """Get list of all tables (collections)"""
        pool = await self._get_connection_pool()
        
        async with pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """)
            return [{"name": table['table_name']} for table in tables]


# Updated RAG Manager to use PostgreSQL
class RAGManager:
    keys_to_ignore = ['user_id', 'document_type', 'memory_type', 'collection_name']

    def __init__(self, collection_name: str = None, model_name: str = None) -> None:
        """
        Initializes the RAGManager with PostgreSQL backend.
        """
        if collection_name is None:
            raise ValueError("Collection name must be provided.")
        if model_name is None:
            raise ValueError("Model name must be provided.")

        self.collection_name = collection_name
        self.vectordb = PostgresVectorDB(collection_name=self.collection_name, model_name=model_name)
        self.limit = self.vectordb.limit
        self.threshold = -1

    @staticmethod
    def generate_id(content):
        """Generates a unique ID for the content to prevent duplicate data."""
        namespace = uuid.UUID('12345678-1234-5678-1234-567812345678')
        return str(uuid.uuid5(namespace, str(content)))

    def get_document(self, data: Dict[str, Any], user_id: Union[uuid.UUID, str], id: Union[uuid.UUID, str, None] = None) -> List[Dict[str, Any]]:
        """
        Converts input data into a document structure for storage.
        """
        # Rename 'show_time' to 'timestamp' and 'content' to 'text'
        if 'show_time' in data:
            data['timestamp'] = data.pop('show_time')
        if 'content' in data:
            data['text'] = data.pop('content')

        from src.schemas.document_schemas import RAGPayload
        
        payload = RAGFormatter.create_payload(
            id=id,
            user_id=str(user_id),
            collection_name=self.collection_name,
            data=data
        )
        formatted_content = payload.text
        document_metadata = payload.metadata
        document_metadata['user_id'] = str(user_id)
        return [{"page_content": formatted_content, "metadata": payload.metadata}]

    async def insert(self, data: List[Dict[str, Any]], user_id: Union[uuid.UUID, str] = None) -> Optional[Union[str, List[str]]]:
        """
        Creates memory entries for data.
        """
        if len(data) > 1:
            # Handle multiple data items
            docs_to_add = []
            for item in data:
                new_uuid = self.generate_id(item)
                doc = self.get_document(data=item, id=new_uuid, user_id=user_id)
                docs_to_add.extend(doc)
                logger.debug(docs_to_add)
            
            if docs_to_add:
                await self.vectordb.insert(docs_to_add, is_batch=True)
                ids = [doc['metadata']['id'] for doc in docs_to_add]
                return ids
        else:
            # Handle single data item
            new_uuid = self.generate_id(data[0])
            data_document = self.get_document(data=data[0], user_id=user_id, id=new_uuid)
            await self.vectordb.insert(data_document, is_batch=False)
            id = data_document[0]['metadata']['id']
            return id

    async def search(
        self,
        query: str,
        user_id: Union[uuid.UUID, str] = None,
        date: Optional[Union[dt, str]] = None,
        end_date: Optional[Union[dt, str]] = None,
        date_operator: str = "gte",
        ref_time: Optional[Union[dt, str]] = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using PostgreSQL vector similarity.
        """
        filters = {}

        if date:
            if isinstance(date, str):
                try:
                    start_dt = dt.fromisoformat(date)
                except Exception as e:
                    raise ValueError(f"Invalid date format: {date}. Expected RFC3339 string. Error: {e}")
            else:
                start_dt = date

            if end_date:
                if isinstance(end_date, str):
                    try:
                        end_dt = dt.fromisoformat(end_date)
                    except Exception as e:
                        raise ValueError(f"Invalid end_date format: {end_date}. Expected RFC3339 string. Error: {e}")
                else:
                    end_dt = end_date

                dt_range_kwargs = {
                    "gte": start_dt.isoformat(),
                    "lte": end_dt.isoformat()
                }
            else:
                dt_range_kwargs = {}
                if date_operator == "gt":
                    dt_range_kwargs["gt"] = start_dt.isoformat()
                elif date_operator == "gte":
                    dt_range_kwargs["gte"] = start_dt.isoformat()
                elif date_operator == "lt":
                    dt_range_kwargs["lt"] = start_dt.isoformat()
                elif date_operator == "lte":
                    dt_range_kwargs["lte"] = start_dt.isoformat()
                elif date_operator == "eq":
                    dt_range_kwargs["gte"] = start_dt.isoformat()
                    dt_range_kwargs["lte"] = start_dt.isoformat()
                else:
                    raise ValueError(f"Unsupported date operator: {date_operator}")

            filters['timestamp'] = MatchAnyOrInterval(**dt_range_kwargs)

        if user_id:
            filters['user_id'] = MatchAnyOrInterval(any=[str(user_id)])

        documents = await self.vectordb.search(query, filters=filters)
        if not documents:
            logger.info("No documents returned from vector search")
            return []

        logger.info(f"Found {len(documents)} documents before filtering")

        # Apply threshold filtering
        if date:
            search_threshold = 0.1
        else:
            search_threshold = threshold if threshold is not None else self.threshold

        # Fix the score key reference
        score_key = self.vectordb._PostgresVectorDB__score_key
        filtered_documents = [doc for doc in documents if doc.score[score_key] >= search_threshold]
        
        logger.info(f"Found {len(filtered_documents)} documents after threshold filtering (threshold: {search_threshold})")
        
        result = [{
            k: v for k, v in document.metadata.items() if k not in self.keys_to_ignore
        } for document in filtered_documents]

        if ref_time:
            if isinstance(ref_time, str):
                try:
                    ref_dt = dt.fromisoformat(ref_time)
                except Exception as e:
                    raise ValueError(f"Invalid ref_time format: {ref_time}. Expected RFC3339 string. Error: {e}")
            else:
                ref_dt = ref_time
            return self.sort_by_time_proximity(result, ref_dt)
        
        return result

    def sort_by_time_proximity(self, documents: List[Dict[str, Any]], ref_time: dt) -> List[Dict[str, Any]]:
        """Sort documents by time proximity to reference time"""
        def time_diff(doc):
            ts_str = doc.get("timestamp") or doc.get("metadata", {}).get("timestamp")
            if ts_str:
                try:
                    ts = dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                    return abs((ts - ref_time).total_seconds())
                except Exception:
                    return float("inf")
            return float("inf")
        
        return sorted(documents, key=time_diff)

    async def get(self, id):
        """Retrieve a document by its identifier"""
        records = await self.vectordb.get(id)
        if not records:
            return []
            
        keys_to_ignore = [key for key in self.keys_to_ignore if key != 'user_id']
        
        metadata = [
            {k: v for k, v in record['payload'][self.vectordb._PostgresVectorDB__metadata_payload_key].items() 
             if k not in keys_to_ignore}
            for record in records
        ]
        return metadata

    async def delete(self, id):
        """Delete a document by ID"""
        await self.vectordb.delete_one([id])
        return {"message": f"Data with id {id} deleted successfully!"}

    async def close(self):
        """Close the database connection"""
        await self.vectordb.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.create_task(self.close())

class RAGFormatter:
    """
    A utility class providing static methods to create document payloads for storing memories.

    Methods:
    - create_payload: Creates a document_schema.RAGPayload instance from input data.
    """
    @staticmethod
    def create_payload(
        id: Union[uuid.UUID, str],
        user_id: str,
        collection_name: str,
        data: Dict[str, Any]
    ) -> RAGPayload:
        """
        Creates a document_schems.RAGPayload instance used for storing memory data.

        Args:
            id (Union[uuid.UUID, str]): Unique identifier for the document.
            user_id (str): Identifier for the user associated with the document.
            collection_name (str): Name of the collection to store the document in.
            data (Dict[str, Any]): The content of the document.

        Returns:
            document_schems.RAGPayload: An instance containing the formatted document data.
        """
        if not id:
            id = str(uuid.uuid4())

        screen_name = data.get('screen_name', '')
        data_source = data.get('dataSource', '')
        author_name = data.get('author_name', '')
        text = data.get('text', '') 

        formatted_content = f"{text} {screen_name} {data_source} {author_name}".strip()

        # Append the id to the metadata
        data['id'] = str(id)
        data['document_type'] = collection_name.replace("-", "_")
        data['memory_type'] = "long_term_memory"
        data['collection_name'] = collection_name

        if 'user_id' in data:
            data['user_id'] = user_id

        # Remove unwanted fields from metadata
        if 'tweetID' in data:
            data.pop('tweetID', None)
        if 'tweet_id' in data:
            data.pop('tweet_id', None)
        if 'dbID' in data:
            data.pop('dbID', None)

        return RAGPayload(
            id=str(id),
            user_id=user_id,
            text=formatted_content,
            document_type="preference",
            memory_type="long_term_memory",
            collection_name=collection_name,
            metadata=data  # Store the entire payload in metadata
        )


# Database setup and migration utilities
class PostgreSQLVectorSetup:
    """Utility class for setting up PostgreSQL with pgvector extension"""
    
    @staticmethod
    async def setup_database():
        """Setup PostgreSQL database with required extensions and configurations"""
        
        # Connection for initial setup (might need superuser privileges)
        admin_config = {
            'host': settings.POSTGRES_HOST,
            'port': int(settings.POSTGRES_PORT),
            'database': 'postgres', # Connect to default database first #settings.POSTGRES_DB, 
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD,
        }
        
        target_db = settings.POSTGRES_DB
        
        try:
            # Connect to create database if it doesn't exist
            conn = await asyncpg.connect(**admin_config)
            
            # Check if target database exists
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", target_db
            )
            
            if not db_exists:
                await conn.execute(f'CREATE DATABASE "{target_db}"')
                logger.info(f"Created database '{target_db}'")
            
            await conn.close()
            
            # Connect to target database to set up extensions
            target_config = admin_config.copy()
            target_config['database'] = target_db
            
            conn = await asyncpg.connect(**target_config)
            
            # Install pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")
            
            # Set recommended configuration for vector operations
            await conn.execute("SET maintenance_work_mem = '512MB';")
            await conn.execute("SET max_parallel_maintenance_workers = 7;")
            
            await conn.close()
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise