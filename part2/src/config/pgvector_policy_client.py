import asyncpg
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime as dt
from sentence_transformers import SentenceTransformer
import uuid
import pandas as pd
from src.utils.database_manager import db_manager
from src.config.settings import settings
from src.utils.app_utils import  AppUtil
from src.config.logger import Logger
from typing import Dict
import os

logger = Logger(__name__)

class PostgresVectorClient:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        
    def encode_text(self, text: str) -> List[float]:
        """Encode text to vector"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to vectors"""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    async def create_tables(self):
        """Create necessary tables with vector support"""
        async with db_manager.get_connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create insurance policies table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS insurance_policies (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    policy_number VARCHAR(100) UNIQUE NOT NULL,
                    insured_name VARCHAR(255) NOT NULL,
                    sum_insured DOUBLE PRECISION NOT NULL,
                    premium DOUBLE PRECISION NOT NULL,
                    own_retention_ppn DOUBLE PRECISION,
                    own_retention_sum_insured DOUBLE PRECISION,
                    own_retention_premium DOUBLE PRECISION,
                    treaty_ppn DOUBLE PRECISION,
                    treaty_sum_insured DOUBLE PRECISION,
                    treaty_premium DOUBLE PRECISION,
                    insurance_period_start_date TIMESTAMP NOT NULL,
                    insurance_period_end_date TIMESTAMP NOT NULL,
                    content_vector vector(384),
                    searchable_content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            # Create indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_policies_vector 
                ON insurance_policies USING ivfflat (content_vector vector_cosine_ops) 
                WITH (lists = 100);
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_policies_metadata 
                ON insurance_policies USING gin (metadata);
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_policies_period 
                ON insurance_policies (insurance_period_start_date, insurance_period_end_date);
            ''')
            
            logger.info("Database tables and indexes created successfully")
    
    async def insert_policy(self, policy_data: Dict[str, Any]) -> str:
        """Insert a single insurance policy with vector embedding"""
        if 'user_id' not in policy_data:
            raise ValueError("user_id is required in policy_data for filtering")

        # Create a cleaned copy for metadata where all non-serializable objects are converted
        metadata = policy_data.copy()

        for key, value in metadata.items():
            metadata[key] = AppUtil.convert_value(value)

        metadata['user_id'] = str(policy_data['user_id'])
        # Create searchable content using the original data
        period_start = policy_data['insurance_period_start_date']
        period_end = policy_data['insurance_period_end_date']

        # Handle potential None/NaT values in the searchable content string
        start_str = str(period_start) if pd.notna(period_start) else "N/A"
        end_str = str(period_end) if pd.notna(period_end) else "N/A"

        searchable_content = f"""
        Policy: {policy_data['policy_number']}
        Insured: {policy_data['insured_name']}
        Sum Insured: ${policy_data['sum_insured']:,.2f}
        Premium: ${policy_data['premium']:,.2f}
        Period: {start_str} to {end_str}
        Own Retention: {policy_data.get('own_retention_ppn', 'N/A')}%
        Treaty: {policy_data.get('treaty_ppn', 'N/A')}%
        """

        print(type(searchable_content))
        print(searchable_content)
        vector = self.encode_text(searchable_content)

        logger.debug(vector)

        async with db_manager.get_connection() as conn:
            policy_id = str(uuid.uuid4())

            query = '''
                INSERT INTO insurance_policies (
                    id, policy_number, insured_name, sum_insured, premium,
                    own_retention_ppn, own_retention_sum_insured, own_retention_premium,
                    treaty_ppn, treaty_sum_insured, treaty_premium,
                    insurance_period_start_date, insurance_period_end_date,
                    content_vector, searchable_content, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            '''

            # Convert pandas timestamps to Python datetime for SQL insertion if needed
            def ensure_datetime(value):
                if isinstance(value, pd.Timestamp):
                    return value.to_pydatetime() if pd.notna(value) else None
                return value

            await conn.execute(
                query,
                uuid.UUID(policy_id),
                policy_data['policy_number'],
                policy_data['insured_name'],
                policy_data['sum_insured'],
                policy_data['premium'],
                policy_data.get('own_retention_ppn'),
                policy_data.get('own_retention_sum_insured'),
                policy_data.get('own_retention_premium'),
                policy_data.get('treaty_ppn'),
                policy_data.get('treaty_sum_insured'),
                policy_data.get('treaty_premium'),
                ensure_datetime(policy_data['insurance_period_start_date']),
                ensure_datetime(policy_data['insurance_period_end_date']),
                vector,
                searchable_content,
                json.dumps(metadata)
            )

            return policy_id
    
    async def search_policies(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.1,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search policies using vector similarity with optional user filtering"""

        query_vector = self.encode_text(query)

        # Build WHERE clause
        where_conditions = ["1 = 1"]
        params = [query_vector, limit]
        param_count = 2

        if filters:
            for key, value in filters.items():
                param_count += 1
                if key == 'user_id':
                    # Filter by user_id stored in metadata
                    where_conditions.append(f"(metadata->>'user_id') = ${param_count}")
                    params.append(str(value))
                elif key == 'insured_name':
                    where_conditions.append(f"insured_name ILIKE ${param_count}")
                    params.append(f"%{value}%")
                elif key == 'policy_number':
                    where_conditions.append(f"policy_number = ${param_count}")
                    params.append(value)
                elif key == 'min_sum_insured':
                    where_conditions.append(f"sum_insured >= ${param_count}")
                    params.append(value)
                elif key == 'date_range':
                    param_count += 1
                    where_conditions.append(f"insurance_period_start_date >= ${param_count} AND insurance_period_end_date <= ${param_count + 1}")
                    params.extend([value['start'], value['end']])
                    param_count += 1

        where_clause = " AND ".join(where_conditions)

        search_query = f'''
            SELECT
                *,
                1 - (content_vector <=> $1::vector) as similarity_score
            FROM insurance_policies
            WHERE {where_clause}
            AND (1 - (content_vector <=> $1::vector)) >= {similarity_threshold}
            ORDER BY content_vector <=> $1::vector
            LIMIT $2
        '''

        async with db_manager.get_connection() as conn:
            rows = await conn.fetch(search_query, *params)

            results = []
            for row in rows:
                result = dict(row)
                result['id'] = str(result['id'])
                result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                results.append(result)

            return results

    
    async def get_policy_by_id(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get policy by ID"""
        async with db_manager.get_connection() as conn:
            query = "SELECT * FROM insurance_policies WHERE id = $1"
            row = await conn.fetchrow(query, uuid.UUID(policy_id))
            
            if row:
                result = dict(row)
                result['id'] = str(result['id'])
                result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                return result
            return None
    
    async def delete_policy(self, policy_id: str) -> bool:
        """Delete policy by ID"""
        async with db_manager.get_connection() as conn:
            result = await conn.execute(
                "DELETE FROM insurance_policies WHERE id = $1",
                uuid.UUID(policy_id)
            )
            return "DELETE 1" in result
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with db_manager.get_connection() as conn:
            stats = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_policies,
                    AVG(sum_insured) as avg_sum_insured,
                    AVG(premium) as avg_premium,
                    MIN(insurance_period_start_date) as earliest_policy,
                    MAX(insurance_period_end_date) as latest_policy
                FROM insurance_policies
            ''')
            
            # Get database size
            db_size = await conn.fetchval('''
                SELECT pg_size_pretty(pg_database_size(current_database()))
            ''')
            
            return {
                'total_policies': stats['total_policies'],
                'avg_sum_insured': float(stats['avg_sum_insured']) if stats['avg_sum_insured'] else 0,
                'avg_premium': float(stats['avg_premium']) if stats['avg_premium'] else 0,
                'earliest_policy': stats['earliest_policy'],
                'latest_policy': stats['latest_policy'],
                'database_size': db_size
            }