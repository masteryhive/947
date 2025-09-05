import asyncpg
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union, Tuple
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
import re

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

    async def drop_table(self, table_name: str) -> bool:
        """
        Drop a table from the database.

        Args:
            table_name: Name of the table to drop.

        Returns:
            bool: True if the table was dropped successfully, False otherwise.
        """
        # Validate table name to prevent SQL injection
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            logger.error(f"Invalid table name: {table_name}")
            return False

        try:
            async with db_manager.get_connection() as conn:
                await conn.execute(f'DROP TABLE IF EXISTS {table_name};')
                logger.info(f"Table {table_name} dropped successfully")
                return True
        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {e}")
            return False
        
    async def insert_or_update_policy(self, policy_data: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Insert a policy if it doesn't exist, or update it if it does.
        Returns a tuple of (policy_id, is_new) where is_new is True if the policy was inserted,
        or False if it was updated.
        """
        if 'user_id' not in policy_data:
            raise ValueError("user_id is required in policy_data for filtering")

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
        vector = str(self.encode_text(searchable_content))

        async with db_manager.get_connection() as conn:
            policy_id = str(uuid.uuid4())

            # First check if the policy already exists
            existing_policy = await conn.fetchrow(
                "SELECT id FROM insurance_policies WHERE policy_number = $1",
                policy_data['policy_number']
            )

            if existing_policy:
                policy_id = str(existing_policy['id'])

                # Convert pandas timestamps to Python datetime for SQL insertion if needed
                def ensure_datetime(value):
                    if isinstance(value, pd.Timestamp):
                        return value.to_pydatetime() if pd.notna(value) else None
                    return value

                await conn.execute('''
                    UPDATE insurance_policies SET
                        insured_name = $2,
                        sum_insured = $3,
                        premium = $4,
                        own_retention_ppn = $5,
                        own_retention_sum_insured = $6,
                        own_retention_premium = $7,
                        treaty_ppn = $8,
                        treaty_sum_insured = $9,
                        treaty_premium = $10,
                        insurance_period_start_date = $11,
                        insurance_period_end_date = $12,
                        content_vector = $13,
                        searchable_content = $14,
                        policy_metadata = $15,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                ''',
                    uuid.UUID(policy_id),
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
                return policy_id, False
            else:
                # Policy doesn't exist, do an insert
                # Convert pandas timestamps to Python datetime for SQL insertion if needed
                def ensure_datetime(value):
                    if isinstance(value, pd.Timestamp):
                        return value.to_pydatetime() if pd.notna(value) else None
                    return value

                query = '''
                    INSERT INTO insurance_policies (
                        id, policy_number, insured_name, sum_insured, premium,
                        own_retention_ppn, own_retention_sum_insured, own_retention_premium,
                        treaty_ppn, treaty_sum_insured, treaty_premium,
                        insurance_period_start_date, insurance_period_end_date,
                        content_vector, searchable_content, policy_metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    RETURNING id
                '''

                result = await conn.fetchrow(
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
                return str(result['id']), True  # True indicates it was a new insert

        
    async def create_tables(self):
        """Create necessary tables with vector support"""
        async with db_manager.get_connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create insurance policies table
            await conn.execute(f'''
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
                    content_vector vector({settings.VECTOR_DIMENSION}),
                    searchable_content TEXT,
                    policy_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            # Create indexes to match your model
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_policy_number
                ON insurance_policies (policy_number);
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_insured_name
                ON insurance_policies (insured_name);
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_insurance_period 
                ON insurance_policies (insurance_period_start_date, insurance_period_end_date);
            ''')
            
            await conn.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_content_vector 
                ON insurance_policies USING ivfflat (content_vector vector_cosine_ops) 
                WITH (lists = 100);
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_policy_metadata 
                ON insurance_policies USING gin (policy_metadata);
            ''')
            
            logger.info("Database tables and indexes created successfully")

    async def recreate_table(self) -> bool:
        """
        Drop and recreate a table in the database.

        Args:
            table_name: Name of the table to recreate.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            table_name = 'insurance_policies'
            # Drop the table
            success = await self.drop_table(table_name)
            if not success:
                return False

            # Recreate the table based on the table_name
            if table_name:
                await self.create_tables()
                logger.info(f"Table {table_name} recreated successfully")
                return True
            else:
                logger.error(f"Recreation of table {table_name} is not supported")
                return False
        except Exception as e:
            logger.error(f"Error recreating table {table_name}: {e}")
            return False
    
    async def insert_policy(self, policy_data: Dict[str, Any]) -> str:
        """Insert a single insurance policy with vector embedding"""
        if 'user_id' not in policy_data:
            raise ValueError("user_id is required in policy_data for filtering")

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

        vector = str(self.encode_text(searchable_content))

        async with db_manager.get_connection() as conn:
            policy_id = str(uuid.uuid4())

            # Updated query to use policy_metadata instead of metadata
            query = '''
                INSERT INTO insurance_policies (
                    id, policy_number, insured_name, sum_insured, premium,
                    own_retention_ppn, own_retention_sum_insured, own_retention_premium,
                    treaty_ppn, treaty_sum_insured, treaty_premium,
                    insurance_period_start_date, insurance_period_end_date,
                    content_vector, searchable_content, policy_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (policy_number) 
                DO UPDATE SET
                    insured_name = EXCLUDED.insured_name,
                    sum_insured = EXCLUDED.sum_insured,
                    premium = EXCLUDED.premium,
                    own_retention_ppn = EXCLUDED.own_retention_ppn,
                    own_retention_sum_insured = EXCLUDED.own_retention_sum_insured,
                    own_retention_premium = EXCLUDED.own_retention_premium,
                    treaty_ppn = EXCLUDED.treaty_ppn,
                    treaty_sum_insured = EXCLUDED.treaty_sum_insured,
                    treaty_premium = EXCLUDED.treaty_premium,
                    insurance_period_start_date = EXCLUDED.insurance_period_start_date,
                    insurance_period_end_date = EXCLUDED.insurance_period_end_date,
                    content_vector = EXCLUDED.content_vector,
                    searchable_content = EXCLUDED.searchable_content,
                    policy_metadata = EXCLUDED.policy_metadata,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            '''

            # Convert pandas timestamps to Python datetime for SQL insertion if needed
            def ensure_datetime(value):
                if isinstance(value, pd.Timestamp):
                    return value.to_pydatetime() if pd.notna(value) else None
                return value

            result = await conn.fetchrow(
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

            return str(result['id'])
    
    async def search_policies(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.1,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search policies using vector similarity with optional user filtering"""

        query_vector = str(self.encode_text(query))

        # Build WHERE clause
        where_conditions = ["1 = 1"]
        params = [query_vector, limit]
        param_count = 2

        if filters:
            for key, value in filters.items():
                param_count += 1
                if key == 'user_id':
                    # Updated to use policy_metadata instead of metadata
                    where_conditions.append(f"(policy_metadata->>'user_id') = ${param_count}")
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
                # Updated to use policy_metadata
                result['policy_metadata'] = json.loads(result['policy_metadata']) if result['policy_metadata'] else {}
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