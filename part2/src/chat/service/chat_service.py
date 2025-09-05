from fastapi import APIRouter, status, Query, Path, Request
import uuid
from pathlib import Path
from datetime import datetime as dt
from src.exceptions.custom_exception import (
    APIAuthenticationFailException,
    RecordNotAllowedException,
    InternalServerException, 
    RecordNotFoundException, 
    UnsupportedFileFormatException,
    ChatNotFoundException, 
    MAPAgentException
)
import base64
# from src.map_ai import MAPAgent
from src.chat.repository import chat_repository
from src.utils.app_notification_message import NotificationMessage
from src.schemas import monitor_schemas, chat_schemas
from src.utils.app_utils import  AppUtil
from src.config.settings import settings
from src.config.logger import Logger
from typing import List, Dict, Union, Any, Optional
import json
import pandas as pd
import numpy as np
import uuid
from typing import List, Dict, Any, Tuple
from datetime import datetime
from io import BytesIO
# from app.services.vector_service import PostgresVectorService
# from app.schemas.models import InsurancePolicySchema

logger = Logger(__name__)

class ChatService:
    def __init__(self):
        # self.vector_service = PostgresVectorService()
        self.required_columns = {
            'policy_number': str,
            'insured_name': str,
            'sum_insured': float,
            'premium': float
        }
        self.optional_columns = {
            'own_retention_ppn': float,
            'own_retention_sum_insured': float,
            'own_retention_premium': float,
            'treaty_ppn': float,
            'treaty_sum_insured': float,
            'treaty_premium': float,
            'period_of_insurance': str,  # Will be split into start/end dates
        }

    async def query(
        request: Request
    ):
        pass
        # try:
        #     form_data = await request.form()
        #     file_str = form_data.get("file")
        #     chat_str = form_data.get("chat")
            
        #     if not chat_str:
        #         raise ChatNotFoundException("Missing 'chat' field in form data")

        #     chat_dict = json.loads(chat_str)
        #     content = chat_dict.setdefault("content", {})
        #     if not file_str and not content.get("message"):
        #         raise ChatNotFoundException("You must enter message before proceeding")
        #     elif file_str:
        #         raw = await file_str.read()
        #         file_ext = file_str.filename.rsplit(".", 1)[-1].lower()
        #         AppUtil.embed_document(
        #             file_bytes=raw,
        #             file_extension=file_ext,
        #             document_type=document_type,
        #             user_id=content.get("user_id")
        #         )
            
        #     chat_request = chat_schemas.ChatRequest(**chat_dict)
        #     chat_data = chat_request.model_dump()
        #     chat_input = AppUtil.extract(chat_data)
        #     logger.info(f"chat_input {chat_input}")
        #     (
        #         session_id, message, category
        #     ) = AppUtil._extract_chat_input(chat_input)

        #     central_agent = MAPAgent(
        #         vdb_doc_upload_client=vdb_doc_upload_client,
        #         actions=actions[0] if actions else None
        #     )
        #     session_id, result = await central_agent(
        #         message, user_id=user['id'], session_id=session_id
        #     )
        #     logger.info("Chat processed successfully")
        # except Exception as ex:
        #     logger.error(f"Processing Sessions -> API v1/chats/session/ask-agent/: {ex}")
        #     raise InternalServerException()

        # return {
        #     "content": result.content if result is not None else None
        # }
    
    async def ingest_excel(self, file, chunk_size, chunk_overlap) -> Dict[str, Any]:
        """
        Perform insert document into the vector database data for a specific collection.
        """
        try:
            if not file.filename.endswith(('.xlsx', '.xls')):
                raise RecordNotAllowedException("File must be an Excel file (.xlsx or .xls)")
            # Check file size (limit to 50MB by default)
            max_size = int(settings.MAX_FILE_SIZE_MB) * 1024 * 1024
            file_content = await file.read()
            
            if len(file_content) > max_size:
                raise RecordNotAllowedException(f"File too large. Maximum size is {max_size // (1024*1024)}MB")
            print(file)
            # client = cache.get(collection_name)
            # result = await client.insert_document(file, chunk_size, chunk_overlap)
            logger.info("document inserted successfully")
        except Exception as ex:
            logger.error(f"Insertion performed -> v1/chat/insert-excel/: {ex}")
            raise InternalServerException()

        # return result
    
    def process_excel_file(
        self,
        file_content: bytes,
        header_row_index: int = 6, 
    ) -> Optional[pd.DataFrame]:
        """
        Process insurance dataframe that's already loaded in memory.
        
        Args:
            df (pd.DataFrame): Input dataframe with insurance data
            header_row_index (int): Row index containing column headers (default: 6)
            debug (bool): Whether to print debug information
            
        Returns:
            Optional[pd.DataFrame]: Processed dataframe matching database schema,
                                or None if processing fails
        """
        try:
            df = pd.read_excel(BytesIO(file_content))

            df_clean = self.extract_header_and_data(df, header_row_index)
            cleaned_data = self.map_columns_to_schema(df_clean)
            filtered_data = self.filter_data_rows(cleaned_data)
            validated_df = self.validate_schema(filtered_data)
            
            return validated_df
            
        except Exception as e:
            logger.error(f"Error processing insurance dataframe: {str(e)}")
            return InternalServerException()
        
    def parse_insurance_period(self, period_str: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Parse insurance period string into start and end dates.
        
        Args:
            period_str (str): Period string in format "dd/mm/yyyy - dd/mm/yyyy"
            
        Returns:
            Tuple[pd.Timestamp, pd.Timestamp]: Start and end dates, or (NaT, NaT) if parsing fails
            
        Example:
            >>> parse_insurance_period("01/01/2024 - 31/12/2024")
            (Timestamp('2024-01-01'), Timestamp('2024-12-31'))
        """
        if pd.isna(period_str):
            return pd.NaT, pd.NaT
        
        try:
            # Handle format like "01/01/2024 - 01/01/2025"
            if ' - ' in str(period_str):
                start_str, end_str = str(period_str).split(' - ')
                start_date = pd.to_datetime(start_str.strip(), format='%d/%m/%Y', errors='coerce')
                end_date = pd.to_datetime(end_str.strip(), format='%d/%m/%Y', errors='coerce')
                return start_date, end_date
            else:
                return pd.NaT, pd.NaT
        except Exception:
            return pd.NaT, pd.NaT


    def clean_numeric(self, series: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Clean numeric values by removing commas and converting to float.
        
        Args:
            series: Pandas Series or DataFrame column containing numeric data
            
        Returns:
            pd.Series: Cleaned numeric series with commas removed and converted to float
            
        Note:
            If a DataFrame is passed, only the first column will be used.
        """
        if series is None:
            return pd.Series([np.nan])
        
        # Ensure we're working with a Series
        if isinstance(series, pd.DataFrame):
            print(f"Warning: Got DataFrame instead of Series, taking first column")
            series = series.iloc[:, 0]
        
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')


    def extract_header_and_data(self, df: pd.DataFrame, header_row_index: int = 6) -> pd.DataFrame:
        """
        Extract column headers from specified row and return cleaned dataframe with data rows.
        
        Args:
            df (pd.DataFrame): Raw dataframe from Excel file
            header_row_index (int): Row index containing column headers (default: 6)
            
        Returns:
            pd.DataFrame: Cleaned dataframe with proper column headers and data rows only
        """
        df_clean = df.copy()
        
        # Extract column headers from specified row
        if len(df_clean) > header_row_index:
            column_headers = df_clean.iloc[header_row_index].values
            df_clean.columns = column_headers
            
            # Keep only data rows (from header_row_index + 1 onwards)
            df_clean = df_clean.iloc[header_row_index + 1:].copy()
        
        # Reset index and remove empty rows
        df_clean.reset_index(drop=True, inplace=True)
        df_clean = df_clean.dropna(how='all')
        
        return df_clean


    def map_columns_to_schema(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Map insurance data columns to the required database schema.
        
        Args:
            df_clean (pd.DataFrame): Cleaned dataframe with proper headers
            debug (bool): Whether to print debug information
            
        Returns:
            pd.DataFrame: Dataframe with columns mapped to schema requirements
            
        Schema mapping:
            - Column 0: SN (not used)
            - Column 1: INSURED → insured_name  
            - Column 2: POLICY NUMBER → policy_number
            - Column 3: DEBIT NOTE (not used)
            - Column 4: PERIOD OF INSURANCE → insurance_period_start_date, insurance_period_end_date
            - Column 5: SUM INSURED (main) → sum_insured
            - Column 6: PREMIUM (main) → premium
            - Column 7: PPN (own retention) → own_retention_ppn
            - Column 8: SUM INSURED (own retention) → own_retention_sum_insured
            - Column 9: PREMIUM (own retention) → own_retention_premium
            - Column 10: PPN (treaty) → treaty_ppn
            - Column 11: SUM INSURED (treaty) → treaty_sum_insured
            - Column 12: PREMIUM (treaty) → treaty_premium
        """
        logger.info("Column names:", df_clean.columns.tolist())
        logger.info("Number of columns:", len(df_clean.columns))
        
        cleaned_data = pd.DataFrame()
        
        # Basic columns using positional indexing
        cleaned_data['policy_number'] = df_clean.iloc[:, 2]  # POLICY NUMBER
        cleaned_data['insured_name'] = df_clean.iloc[:, 1]   # INSURED
        
        # Handle PERIOD OF INSURANCE - split into start and end dates
        period_col = df_clean.iloc[:, 4]  # PERIOD OF INSURANCE
        period_parsed = period_col.apply(self.parse_insurance_period)
        cleaned_data['insurance_period_start_date'] = [x[0] for x in period_parsed]
        cleaned_data['insurance_period_end_date'] = [x[1] for x in period_parsed]
        
        # Main financial columns
        cleaned_data['sum_insured'] = self.clean_numeric(df_clean.iloc[:, 5])
        cleaned_data['premium'] = self.clean_numeric(df_clean.iloc[:, 6])
        
        # Own Retention columns (positions 7-9)
        cleaned_data['own_retention_ppn'] = self.clean_numeric(
            df_clean.iloc[:, 7] if len(df_clean.columns) > 7 else pd.Series([np.nan] * len(df_clean))
        )
        cleaned_data['own_retention_sum_insured'] = self.clean_numeric(
            df_clean.iloc[:, 8] if len(df_clean.columns) > 8 else pd.Series([np.nan] * len(df_clean))
        )
        cleaned_data['own_retention_premium'] = self.clean_numeric(
            df_clean.iloc[:, 9] if len(df_clean.columns) > 9 else pd.Series([np.nan] * len(df_clean))
        )
        
        # Treaty columns (positions 10-12)
        cleaned_data['treaty_ppn'] = self.clean_numeric(
            df_clean.iloc[:, 10] if len(df_clean.columns) > 10 else pd.Series([np.nan] * len(df_clean))
        )
        cleaned_data['treaty_sum_insured'] = self.clean_numeric(
            df_clean.iloc[:, 11] if len(df_clean.columns) > 11 else pd.Series([np.nan] * len(df_clean))
        )
        cleaned_data['treaty_premium'] = self.clean_numeric(
            df_clean.iloc[:, 12] if len(df_clean.columns) > 12 else pd.Series([np.nan] * len(df_clean))
        )
        
        return cleaned_data


    def filter_data_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out header rows and invalid data entries.
        
        Args:
            df (pd.DataFrame): Dataframe with mapped columns
            debug (bool): Whether to print debug information
            
        Returns:
            pd.DataFrame: Filtered dataframe with only valid data rows
        """
        df_filtered = df.dropna(subset=['policy_number'])
        if not df_filtered.empty:
            mask = ~df_filtered['insured_name'].astype(str).str.contains(
                'insured|sn|head office|ppn|policy', na=False, case=False
            )
            df_filtered = df_filtered[mask]
        df_filtered.reset_index(drop=True, inplace=True)
        
        return df_filtered


    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that the dataframe matches the required database schema.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            pd.DataFrame: Validated dataframe with proper data types
            
        Required schema:
            - policy_number: string
            - insured_name: string
            - sum_insured: double
            - premium: double
            - own_retention_ppn: double
            - own_retention_sum_insured: double
            - own_retention_premium: double
            - treaty_ppn: double
            - treaty_sum_insured: double
            - treaty_premium: double
            - insurance_period_start_date: timestamp
            - insurance_period_end_date: timestamp
        """
        required_columns = [
            'policy_number', 'insured_name', 'sum_insured', 'premium',
            'own_retention_ppn', 'own_retention_sum_insured', 'own_retention_premium',
            'treaty_ppn', 'treaty_sum_insured', 'treaty_premium',
            'insurance_period_start_date', 'insurance_period_end_date'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Warning: Missing columns: {missing_columns}")
        
        # Ensure proper data types for numeric columns
        numeric_columns = [
            'sum_insured', 'premium', 'own_retention_ppn', 
            'own_retention_sum_insured', 'own_retention_premium',
            'treaty_ppn', 'treaty_sum_insured', 'treaty_premium'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure datetime columns are datetime type
        datetime_columns = ['insurance_period_start_date', 'insurance_period_end_date']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
        

