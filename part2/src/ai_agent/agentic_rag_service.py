from .nodes.node import Node
from src.schemas import monitor_schemas, chat_schemas, router_schema, state, thought_schema
from ..utils.app_utils import AppUtil
from .routers.router import router
from langgraph.graph import StateGraph
from src.config.pgvector_policy_client import PostgresVectorClient
from src.ai_agent.llm_service import GeminiLLMService
from src.schemas.chat_schemas import QueryType, QueryClassification
import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime as dt
from uuid import UUID
import asyncio
from src.config.config_helper import Configuration
from ..config.logger import Logger

logger = Logger(__name__)

class AgenticRAGService:
    def __init__(self):
        self.llm_service = GeminiLLMService()
        self.vector_client = PostgresVectorClient()
        self.query_patterns = self._initialize_query_patterns()
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for query classification"""
        return {
            'policy_numbers': [
                r'policy\s+([A-Za-z0-9\/\-_]+)',
                r'([A-Za-z0-9\/\-_]{5,})',
                r'policy\s*#?\s*([A-Za-z0-9\/\-_]+)'
            ],
            'amounts': [
                r'\$?([\d,]+(?:\.\d{2})?)',
                r'over\s+\$?([\d,]+)',
                r'above\s+\$?([\d,]+)',
                r'exceeds?\s+\$?([\d,]+)'
            ],
            'calculations': [
                r'total|sum|average|mean|percentage|%|calculate|compute',
                r'highest|lowest|maximum|minimum|max|min',
                r'by month|monthly|annual|yearly'
            ],
            'comparisons': [
                r'compare|vs|versus|against',
                r'highest|lowest|best|worst',
                r'which.*has.*most|which.*has.*least'
            ],
            'claims': [
                r'claim|claims|loss|losses',
                r'claim amount|loss amount',
                r'claim date|loss date'
            ]
        }
    
    async def process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Main entry point for processing user queries"""
        try:
            # Step 1: Classify the query
            classification = await self._classify_query_enhanced(query)
            logger.info(f"Query classification: {classification}")
            
            # Step 2: Extract search parameters
            search_params = await self._extract_search_parameters(query, classification)
            
            # Step 3: Retrieve relevant documents using multi-stage retrieval
            context_documents = await self._multi_stage_retrieval(
                query, search_params, user_id, classification
            )
            
            # Step 4: Post-process documents if calculations are needed
            if classification.requires_calculation:
                processed_docs = await self._process_for_calculations(
                    context_documents, classification
                )
            else:
                processed_docs = context_documents
            
            # Step 5: Generate intelligent response
            response = await self._generate_intelligent_response(
                query, processed_docs, classification
            )
            
            return {
                'answer': response['answer'],
                'classification': classification.__dict__,
                'documents_used': len(processed_docs),
                'search_params': search_params,
                'metadata': response.get('metadata', {}),
                'confidence': classification.confidence,
                'context_documents': context_documents
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"I encountered an error processing your question: {str(e)}",
                'error': True,
                'classification': None,
                'documents_used': 0
            }
    
    async def _classify_query_enhanced(self, query: str) -> QueryClassification:
        """Enhanced query classification using patterns + LLM"""
        
        # Pattern-based classification
        entities = self._extract_entities(query)
        numerical_filters = self._extract_numerical_filters(query)
        date_filters = self._extract_date_filters(query)
        
        # Determine query type based on patterns
        query_lower = query.lower()
        
        if any(pattern in query_lower for pattern in ['claim', 'loss']):
            query_type = QueryType.CLAIMS_ANALYSIS
        elif any(pattern in query_lower for pattern in ['total', 'sum', 'average', 'calculate', 'percentage']):
            query_type = QueryType.CALCULATION
        elif any(pattern in query_lower for pattern in ['compare', 'highest', 'lowest', 'which', 'most']):
            query_type = QueryType.COMPARISON
        elif entities.get('policy_numbers'):
            query_type = QueryType.SPECIFIC_POLICY
        elif any(pattern in query_lower for pattern in ['all policies', 'show me', 'find policies']):
            query_type = QueryType.AGGREGATE_ANALYSIS
        else:
            query_type = QueryType.GENERAL_INFO
        
        # Determine if calculation is required
        calculation_keywords = ['total', 'sum', 'average', 'mean', 'percentage', 'calculate', 'compute']
        requires_calculation = any(keyword in query_lower for keyword in calculation_keywords)
        
        # Determine calculation type
        calculation_type = None
        if requires_calculation:
            if 'total' in query_lower or 'sum' in query_lower:
                calculation_type = 'sum'
            elif 'average' in query_lower or 'mean' in query_lower:
                calculation_type = 'average'
            elif 'percentage' in query_lower or '%' in query_lower:
                calculation_type = 'percentage'
            elif 'count' in query_lower:
                calculation_type = 'count'
        
        return QueryClassification(
            query_type=query_type,
            entities=entities,
            numerical_filters=numerical_filters,
            date_filters=date_filters,
            requires_calculation=requires_calculation,
            calculation_type=calculation_type,
            confidence=0.8  # Pattern-based classification confidence
        )
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities like policy numbers, names, etc."""
        entities = {
            'policy_numbers': [],
            'insured_names': [],
            'amounts': []
        }
        
        # Extract policy numbers
        for pattern in self.query_patterns['policy_numbers']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['policy_numbers'].extend(matches)
        
        # Extract amounts
        for pattern in self.query_patterns['amounts']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['amounts'].extend([float(m.replace(',', '')) for m in matches])
        
        return entities
    
    def _extract_numerical_filters(self, query: str) -> Dict[str, Any]:
        """Extract numerical filters and thresholds"""
        filters = {}
        query_lower = query.lower()
        
        # Extract threshold amounts
        over_match = re.search(r'over\s+\$?([\d,]+(?:\.\d{2})?)', query_lower)
        if over_match:
            filters['min_amount'] = float(over_match.group(1).replace(',', ''))
        
        above_match = re.search(r'above\s+\$?([\d,]+(?:\.\d{2})?)', query_lower)
        if above_match:
            filters['min_amount'] = float(above_match.group(1).replace(',', ''))
        
        under_match = re.search(r'under\s+\$?([\d,]+(?:\.\d{2})?)', query_lower)
        if under_match:
            filters['max_amount'] = float(under_match.group(1).replace(',', ''))
        
        return filters
    
    def _extract_date_filters(self, query: str) -> Dict[str, Any]:
        """Extract date-related filters"""
        filters = {}
        query_lower = query.lower()
        
        # Extract year mentions
        year_match = re.search(r'(20\d{2})', query)
        if year_match:
            filters['year'] = int(year_match.group(1))
        
        # Extract month mentions
        month_patterns = {
            r'january|jan': 1, r'february|feb': 2, r'march|mar': 3,
            r'april|apr': 4, r'may': 5, r'june|jun': 6,
            r'july|jul': 7, r'august|aug': 8, r'september|sep': 9,
            r'october|oct': 10, r'november|nov': 11, r'december|dec': 12
        }
        
        for pattern, month_num in month_patterns.items():
            if re.search(pattern, query_lower):
                filters['month'] = month_num
                break
        
        return filters
    
    async def _extract_search_parameters(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Extract parameters for vector search based on classification"""
        search_params = {
            'limit': 10,
            'similarity_threshold': 0.1,
            'filters': {}
        }
        
        # Adjust based on query type
        if classification.query_type == QueryType.SPECIFIC_POLICY:
            search_params['limit'] = 5
            search_params['similarity_threshold'] = 0.3
        elif classification.query_type == QueryType.AGGREGATE_ANALYSIS:
            search_params['limit'] = 50
            search_params['similarity_threshold'] = 0.1
        elif classification.query_type == QueryType.CALCULATION:
            search_params['limit'] = 100
            search_params['similarity_threshold'] = 0.05
        
        # Add entity-based filters with validation
        if classification.entities.get('policy_numbers'):
            policy_numbers = classification.entities['policy_numbers']
            if isinstance(policy_numbers, list) and policy_numbers:
                # Filter out invalid policy numbers
                valid_policies = [
                    pn for pn in policy_numbers 
                    if isinstance(pn, str) and len(pn) >= 3 and pn.lower() not in [
                        'which', 'insured', 'party', 'highest', 'lowest', 'treaty'
                    ]
                ]
                if valid_policies:
                    if len(valid_policies) == 1:
                        search_params['filters']['policy_number'] = valid_policies[0]
                    else:
                        search_params['filters']['policy_numbers'] = valid_policies
        
        # Add numerical filters with validation
        if classification.numerical_filters.get('min_amount'):
            try:
                min_amount = float(classification.numerical_filters['min_amount'])
                search_params['filters']['min_sum_insured'] = min_amount
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_amount: {classification.numerical_filters.get('min_amount')}")
        
        if classification.numerical_filters.get('max_amount'):
            try:
                max_amount = float(classification.numerical_filters['max_amount'])
                search_params['filters']['max_sum_insured'] = max_amount
            except (ValueError, TypeError):
                logger.warning(f"Invalid max_amount: {classification.numerical_filters.get('max_amount')}")
        
        # Add date filters with validation
        if classification.date_filters.get('year'):
            try:
                year = int(classification.date_filters['year'])
                if 2020 <= year <= 2030:  # Reasonable year range
                    search_params['filters']['period_year'] = year
            except (ValueError, TypeError):
                logger.warning(f"Invalid year: {classification.date_filters.get('year')}")
        
        return search_params
    
    async def _multi_stage_retrieval(
        self, 
        query: str, 
        search_params: Dict[str, Any], 
        user_id: str,
        classification: QueryClassification
    ) -> List[Dict[str, Any]]:
        """Multi-stage document retrieval strategy"""
        
        # Add user filter
        if user_id:
            search_params['filters']['user_id'] = user_id
        
        # Stage 1: Primary vector search
        documents = await self.vector_client.search_policies(
            query=query,
            limit=search_params['limit'],
            similarity_threshold=search_params['similarity_threshold'],
            filters=search_params['filters']
        )
        
        # Stage 2: If insufficient results and specific entities mentioned, try relaxed search
        if len(documents) < 5 and classification.entities:
            relaxed_params = search_params.copy()
            relaxed_params['similarity_threshold'] = 0.05
            relaxed_params['limit'] = 20
            
            additional_docs = await self.vector_client.search_policies(
                query=" ".join(classification.entities.get('policy_numbers', []) + 
                              [str(amt) for amt in classification.entities.get('amounts', [])]),
                limit=relaxed_params['limit'],
                similarity_threshold=relaxed_params['similarity_threshold'],
                filters=relaxed_params['filters']
            )
            
            # Merge and deduplicate
            seen_ids = {doc['id'] for doc in documents}
            for doc in additional_docs:
                if doc['id'] not in seen_ids:
                    documents.append(doc)
                    seen_ids.add(doc['id'])
        
        return documents
    
    async def _process_for_calculations(
        self, 
        documents: List[Dict[str, Any]], 
        classification: QueryClassification
    ) -> List[Dict[str, Any]]:
        """Process documents for calculations and aggregations"""
        
        if not documents:
            return documents
        
        # Convert to DataFrame for easier calculations
        df_data = []
        for doc in documents:
            metadata = doc.get('policy_metadata', {})
            df_data.append({
                'id': doc['id'],
                'policy_number': metadata.get('policy_number', ''),
                'insured_name': metadata.get('insured_name', ''),
                'sum_insured': metadata.get('sum_insured', 0),
                'premium': metadata.get('premium', 0),
                'treaty_ppn': metadata.get('treaty_ppn', 0),
                'treaty_premium': metadata.get('treaty_premium', 0),
                'own_retention_ppn': metadata.get('own_retention_ppn', 0),
                'similarity_score': doc.get('similarity_score', 0),
                'searchable_content': doc.get('searchable_content', '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Perform calculations based on type
        calculation_results = {}
        
        if classification.calculation_type == 'sum':
            calculation_results['total_sum_insured'] = df['sum_insured'].sum()
            calculation_results['total_premium'] = df['premium'].sum()
            calculation_results['total_treaty_premium'] = df['treaty_premium'].sum()
        
        elif classification.calculation_type == 'average':
            calculation_results['avg_sum_insured'] = df['sum_insured'].mean()
            calculation_results['avg_premium'] = df['premium'].mean()
            calculation_results['avg_treaty_ppn'] = df['treaty_ppn'].mean()
        
        elif classification.calculation_type == 'percentage':
            total_premium = df['premium'].sum()
            total_treaty_premium = df['treaty_premium'].sum()
            if total_premium > 0:
                calculation_results['treaty_percentage'] = (total_treaty_premium / total_premium) * 100
        
        # Add calculation results to first document
        if documents and calculation_results:
            documents[0]['calculation_results'] = calculation_results
        
        return documents
    
    async def _generate_intelligent_response(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        classification: QueryClassification
    ) -> Dict[str, Any]:
        """Generate contextually appropriate response based on classification"""
        
        # Create enhanced system prompt based on query type
        system_prompts = {
            QueryType.SPECIFIC_POLICY: '''You are an insurance policy specialist. Provide detailed information about specific policies, including exact figures, dates, and coverage details. Always cite policy numbers.''',
            
            QueryType.AGGREGATE_ANALYSIS: '''You are an insurance portfolio analyst. Analyze the provided policies collectively, identify patterns, trends, and provide comprehensive summaries. Use tables and bullet points when helpful.''',
            
            QueryType.COMPARISON: '''You are an insurance comparison expert. Compare policies or metrics clearly, highlighting differences and similarities. Present comparisons in an organized, easy-to-understand format.''',
            
            QueryType.CALCULATION: '''You are an insurance data analyst. Perform accurate calculations based on the provided data. Show your work step-by-step and present results clearly with proper formatting.''',
            
            QueryType.CLAIMS_ANALYSIS: '''You are a claims analysis expert. Analyze claim-related data, calculate totals, identify patterns, and provide insights about claim performance and coverage effectiveness.''',
            
            QueryType.GENERAL_INFO: '''You are a knowledgeable insurance assistant. Provide helpful, accurate information about insurance concepts and policies based on the available data.'''
        }
        
        system_prompt = system_prompts.get(classification.query_type, system_prompts[QueryType.GENERAL_INFO])
        
        # Add calculation results context if available
        calculation_context = ""
        if documents and 'calculation_results' in documents[0]:
            calc_results = documents[0]['calculation_results']
            calculation_context = f"\nCalculation Results:\n{json.dumps(calc_results, indent=2)}\n"
        
        # Enhanced context preparation
        context_text = calculation_context
        for i, doc in enumerate(documents[:10]):  # Limit to top 10 for context
            metadata = doc.get('policy_metadata', {})
            context_text += f"\nDocument {i+1}:\n"
            context_text += f"Policy: {metadata.get('policy_number', 'N/A')}\n"
            context_text += f"Insured: {metadata.get('insured_name', 'N/A')}\n"
            context_text += f"Sum Insured: ${metadata.get('sum_insured', 0):,.2f}\n"
            context_text += f"Premium: ${metadata.get('premium', 0):,.2f}\n"
            context_text += f"Treaty %: {metadata.get('treaty_ppn', 0):.2f}%\n"
            context_text += f"Similarity: {doc.get('similarity_score', 0):.3f}\n\n"
        
        # Generate response using enhanced LLM service
        response = await self.llm_service.generate_response(
            query=query,
            context_documents=documents,
            system_prompt=system_prompt,
            context_text=context_text
        )
        
        return response