import google.generativeai as genai
from src.config.settings import settings
from typing import List, Dict, Any, Optional
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from datetime import datetime

from src.config.config_helper import Configuration
from ..config.logger import Logger

logger = Logger(__name__)

class GeminiLLMService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        context_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate enhanced response using Gemini with intelligent context handling"""
        
        # Use provided context_text or prepare from documents
        if not context_text:
            context_text = self._prepare_enhanced_context(context_documents)
        
        # Use provided system prompt or create default
        if not system_prompt:
            system_prompt = self._get_adaptive_system_prompt(query, context_documents)
        
        # Create enhanced prompt with structured formatting
        prompt = self._create_enhanced_prompt(query, context_text, system_prompt)
        
        try:
            # Configure generation parameters for better responses
            generation_config = {
                'temperature': 0.3,  # Lower for more factual responses
                'top_p': 0.9,
                'max_output_tokens': 2048,
            }
            
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            # Post-process response for better formatting
            processed_answer = self._post_process_response(response.text, query, context_documents)
            
            return {
                'answer': processed_answer,
                'model_used': 'gemini-2.0-flash-lite',
                'context_documents_count': len(context_documents),
                'token_usage': self._estimate_tokens(prompt + response.text),
                'metadata': {
                    'response_length': len(processed_answer),
                    'has_calculations': self._has_calculations(response.text),
                    'confidence_level': self._assess_confidence(response.text, context_documents)
                }
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            fallback_answer = self._generate_fallback_response(query, context_documents)
            return {
                'answer': fallback_answer,
                'model_used': 'gemini-2.0-flash-lite',
                'context_documents_count': len(context_documents),
                'error': str(e),
                'is_fallback': True
            }
    
    def _prepare_enhanced_context(self, context_documents: List[Dict[str, Any]]) -> str:
        """Prepare enhanced context with intelligent structuring"""
        if not context_documents:
            return "No relevant policies found in the database."
        
        context_parts = []
        
        # Check if there are calculation results
        calc_results = None
        if context_documents and 'calculation_results' in context_documents[0]:
            calc_results = context_documents[0]['calculation_results']
        
        # Add calculation results summary if available
        if calc_results:
            context_parts.append("=== CALCULATED RESULTS ===")
            for key, value in calc_results.items():
                if isinstance(value, float):
                    if 'percentage' in key or 'ppn' in key:
                        context_parts.append(f"{key.replace('_', ' ').title()}: {value:.2f}%")
                    else:
                        context_parts.append(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
                else:
                    context_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            context_parts.append("")
        
        # Add policy details
        context_parts.append("=== POLICY DETAILS ===")
        
        for i, doc in enumerate(context_documents[:15]):  # Limit to top 15
            metadata = doc.get('policy_metadata', {})
            similarity = doc.get('similarity_score', 0)
            
            # Skip very low similarity documents unless there are few results
            if similarity < 0.1 and len(context_documents) > 5:
                continue
                
            context_parts.append(f"\n--- Policy {i+1} (Relevance: {similarity:.1%}) ---")
            context_parts.append(f"Policy Number: {metadata.get('policy_number', 'N/A')}")
            context_parts.append(f"Insured Name: {metadata.get('insured_name', 'N/A')}")
            context_parts.append(f"Sum Insured: ${metadata.get('sum_insured', 0):,.2f}")
            context_parts.append(f"Premium: ${metadata.get('premium', 0):,.2f}")
            
            if metadata.get('treaty_ppn'):
                context_parts.append(f"Treaty Rate: {metadata.get('treaty_ppn', 0):.2f}%")
                context_parts.append(f"Treaty Premium: ${metadata.get('treaty_premium', 0):,.2f}")
            
            if metadata.get('own_retention_ppn'):
                context_parts.append(f"Own Retention Rate: {metadata.get('own_retention_ppn', 0):.2f}%")
                context_parts.append(f"Own Retention Premium: ${metadata.get('own_retention_premium', 0):,.2f}")
            
            # Add period information
            start_date = metadata.get('insurance_period_start_date', 'N/A')
            end_date = metadata.get('insurance_period_end_date', 'N/A')
            context_parts.append(f"Period: {start_date} to {end_date}")
        
        return "\n".join(context_parts)
    
    def _get_adaptive_system_prompt(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate adaptive system prompt based on query characteristics"""
        
        base_prompt = '''You are an expert insurance analyst and advisor with deep knowledge of insurance policies, claims, and financial calculations.

CORE INSTRUCTIONS:
1. Provide accurate, data-driven responses based strictly on the provided context
2. Use specific policy numbers, amounts, and dates when referencing information
3. Show calculations step-by-step when performing mathematical operations
4. Format financial figures with proper currency symbols and thousand separators
5. Be precise with percentages and rates
6. If information is not available in the context, clearly state this limitation'''
        
        query_lower = query.lower()
        
        # Add specific instructions based on query type
        additional_instructions = []
        
        if any(word in query_lower for word in ['calculate', 'total', 'sum', 'average', 'percentage']):
            additional_instructions.append('''
CALCULATION GUIDELINES:
- Show mathematical work step-by-step
- Double-check all arithmetic
- Present final results prominently
- Explain the basis for calculations
- Use proper formatting for financial figures''')
        
        if any(word in query_lower for word in ['compare', 'highest', 'lowest', 'best', 'worst']):
            additional_instructions.append('''
COMPARISON GUIDELINES:
- Present comparisons in clear, structured format
- Highlight key differences and similarities
- Rank items when appropriate
- Provide context for the comparison criteria''')
        
        if any(word in query_lower for word in ['claim', 'loss', 'coverage']):
            additional_instructions.append('''
CLAIMS ANALYSIS GUIDELINES:
- Focus on claim amounts, dates, and coverage details
- Analyze patterns in claims data
- Consider policy periods when evaluating claims
- Highlight any irregularities or noteworthy findings''')
        
        if any(word in query_lower for word in ['treaty', 'retention', 'reinsurance']):
            additional_instructions.append('''
REINSURANCE ANALYSIS GUIDELINES:
- Clearly explain treaty vs. own retention splits
- Calculate proportional shares accurately
- Discuss risk transfer implications
- Present treaty arrangements clearly''')
        
        # Combine base prompt with additional instructions
        full_prompt = base_prompt
        if additional_instructions:
            full_prompt += "\n\nSPECIALIZED INSTRUCTIONS:" + "".join(additional_instructions)
        
        return full_prompt
    
    def _create_enhanced_prompt(self, query: str, context_text: str, system_prompt: str) -> str:
        """Create well-structured prompt for optimal LLM performance"""
        
        prompt = f"""{system_prompt}

=== CONTEXT DATA ===
{context_text}

=== USER QUESTION ===
{query}

=== RESPONSE REQUIREMENTS ===
- Base your answer strictly on the provided context data
- Reference specific policy numbers and document details
- If performing calculations, show your work clearly  
- Format financial figures properly (e.g., $1,234,567.89)
- Use bullet points or tables when presenting multiple items
- If data is incomplete or unavailable, acknowledge this clearly
- Provide actionable insights when appropriate

Please provide a comprehensive and well-formatted response:"""
        
        return prompt
    
    def _post_process_response(self, response_text: str, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Post-process response for better formatting and accuracy"""
        
        # Clean up common formatting issues
        processed = response_text.strip()
        
        # Ensure proper financial formatting
        processed = re.sub(r'\$(\d+)(?=\d)', r'$\1,', processed)
        
        # Add confidence disclaimer if few documents available
        if len(context_docs) < 3:
            processed += f"\n\n*Note: This response is based on {len(context_docs)} relevant document(s) from your insurance database.*"
        
        # Add data freshness reminder for time-sensitive queries
        time_sensitive_keywords = ['current', 'latest', 'recent', 'today', 'now']
        if any(keyword in query.lower() for keyword in time_sensitive_keywords):
            processed += f"\n\n*Note: This information is based on the policies in your database as of the last update.*"
        
        return processed
    
    def _generate_fallback_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate fallback response when LLM fails"""
        
        if not context_docs:
            return f"""I apologize, but I couldn't find any relevant policies in the database to answer your question: "{query}". 

This could be because:
- No policies match your search criteria
- The query terms are too specific or contain misspellings
- The database might be empty or have connectivity issues

Please try:
- Using broader search terms
- Checking policy numbers for accuracy
- Verifying that policies exist in the system"""
        
        # Provide basic information from available documents
        doc_count = len(context_docs)
        sample_policies = [doc.get('policy_metadata', {}).get('policy_number', 'Unknown') 
                          for doc in context_docs[:3]]
        
        return f"""I found {doc_count} potentially relevant policies (e.g., {', '.join(sample_policies[:3])}) but encountered an error generating a complete response to your question: "{query}".

The search returned policies from your database, but I'm unable to process them fully at the moment. Please try rephrasing your question or contact support if this issue persists."""
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token usage (rough approximation)"""
        return len(text.split()) * 1.3  # Rough approximation
    
    def _has_calculations(self, response_text: str) -> bool:
        """Check if response contains calculations"""
        calc_indicators = ['$', '%', 'total', 'sum', 'average', 'calculated', '=', '+', '-', '*', '/']
        return any(indicator in response_text.lower() for indicator in calc_indicators)
    
    def _assess_confidence(self, response_text: str, context_docs: List[Dict[str, Any]]) -> str:
        """Assess confidence level of the response"""
        if not context_docs:
            return "low"
        
        avg_similarity = sum(doc.get('similarity_score', 0) for doc in context_docs) / len(context_docs)
        
        if avg_similarity > 0.7:
            return "high"
        elif avg_similarity > 0.4:
            return "medium"
        else:
            return "low"
    
    async def classify_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query classification with better accuracy"""
        
        classification_prompt = f"""
Analyze this insurance-related query and provide a detailed classification.

Query: "{query}"

Provide a JSON response with the following structure:
{{
    "search_type": "specific_policy" | "aggregate_analysis" | "comparison" | "calculation" | "general_info" | "claims_analysis",
    "requires_calculation": true | false,
    "calculation_type": "sum" | "average" | "percentage" | "count" | "comparison" | null,
    "entities_mentioned": {{
        "policy_numbers": ["list of policy numbers mentioned"],
        "insured_names": ["list of company names mentioned"],
        "amounts": ["list of monetary amounts mentioned"]
    }},
    "date_filters": {{
        "specific_dates": ["YYYY-MM-DD format if mentioned"],
        "date_ranges": ["period descriptions like 'last year', '2024'"],
        "relative_dates": ["relative terms like 'recent', 'current'"]
    }},
    "numerical_filters": {{
        "thresholds": ["over $100,000", "above 50%", etc.],
        "ranges": ["between X and Y"],
        "comparatives": ["highest", "lowest", "maximum", "minimum"]
    }},
    "intent_keywords": ["key action words from the query"],
    "confidence": 0.0 to 1.0,
    "complexity": "simple" | "medium" | "complex"
}}

Examples of classification:
- "What is the total claim amount for policy ABC123?" → specific_policy, requires_calculation: true, calculation_type: "sum"
- "Show me all policies with claims over $100,000" → aggregate_analysis, numerical_filters with threshold
- "Which insured party has the highest treaty rate?" → comparison, comparatives: ["highest"]
- "Calculate the average claim amount by month" → calculation, calculation_type: "average"

Return ONLY valid JSON without any additional text or explanation.
        """
        
        try:
            response = self.model.generate_content(
                classification_prompt,
                generation_config={
                    'temperature': 0.1,  # Lower temperature for more consistent JSON
                    'top_p': 0.8,
                    'max_output_tokens': 1024,
                }
            )
            
            # Clean response text and extract JSON
            response_text = response.text.strip()
            
            # Remove any markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].strip()
            
            # Parse JSON response
            classification = json.loads(response_text)
            
            # Validate and set defaults for missing fields
            classification.setdefault('search_type', 'general_info')
            classification.setdefault('requires_calculation', False)
            classification.setdefault('calculation_type', None)
            classification.setdefault('entities_mentioned', {})
            classification.setdefault('date_filters', {})
            classification.setdefault('numerical_filters', {})
            classification.setdefault('intent_keywords', [])
            classification.setdefault('confidence', 0.5)
            classification.setdefault('complexity', 'medium')
            
            return classification
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed in query classification: {e}")
            logger.error(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
            return self._fallback_classification(query)
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Provide fallback classification using rule-based approach"""
        
        query_lower = query.lower()
        
        # Determine search type
        search_type = "general_info"
        if any(word in query_lower for word in ['policy', 'specific', 'particular']):
            search_type = "specific_policy"
        elif any(word in query_lower for word in ['all', 'show', 'list', 'find']):
            search_type = "aggregate_analysis"
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'highest', 'lowest']):
            search_type = "comparison"
        elif any(word in query_lower for word in ['calculate', 'total', 'sum', 'average']):
            search_type = "calculation"
        elif any(word in query_lower for word in ['claim', 'loss', 'coverage']):
            search_type = "claims_analysis"
        
        # Determine if calculation is required
        requires_calculation = any(word in query_lower for word in [
            'total', 'sum', 'average', 'mean', 'calculate', 'compute', 
            'percentage', '%', 'count', 'how many', 'how much'
        ])
        
        # Determine calculation type
        calculation_type = None
        if requires_calculation:
            if any(word in query_lower for word in ['total', 'sum']):
                calculation_type = "sum"
            elif any(word in query_lower for word in ['average', 'mean']):
                calculation_type = "average"
            elif any(word in query_lower for word in ['percentage', '%', 'percent']):
                calculation_type = "percentage"
            elif any(word in query_lower for word in ['count', 'how many']):
                calculation_type = "count"
        
        # Extract basic entities using regex
        policy_pattern = r'[A-Z]{1,3}/[A-Z0-9]+'
        policy_numbers = re.findall(policy_pattern, query)
        
        amount_pattern = r'\$?[\d,]+(?:\.\d{2})?'
        amounts = re.findall(amount_pattern, query)
        
        return {
            'search_type': search_type,
            'requires_calculation': requires_calculation,
            'calculation_type': calculation_type,
            'entities_mentioned': {
                'policy_numbers': policy_numbers,
                'insured_names': [],
                'amounts': amounts
            },
            'date_filters': {},
            'numerical_filters': {},
            'intent_keywords': query_lower.split(),
            'confidence': 0.3,  # Lower confidence for fallback
            'complexity': 'medium',
            'is_fallback': True
        }
    
    async def generate_sql_query(self, natural_language_query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate SQL query from natural language (advanced feature)"""
        
        sql_prompt = f"""
Given this natural language query about insurance policies, generate a PostgreSQL SQL query.

Query: "{natural_language_query}"

Database schema:
- Table: insurance_policies
- Columns: id, policy_number, insured_name, sum_insured, premium, 
           own_retention_ppn, own_retention_sum_insured, own_retention_premium,
           treaty_ppn, treaty_sum_insured, treaty_premium,
           insurance_period_start_date, insurance_period_end_date,
           policy_metadata (JSONB), created_at, updated_at

Available context documents: {len(context_docs)} policies

Generate a valid PostgreSQL query that would answer this question. 
Include appropriate WHERE clauses, GROUP BY, ORDER BY, and aggregate functions as needed.

Return only the SQL query without any explanation.
        """
        
        try:
            response = self.model.generate_content(
                sql_prompt,
                generation_config={
                    'temperature': 0.2,
                    'top_p': 0.8,
                    'max_output_tokens': 512,
                }
            )
            
            sql_query = response.text.strip()
            
            # Clean up the SQL query
            sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r"\s*```", '', sql_query)
            sql_query = sql_query.strip()
            
            return {
                'sql_query': sql_query,
                'success': True,
                'confidence': 0.7
            }
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {
                'sql_query': None,
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }