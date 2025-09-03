
## PART 2: Build Production Agentic RAG System

OBJECTIVE:
Build a production-ready Agentic RAG FastAPI application that ingests data 
and provides intelligent question-answering capabilities.

TECH STACK REQUIREMENTS:
- FastAPI for API framework
- Google Gemini for LLM capabilities  
- LangGraph for agentic workflows
- PostgreSQL for data storage

DATA REQUIREMENTS:

1. EXCEL INGESTION:
    - Accept Excel file uploads via API endpoint
    - Parse and validate data against required schema
    - Handle data cleaning and transformation

2. DATABASE SCHEMA (PostgreSQL):
    {
        "policy_number": "string",
        "insured_name": "string", 
        "sum_insured": "double",
        "premium": "double",
        "own_retention_ppn": "double",
        "own_retention_sum_insured": "double",
        "own_retention_premium": "double",
        "treaty_ppn": "double",
        "treaty_sum_insured": "double",
        "treaty_premium": "double",
        "insurance_period_start_date": "timestamp",
        "insurance_period_end_date": "timestamp"
    }

CORE FEATURES TO IMPLEMENT:

1. DATA INGESTION API
    POST /ingest-excel
    - Accept file upload
    - Validate Excel structure
    - Dynamically Transform data to match schema(split PERIOD OF INSURANCE into insurance_period_start_date and insurance_period_end_date)
    - Store in PostgreSQL
    - Return ingestion summary and errors

2. AGENTIC RAG QUERY API  
    POST /query
    - Accept natural language questions
    - Use LangGraph for multi-step reasoning
    - Query database intelligently based on question
    - Generate contextual responses using Gemini
    - Return answer with source citations

3. HEALTH & MONITORING
    GET /health - System health check
    GET /metrics - Basic usage metrics
    GET /data-summary - Database statistics

PRODUCTION REQUIREMENTS:

✅ Error Handling & Validation
- Comprehensive input validation
- Graceful error responses with proper HTTP codes
- Logging for debugging and monitoring

✅ Database Management
- Connection pooling
- Migration support
- Query optimization
- Transaction management

✅ LLM Integration
- Proper API key management
- Rate limiting and retry logic
- Response streaming for long queries
- Cost optimization strategies

✅ Security
- Input sanitization
- SQL injection prevention  
- API authentication (basic auth minimum)
- Environment variable configuration

✅ Performance
- Async/await patterns
- Caching strategies
- Efficient database queries
- Memory management for large files

✅ Documentation
- OpenAPI/Swagger documentation
- README with setup instructions
- Docker containerization
- Environment configuration guide

SAMPLE QUESTIONS YOUR SYSTEM SHOULD HANDLE:
- "What is the total claim amount for policy ABC123?"
- "Show me all policies with claims over $100,000"
- "Which insured party has the highest treaty rate?"
- "Find policies where the loss date is after the insurance period end date"
- "Calculate the average claim amount by month"
- "What percentage of total claims does the treaty cover?"


EVALUATION CRITERIA:
- Code quality and organization
- Production readiness features
- Error handling robustness  
- LLM integration effectiveness
- Database design and performance
- API design and documentation
- Testing coverage
- Deployment readiness

SUBMISSION FORMAT:
- Complete code in the part2 folder


TIME LIMIT: 4-6 hours
