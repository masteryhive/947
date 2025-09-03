## PART 1: RAG System Design Challenge

OBJECTIVE:
Design a comprehensive RAG system architecture using draw.io, focusing on either:
- RAG Retrieval System, OR  
- RAG Ingestion System

DELIVERABLES:

1. CREATE A DRAW.IO DIAGRAM showing your chosen system architecture
    - Include all major components and their relationships
    - Show data flow directions with labeled arrows
    - Use appropriate shapes and colors for different component types
    - Export as PNG and include in your submission

2. COMPONENT DOCUMENTATION:
    For each component in your diagram, provide:
    
    a. Component Name
    b. Purpose: What this component does
    c. Limitations: Technical constraints, bottlenecks, failure points
    d. Necessity: Critical/Important/Optional and why
    e. Alternatives: Other options you considered

EXAMPLE COMPONENT DOCUMENTATION FORMAT:

Component Name: Vector Database (Pinecone/Weaviate)
Purpose: Stores document embeddings for semantic similarity search
Limitations: 
- Limited by embedding dimensions and index size
- Query latency increases with dataset size
- Requires periodic index optimization
- Cost scales with storage and queries
Necessity: Critical - Core requirement for semantic retrieval
Alternatives: Elasticsearch with dense vectors, FAISS, ChromaDB

EVALUATION CRITERIA:
- Architectural completeness and scalability considerations
- Understanding of component interactions and data flow
- Realistic assessment of limitations and trade-offs
- Justification of design choices
- Production readiness considerations (monitoring, error handling, etc.)
- Addressing of security and compliance requirements
- Show of failover and redundancy mechanisms