# Insurance RAG API

A production-ready Agentic RAG (Retrieval-Augmented Generation) system for intelligent insurance policy management and querying.

## Features

- **Excel Data Ingestion**: Upload and process insurance policy data from Excel files
- **Intelligent Querying**: Natural language queries with contextual responses
- **Vector Search**: PostgreSQL + pgvector for efficient similarity search
- **Agentic Workflows**: LangGraph-inspired multi-step reasoning
- **Production Ready**: Error handling, monitoring

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Google Gemini API key
- Python 3.11+ (for local development)

<img width="1800" height="1169" alt="Screenshot 2025-09-06 at 00 38 12" src="https://github.com/user-attachments/assets/40948ca8-14e8-4c3d-aa6a-5b7684b933c8" />

### Setup

1. Clone the repository:
```bash
git clone https://github.com/masteryhive/947.git
cd 947
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run PostgreSQL Database:
```bash
docker-compose -f docker-compose.yml up
```

### API Usage

#### Upload Excel Data
```bash
curl -X 'POST' \
  'http://0.0.0.0:8089/v1/chats/ingest-excel/user_123?chunk_size=1000&chunk_overlap=200' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_data.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
```
<img width="1800" height="1169" alt="Screenshot 2025-09-06 at 00 55 46" src="https://github.com/user-attachments/assets/f39c9a23-d709-46ae-89d5-6b41c67459e0" />

#### Query Agent
```bash
curl -X 'POST' \
  'http://0.0.0.0:8089/v1/chats/ask-agent/user_123' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Which insured party has the highest treaty rate?"
}'
```
<img width="1800" height="1169" alt="Screenshot 2025-09-06 at 00 38 37" src="https://github.com/user-attachments/assets/7aa0cec7-8fbf-4d66-81f0-63fbb5317082" />

#### Monitoring
```bash
curl -X 'GET' \
  'http://0.0.0.0:8089/v1/monitor/health' \
  -H 'accept: application/json'
```
<img width="1800" height="1169" alt="Screenshot 2025-09-06 at 00 39 01" src="https://github.com/user-attachments/assets/954c8c85-0470-43d7-a290-f08522e45c8e" />

```bash
curl -X 'GET' \
  'http://0.0.0.0:8089/v1/monitor/metrics' \
  -H 'accept: application/json'
```
<img width="1800" height="1169" alt="Screenshot 2025-09-06 at 00 43 23" src="https://github.com/user-attachments/assets/6c9e1ac6-23fa-4220-bcd0-c28e03f62edc" />

```bash
curl -X 'GET' \
  'http://0.0.0.0:8089/v1/monitor/data-summary' \
  -H 'accept: application/json'
```
<img width="1800" height="1169" alt="Screenshot 2025-09-06 at 00 43 49" src="https://github.com/user-attachments/assets/a974e6d2-0348-40aa-b005-3cb06e0d83f4" />



# AI-engineer-Test

### Overview

This exercise consists of two parts: (1) designing a RAG system design, and (2) implementing a production-ready chat Agentic RAG API.

**Carefully read all exercise instructions. This is your opportunity to demonstrate depth of expertise. Showcase your engineering experience, architectural judgment, and ability to reason about complex design decisions. Submissions that only address surface-level issues may not stand out versus more junior candidates. Take the time to show your thinking and approach at a level appropriate to your experience.**

## Duration: 72hours.

## Technical Exercise Part1: RAG System Design Challenge

OBJECTIVE:
Design a comprehensive RAG system architecture using draw.io, focusing on either:
- RAG Retrieval System, OR  
- RAG Ingestion System

## Technical Exercise PART2: Application Design

OBJECTIVE:
Build a production-ready Agentic RAG FastAPI application that ingests data 
and provides intelligent question-answering capabilities.

## General Guidance & Submission

FORKING AND PULL REQUEST PROCESS:
- Fork the provided repository immediately upon access (do not make public)
- Create a pull request (PR) from your forked repository back to the source
- You may create the PR immediately, even before starting; grading begins only when you indicate completion or the duration elapse.
- Organize your work in the following folder structure:
    ```
    repository-root/
    ├── part_1/                 # System design deliverables
    │   ├── architecture.drawio  # Your draw.io diagram file
    │   ├── architecture.png     # Exported diagram
    │   └── DESIGN_ANALYSIS.md   # Component documentation
    └── part_2/                 # RAG implementation
        ├── src/                # Your FastAPI application
        ├── sample_data.xlsx    # Test Excel file
        ├── Dockerfile          # Local development setup
        └── README.md           # Setup instructions
    ```

TIME MANAGEMENT:
- Recommended time: 48-72 hours total
- You do not need a full production-grade system
- Focus on demonstrating architectural thinking and clean implementation
- Prioritize core functionality over edge cases

PART 1 DELIVERABLES:
- draw.io architecture diagram (.drawio file + exported .png/.pdf)
- DESIGN_ANALYSIS.md with component-by-component analysis
- Follow bullet-point format: 1-3 sentences per point with explicit "why"
- Address: design decisions, challenges/mitigations, trade-offs, alternatives considered

PART 2 DELIVERABLES:
- Working FastAPI application with all core endpoints
- Docker setup for local development
- Sample Excel file with realistic insurance data
- Comprehensive README.md with setup instructions
- API documentation (Swagger auto-generated acceptable)
- Basic error handling and input validation

COMMIT PRACTICES (Part 2):
- Make separate, well-scoped commits for each cohesive change
- Each commit message MUST include:
    - Concise summary line
    - 1-3 sentences explaining what changed and why
    - Avoid vague phrases like "code cleanup"; be specific about the problem solved
- If you identify valuable improvements too large to implement, document in IMPROVEMENTS.md

BONUS CONSIDERATIONS:
- Production-ready error handling and logging
- Performance optimization strategies
- Security implementation (authentication, input sanitization)
- Test coverage for critical paths
- Monitoring and observability hooks
- Scalability considerations documentation

ASSESSMENT FOCUS:
- Architectural soundness and design justification
- Code quality, maintainability, and production readiness
- Ability to handle complex requirements (agentic workflows, data ingestion)
- Problem-solving approach and technical decision-making
- Documentation quality and setup simplicity

SUBMISSION COMPLETION:
- Indicate completion by updating PR description with "READY FOR REVIEW"
- Include brief summary of key architectural decisions
- Highlight any known limitations or trade-offs made due to time constraints
- Ensure all code is runnable with provided setup instructions
