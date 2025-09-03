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