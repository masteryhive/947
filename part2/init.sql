-- Create the application user
CREATE USER rag_user WITH PASSWORD 'postgres';

-- Grant permissions to the application user on the database
GRANT ALL PRIVILEGES ON DATABASE insurance_rag TO rag_user;

-- Connect to the database to create extensions and tables
