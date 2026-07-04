# Dockerfile for a Kestrion agent.
# Place this in your project root alongside your agent script.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir kestrion[anthropic]

# Copy agent code
COPY agent.py ./

# Store lives at /data — mount a volume here in production.
ENV KESTRION_STORE_URL="sqlite:////data/agent.db"

CMD ["python3", "agent.py"]
