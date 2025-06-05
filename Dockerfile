FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Java and procps
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     openjdk-17-jdk \
#     procps \
#     wget \
#     gnupg \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential \
    procps \
    gnupg \
    && rm -rf /var/lib/apt/lists/*
#     && rm -rf /var/lib/apt/lists/*
# Verify Java installation
# RUN java -version

# Set JAVA_HOME
# ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# ENV PATH=$PATH:$JAVA_HOME/bin

# Verify JAVA_HOME
# RUN echo $JAVA_HOME

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenAI package
RUN pip install --no-cache-dir openai

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download Google Search Results package
RUN pip install --no-cache-dir google-search-results

# Download MCP (Market Center Platform)
## 安装 MCP 及其 CLI 工具
RUN pip install --no-cache-dir "mcp[cli]"

# 复制项目文件（如果有）
COPY . .


# Install package in development mode
RUN pip install -e .

# Set PySpark environment variables
# ENV SPARK_HOME=/usr/local/lib/python3.9/site-packages/pyspark
# ENV PATH=$PATH:$SPARK_HOME/bin
# ENV PYSPARK_PYTHON=/usr/local/bin/python
# ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python
# ENV SPARK_LOCAL_IP=0.0.0.0
# ENV SPARK_PUBLIC_DNS=localhost

# Create necessary directories
# RUN mkdir -p /tmp/spark-events

# # Expose ports for Spark
# EXPOSE 8501 7077 8080 8081

# Command to run the application
CMD ["streamlit", "run", "dashboard/advanced_dashboard.py"]
