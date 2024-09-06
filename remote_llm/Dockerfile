FROM python:3.11-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . $APP_HOME

RUN pip install --no-cache-dir -r $APP_HOME/requirements.txt

ENTRYPOINT ["python", "remote_llm.py"]
