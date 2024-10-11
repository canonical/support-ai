FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . $APP_HOME

RUN apt update && apt install -y python3 python3-pip
RUN pip install --no-cache-dir -r $APP_HOME/requirements.txt

ENTRYPOINT ["python3", "remote_llm.py"]
CMD ["--model_config", "config.yaml"]
