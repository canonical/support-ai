# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies.
COPY requirements.txt $APP_HOME
RUN pip install --no-cache-dir -r $APP_HOME/requirements.txt

# Copy local code to container image
COPY remote_model.py $APP_HOME

ENTRYPOINT ["python", "remote_model.py"]
