FROM python:3.11

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN pip install flit

COPY . $APP_HOME

ENV FLIT_ROOT_INSTALL 1
RUN flit build && flit install

ENTRYPOINT ["api-server"]
