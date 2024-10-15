FROM ubuntu:22.04

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt update && apt install -y python3 python3-pip git
RUN pip install flit

COPY . $APP_HOME

ENV FLIT_ROOT_INSTALL 1
RUN flit build && flit install

ENTRYPOINT ["api-server"]
