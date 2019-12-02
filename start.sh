#!/bin/bash

function manage_app () {
    python manage.py makemigrations
    python manage.py migrate
}

function start_development() {
    # use django runserver as development server here.
    manage_app
    kill -9 $(lsof -t -i:8000) || echo "OK"
    python manage.py runserver 0.0.0.0:8000
}

function start_production() {
    # use gunicorn for production server here
    manage_app
    gunicorn source.wsgi -w 4 -b 0.0.0.0:8000 --chdir=/app --log-file -
}

function start_elk() {
  kill -9 $(lsof -t -i:5044) || echo "ok"  &&  kill -9 $(lsof -t -i:5601) || echo "ok" &&  kill -9 $(lsof -t -i:9200) || echo "ok" && docker-compose -f elk/elk-docker/docker-compose.yml down -v && docker-compose -f elk/elk-docker/docker-compose.yml up -d
docker container stop $(docker container ls -aq) &&
docker container rm $(docker container ls -aq) &&
docker volume prune &&
docker network prune &&
docker image prune &&
docker image prune -a || echo "ok"

}

function clear_docker() {

docker container stop $(docker container ls -aq) || echo "ok" &&
docker container rm $(docker container ls -aq) || echo "ok" &&
docker volume prune   || echo "ok" &&
docker network prune   || echo "ok" &&
docker image prune  || echo "ok" &&
docker image prune -a || echo "ok"


}

function clear() {
  systemctl stop elasticsearch &&
    systemctl stop kibana &&
      systemctl stop logstash &&
   kill -9 $(lsof -t -i:5044) || echo "ok"  &&
 kill -9 $(lsof -t -i:5601) || echo "ok" &&
 kill -9 $(lsof -t -i:9200) || echo "ok" &&
 kill -9 $(lsof -t -i:80) || echo "ok" &&
 kill -9 $(lsof -t -i:8000) || echo "ok" &&
docker container stop $(docker container ls -aq) || echo "ok" &&
docker container rm $(docker container ls -aq) || echo "ok" &&
docker volume prune   || echo "ok" &&
docker network prune   || echo "ok"


}

function clear_ports() {
  systemctl stop elasticsearch &&
    systemctl stop kibana &&
      systemctl stop logstash &&
   kill -9 $(lsof -t -i:5044) || echo "ok"  &&
 kill -9 $(lsof -t -i:5601) || echo "ok" &&
 kill -9 $(lsof -t -i:9200) || echo "ok" &&
 kill -9 $(lsof -t -i:80) || echo "ok" &&
 kill -9 $(lsof -t -i:8000) || echo "ok"
}


function clean_docker(){

docker volume rm $(docker volume ls -qf dangling=true)
docker volume ls -qf dangling=true | xargs -r docker volume rm


docker network ls
docker network ls | grep "bridge"
docker network rm $(docker network ls | grep "bridge" | awk '/ / { print $1 }')

docker ps
docker ps -a
docker rm $(docker ps -qa --no-trunc --filter "status=exited")


}

function init() {
docker-compose -f docker-compose.yml down -v &&
docker-compose -f docker-compose.yml up -d --build &&
docker-compose -f docker-compose.yml exec web python manage.py migrate --noinput &&
docker-compose -f docker-compose.yml exec web python manage.py collectstatic --no-input&&
docker-compose up -d --no-deps --build web
}

function  start_docker() {
docker-compose -f docker-compose.yml down -v || echo "ok" &&
docker-compose -f docker-compose.yml up -d
}
# Check if the function exists (bash specific)
if declare -f "$1" > /dev/null
then
  # call arguments verbatim
  "$@"
else
  # Show a helpful error
  echo "'$1' is not a known function name" >&2
  exit 1
fi
