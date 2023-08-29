# Set service name
SERVICE_NAME = railroad
SERVICE_NAME_BASE = ${SERVICE_NAME}-base
SERVICE_NAME_DEV = ${SERVICE_NAME}-dev
SERVICE_NAME_DEV_MAC = ${SERVICE_NAME}-dev-mac

# Set working & current path 
DATA_PATH = /data/railroad
WORKDIR_PATH = /opt/${SERVICE_NAME}
CURRENT_PATH = $(shell pwd)

# Set command
COMMAND_BASE = /bin/bash
COMMAND_PROD = /bin/bash
COMMAND_DEV = /bin/zsh
COMMAND_DEV_MAC = /bin/zsh
COMMAND_RESEARCH = /bin/zsh

# Get IDs
GID = $(shell id -g)
UID = $(shell id -u)
GRP = $(shell id -gn)
USR = $(shell id -un)

# Get docker image name
IMAGE_NAME_BASE = ${SERVICE_NAME}-base:1.0.0
IMAGE_NAME_DEV = ${SERVICE_NAME}-${USR}-dev:1.0.0
IMAGE_NAME_DEV_MAC = ${SERVICE_NAME}-${USR}-dev-mac:1.0.0

# Get docker container name
CONTAINER_NAME_DEV = ${SERVICE_NAME}-${USR}-dev
CONTAINER_NAME_DEV_MAC = ${SERVICE_NAME}-${USR}-dev-mac

# Docker build context
DOCKER_BUILD_CONTEXT_PATH = ./dockerfile
DOCKERFILE_NAME_BASE = dockerfile_base
DOCKERFILE_NAME_DEV = dockerfile_dev
DOCKERFILE_NAME_DEV_MAC = dockerfile_dev_mac
DOCKER_COMPOSE_NAME = docker_compose.yaml
ENV_FILE_PATH = ${DOCKER_BUILD_CONTEXT_PATH}/.env

# Set enviornments
ENV_TEXT = "$\
	GID=${GID}\n$\
	UID=${UID}\n$\
	GRP=${GRP}\n$\
	USR=${USR}\n$\
	IMAGE_NAME_BASE=${IMAGE_NAME_BASE}\n$\
	IMAGE_NAME_DEV=${IMAGE_NAME_DEV}\n$\
	IMAGE_NAME_DEV_MAC=${IMAGE_NAME_DEV_MAC}\n$\
	CONTAINER_NAME_DEV=${CONTAINER_NAME_DEV}\n$\
	CONTAINER_NAME_DEV_MAC=${CONTAINER_NAME_DEV_MAC}\n$\
	DATA_PATH=${DATA_PATH}\n$\
	WORKDIR_PATH=${WORKDIR_PATH}\n$\
	CURRENT_PATH=${CURRENT_PATH}\n$\
	DOCKER_BUILD_CONTEXT_PATH=${DOCKER_BUILD_CONTEXT_PATH}\n$\
	DOCKERFILE_NAME_BASE=${DOCKERFILE_NAME_BASE}\n$\
	DOCKERFILE_NAME_DEV=${DOCKERFILE_NAME_DEV}\n$\
	DOCKERFILE_NAME_DEV_MAC=${DOCKERFILE_NAME_DEV_MAC}\n$\
	DOCKER_COMPOSE_NAME=${DOCKER_COMPOSE_NAME}\n$\"
${ENV_FILE_PATH}:
	printf ${ENV_TEXT} >> ${ENV_FILE_PATH}

# env  
env: ${ENV_FILE_PATH}

# mlflow 
pull-mlflow:
	docker pull ghcr.io/mlflow/mlflow:v2.0.1
up-mlflow:
	docker run -p 5000:5000 --name mlflow-server -d -v ${CURRENT_PATH}:/mlflow ghcr.io/mlflow/mlflow:v2.3.0 mlflow server --host 0.0.0.0
down-mlflow:
	docker rm -f mlflow-server

# base docker 
build-base:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} up --build -d ${SERVICE_NAME_BASE}
up-base:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} up -d ${SERVICE_NAME_BASE}
exec-base: 
	DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} exec ${SERVICE_NAME_BASE} ${COMMAND_BASE}
start-base:
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} start ${SERVICE_NAME_BASE}
down-base: 
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} down
run-base: 
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} run ${SERVICE_NAME_BASE} ${COMMAND_BASE}
ls-base: 
	docker compose ls -a

# development docker
build-dev: build-base
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} up --build -d ${SERVICE_NAME_DEV}
up-dev:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} up -d ${SERVICE_NAME_DEV}
exec-dev: 
	DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} exec ${SERVICE_NAME_DEV} ${COMMAND_DEV}
start-dev:
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} start ${SERVICE_NAME_DEV}
down-dev: 
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} down
run-dev: 
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} run ${SERVICE_NAME_DEV} ${COMMAND_DEV}
ls-dev: 
	docker compose ls -a

# development docker for mac (with no gpu)
build-dev-mac: build-base
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} up --build -d ${SERVICE_NAME_DEV_MAC}
up-dev-mac:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} up -d ${SERVICE_NAME_DEV_MAC}
exec-dev-mac: 
	DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} exec ${SERVICE_NAME_DEV_MAC} ${COMMAND_DEV_MAC}
start-dev-mac:
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} start ${SERVICE_NAME_DEV_MAC}
down-dev-mac: 
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} down
run-dev-mac: 
	docker compose -f ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME} run ${SERVICE_NAME_DEV_MAC} ${COMMAND_DEV_MAC}
ls-dev-mac: 
	docker compose ls -a
