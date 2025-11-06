ifneq (,$(wildcard ./.env))
	include .env
    export
endif

# Virtual Environment Management
ifeq ($(OS), Windows_NT)
	BIN = ${VENV}/Scripts/
else
	BIN = ${VENV}/bin/
endif
VENV := .venv
ENV_LOCAL := environment-local.yml
ENV_G5000 := environment-g5000.yml
ifeq (${ENVIRONMENT}, local)
	ENVIRONMENT_FILE = ${ENV_LOCAL}
else ifeq (${ENVIRONMENT}, grid5000)
	ENVIRONMENT_FILE = ${ENV_G5000}
endif
# Binaries
PIP:=${BIN}/pip

# Logs
LOGS:=logs

all:
	@${MAKE} install

clean:
	@${MAKE} clean-venv
	@${MAKE} clean-logs

clean-venv:
	@${CONDA_EXE} env remove --prefix ${VENV}

clean-logs:
	@rm logs/*

${VENV}:
	@${CONDA_EXE} env create --file=${ENVIRONMENT_FILE} --prefix=${VENV}

venv: ${VENV}

${LOGS}:
	@mkdir -p ${LOGS}

install: ${VENV} ${LOGS}
	@${PIP} install -e .

# GRID 5000 -----------------------------------------
g5k-import-%:
	rsync -avzP ${G5K_LOGIN}@rennes.g5k:${G5K_STORAGE}/$* ${G5K_IMPORT_STORAGE}
oar-stop-all:
	oarstat -u | awk 'NR>2 {print substr($$1, 1, 7)}' | xargs -r oardel
# ---------------------------------------------------