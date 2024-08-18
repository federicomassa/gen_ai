### Setup the repo
`./setup.sh`

### Building the docker container
`docker compose -f docker-compose.gpu.yml build`

### Running the docker container
It's recommended to install the VSCode "docker" extension and open the container as a remote window.

To run the container:
`docker compose -f docker-compose.gpu.yml up`
