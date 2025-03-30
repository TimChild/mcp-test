import logging.config

from dependency_injector import containers, providers
from mcp_client import MultiMCPClient


class Core(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


class Adapters(containers.DeclarativeContainer):
    config = providers.Configuration()

    mcp_client = providers.Factory(
        MultiMCPClient,
        connections=config.mcp_servers,
    )


class Application(containers.DeclarativeContainer):
    config = providers.Configuration(yaml_files=["config.yml"])

    core = providers.Container(
        Core,
        config=config.core,
    )

    adapters = providers.Container(
        Adapters,
        config=config.adapters,
    )
