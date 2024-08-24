import os
import click
import warnings
from commands import Commands

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

commands = Commands()


@click.group()
def cli():
    """RAG Program CLI"""
    pass


@cli.command()
@click.argument('pdf_dir')
def init(pdf_dir):
    """Initialize the vector store with the given PDF directory."""
    commands.init(pdf_dir)


@cli.command()
@click.argument('query_text')
def query(query_text):
    """Send a query to the system."""
    result = commands.query(query_text)
    print(f"Response: {result}")


@cli.command()
def restore():
    """Restore (clear) the vector store."""
    commands.restore()


@cli.command()
def exit():
    """Exit the CLI."""
    print("Exiting the CLI.")
    raise SystemExit


if __name__ == '__main__':
    cli()
