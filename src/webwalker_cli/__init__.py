from webwalker.logging import setup_rich_logging
import typer

from .experiments import experiments_app
from .utils import utils_app

setup_rich_logging("webwalker", allow=["__main__"])
app = typer.Typer(help="webwalker")

app.add_typer(experiments_app, name="experiments", help="experiment runners")
app.add_typer(utils_app, name="utils", help="utilities")

if __name__ == "__main__":
	app()
