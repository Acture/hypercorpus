from hypercorpus.logging import setup_rich_logging
import typer

from .baselines import baselines_app
from .datasets import datasets_app
from .experiments import experiments_app
from .utils import utils_app

setup_rich_logging("hypercorpus", allow=["__main__"])
app = typer.Typer(help="hypercorpus")

app.add_typer(baselines_app, name="baselines", help="baseline wrappers")
app.add_typer(datasets_app, name="datasets", help="dataset fetchers")
app.add_typer(experiments_app, name="experiments", help="experiment runners")
app.add_typer(utils_app, name="utils", help="utilities")

if __name__ == "__main__":
	app()
