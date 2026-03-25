import logging

from rich.console import Console

from hypercorpus.logging import (
	DashboardLogBuffer,
	DashboardProgressState,
	create_progress,
	dashboard_session,
	setup_rich_logging,
)


def test_dashboard_log_buffer_captures_first_party_info_and_third_party_warnings():
	setup_rich_logging("hypercorpus", console=Console(record=True), force=True)
	first_party = logging.getLogger("hypercorpus.experiments")
	third_party = logging.getLogger("urllib3.connectionpool")
	log_buffer = DashboardLogBuffer()

	with dashboard_session(
		log_buffer=log_buffer, progress_state=DashboardProgressState()
	):
		first_party.info("first party info")
		third_party.info("third party info")
		third_party.warning("third party warning")

	rendered = [entry.rendered for entry in log_buffer.tail()]
	assert any("first party info" in line for line in rendered)
	assert any("third party warning" in line for line in rendered)
	assert not any("third party info" in line for line in rendered)


def test_dashboard_progress_adapter_tracks_tasks():
	progress_state = DashboardProgressState()

	with dashboard_session(progress_state=progress_state):
		with create_progress(transient=True) as progress:
			task_id = progress.add_task("evaluate", total=10, detail="q1")
			progress.advance(task_id, 3)
			progress.update(
				task_id, description="evaluate cases", detail="q2", advance=2
			)

	task = progress_state.latest_task()
	assert task is not None
	assert task.description == "evaluate cases"
	assert task.completed == 5
	assert task.total == 10
	assert task.detail == "q2"
