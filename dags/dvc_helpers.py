"""Helper functions for working with DVC in Airflow DAG."""

import os
import subprocess


def run_command(command: str, cwd: str | None = None) -> tuple[int, str, str]:
    """
    Executes a command and returns the return code, stdout, and stderr.

    Args:
        command: Command to execute
        cwd: Working directory (optional)

    Returns:
        Tuple of (return code, stdout, stderr)
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def check_dvc_initialized(project_path: str) -> bool:
    """
    Checks if DVC is initialized in the project.

    Args:
        project_path: Path to the project
    Returns:
        True if DVC is initialized, otherwise False
    """
    return os.path.exists(os.path.join(project_path, ".dvc"))


def initialize_dvc(project_path: str) -> tuple[bool, str]:
    """
    Initializes DVC in the project, if it is not already initialized.

    Args:
        project_path: Path to the project
    Returns:
        Tuple of (success, message)
    """
    if check_dvc_initialized(project_path):
        return True, "DVC already initialized"

    returncode, stdout, stderr = run_command("dvc init", cwd=project_path)
    if returncode != 0:
        return False, f"Failed to initialize DVC: {stderr}"
    return True, "DVC initialized successfully"


def setup_dvc_remote(
    project_path: str, remote_name: str, remote_url: str
) -> tuple[bool, str]:
    """
    Configures the DVC remote storage.

    Args:
        project_path: Path to the project
        remote_name: Name of the remote storage
        remote_url: URL of the remote storage
    Returns:
        Tuple of (success, message)
    """
    cmd_add = f"dvc remote add {remote_name} {remote_url}"
    returncode, stdout, stderr = run_command(cmd_add, cwd=project_path)
    if returncode != 0:
        cmd_modify = f"dvc remote modify {remote_name} url {remote_url}"
        returncode, stdout, stderr = run_command(cmd_modify, cwd=project_path)
        if returncode != 0:
            return False, f"Failed to setup DVC remote: {stderr}"

    cmd_default = f"dvc remote default {remote_name}"
    returncode, stdout, stderr = run_command(cmd_default, cwd=project_path)
    if returncode != 0:
        return False, f"Failed to set default DVC remote: {stderr}"

    return True, "DVC remote setup successfully"


def track_file_with_dvc(project_path: str, file_path: str) -> tuple[bool, str]:
    """
    Adds a file under DVC control.

    Args:
        project_path: Path to the project
        file_path: Relative path to the file from the project root
    Returns:
        Tuple of (success, message)
    """
    abs_file_path = os.path.join(project_path, file_path)
    if not os.path.exists(abs_file_path):
        return False, f"File {abs_file_path} does not exist"

    returncode, stdout, stderr = run_command(f"dvc add {file_path}", cwd=project_path)
    if returncode != 0:
        return False, f"Failed to add file to DVC: {stderr}"
    return True, f"File {file_path} added to DVC successfully"


def push_to_dvc_remote(project_path: str) -> tuple[bool, str]:
    """
    Pushes data to the DVC remote storage.

    Args:
        project_path: Path to the project
    Returns:
        Tuple of (success, message)
    """
    returncode, stdout, stderr = run_command("dvc push", cwd=project_path)
    if returncode != 0:
        return False, f"Failed to push data to DVC remote: {stderr}"
    return True, "Data pushed to DVC remote successfully"
