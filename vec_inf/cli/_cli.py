"""Command line interface for Vector Inference.

This module provides the command-line interface for interacting with Vector
Inference services, including model launching, status checking, metrics
monitoring, and shutdown operations.

Commands
--------
launch
    Launch a model on the cluster
status
    Check the status of a running model
shutdown
    Stop a running model
list
    List available models or get specific model configuration
metrics
    Stream real-time performance metrics
"""

import json
import time
from typing import Optional, Union

import click
from rich.console import Console
from rich.live import Live

from vec_inf.cli._helper import (
    BatchLaunchResponseFormatter,
    LaunchResponseFormatter,
    ListCmdDisplay,
    MetricsResponseFormatter,
    StatusResponseFormatter,
)
from vec_inf.client import LaunchOptions, VecInfClient


CONSOLE = Console()


@click.group()
def cli() -> None:
    """Vector Inference CLI."""
    pass


@cli.command("launch")
@click.argument("model-name", type=str, nargs=1)
@click.option("--model-family", type=str, help="The model family")
@click.option("--model-variant", type=str, help="The model variant")
@click.option(
    "--partition",
    type=str,
    help="Type of compute partition",
)
@click.option(
    "--num-nodes",
    type=int,
    help="Number of nodes to use, default to suggested resource allocation for model",
)
@click.option(
    "--gpus-per-node",
    type=int,
    help="Number of GPUs/node to use, default to suggested resource allocation for model",
)
@click.option(
    "--account",
    type=str,
    help="Charge resources used by this job to specified account.",
)
@click.option(
    "--qos",
    type=str,
    help="Quality of service",
)
@click.option(
    "--exclude",
    type=str,
    help="Exclude certain nodes from the resources granted to the job",
)
@click.option(
    "--nodelist",
    type=str,
    help="Request a specific list of nodes for deployment",
)
@click.option(
    "--bind",
    type=str,
    help="Additional binds for the singularity container as a comma separated list of bind paths",
)
@click.option(
    "--time",
    type=str,
    help="Time limit for job, this should comply with QoS limits",
)
@click.option(
    "--venv",
    type=str,
    help="Path to virtual environment",
)
@click.option(
    "--log-dir",
    type=str,
    help="Path to slurm log directory",
)
@click.option(
    "--model-weights-parent-dir",
    type=str,
    help="Path to parent directory containing model weights",
)
@click.option(
    "--vllm-args",
    type=str,
    help="vLLM engine arguments to be set, use the format as specified in vLLM documentation and separate arguments with commas, e.g. --vllm-args '--max-model-len=8192,--max-num-seqs=256,--enable-prefix-caching'",
)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def launch(
    model_name: str,
    **cli_kwargs: Optional[Union[str, int, float, bool]],
) -> None:
    """Launch a model on the cluster.

    Parameters
    ----------
    model_name : str
        Name of the model to launch
    **cli_kwargs : dict
        Additional launch options including:
        - model_family : str, optional
            Family/architecture of the model
        - model_variant : str, optional
            Specific variant of the model
        - partition : str, optional
            Type of compute partition
        - num_nodes : int, optional
            Number of nodes to use
        - gpus_per_node : int, optional
            Number of GPUs per node
        - account : str, optional
            Charge resources used by this job to specified account
        - qos : str, optional
            Quality of service tier
        - exclude : str, optional
            Exclude certain nodes from the resources granted to the job
        - nodelist : str, optional
            Request a specific list of nodes for deployment
        - bind : str, optional
            Additional binds for the singularity container
        - time : str, optional
            Time limit for job
        - venv : str, optional
            Path to virtual environment
        - log_dir : str, optional
            Path to SLURM log directory
        - model_weights_parent_dir : str, optional
            Path to model weights directory
        - vllm_args : str, optional
            vLLM engine arguments
        - json_mode : bool, optional
            Output in JSON format

    Raises
    ------
    click.ClickException
        If launch fails for any reason
    """
    try:
        # Convert cli_kwargs to LaunchOptions
        json_mode = cli_kwargs["json_mode"]
        del cli_kwargs["json_mode"]

        launch_options = LaunchOptions(**cli_kwargs)  # type: ignore

        # Start the client and launch model inference server
        client = VecInfClient()
        launch_response = client.launch_model(model_name, launch_options)

        # Display launch information
        if json_mode:
            click.echo(json.dumps(launch_response.config))
        else:
            launch_formatter = LaunchResponseFormatter(
                model_name, launch_response.config
            )
            launch_info_table = launch_formatter.format_table_output()
            CONSOLE.print(launch_info_table)

    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Launch failed: {str(e)}") from e


@cli.command("batch-launch")
@click.argument("model-names", type=str, nargs=-1)
@click.option(
    "--batch-config",
    type=str,
    help="Model configuration for batch launch",
)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def batch_launch(
    model_names: tuple[str, ...],
    batch_config: Optional[str] = None,
    json_mode: Optional[bool] = False,
) -> None:
    """Launch multiple models in a batch.

    Parameters
    ----------
    model_names : tuple[str, ...]
        Names of the models to launch
    batch_config : str
        Model configuration for batch launch
    json_mode : bool, default=False
        Whether to output in JSON format

    Raises
    ------
    click.ClickException
        If batch launch fails
    """
    try:
        # Start the client and launch models in batch mode
        client = VecInfClient()
        batch_launch_response = client.batch_launch_models(
            list(model_names), batch_config
        )

        # Display batch launch information
        if json_mode:
            click.echo(batch_launch_response.config)
        else:
            batch_launch_formatter = BatchLaunchResponseFormatter(
                batch_launch_response.config
            )
            batch_launch_info_table = batch_launch_formatter.format_table_output()
            CONSOLE.print(batch_launch_info_table)

    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Batch launch failed: {str(e)}") from e


@cli.command("status")
@click.argument("slurm_job_id", type=str, nargs=1)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def status(slurm_job_id: str, json_mode: bool = False) -> None:
    """Get the status of a running model on the cluster.

    Parameters
    ----------
    slurm_job_id : str
        ID of the SLURM job to check
    json_mode : bool, default=False
        Whether to output in JSON format

    Raises
    ------
    click.ClickException
        If status check fails
    """
    try:
        # Start the client and get model inference server status
        client = VecInfClient()
        status_response = client.get_status(slurm_job_id)
        # Display status information
        status_formatter = StatusResponseFormatter(status_response)
        if json_mode:
            status_formatter.output_json()
        else:
            status_info_table = status_formatter.output_table()
            CONSOLE.print(status_info_table)

    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Status check failed: {str(e)}") from e


@cli.command("shutdown")
@click.argument("slurm_job_id", type=str, nargs=1)
def shutdown(slurm_job_id: str) -> None:
    """Shutdown a running model on the cluster.

    Parameters
    ----------
    slurm_job_id : str
        ID of the SLURM job to shut down

    Raises
    ------
    click.ClickException
        If shutdown operation fails
    """
    try:
        client = VecInfClient()
        client.shutdown_model(slurm_job_id)
        click.echo(f"Shutting down model with Slurm Job ID: {slurm_job_id}")
    except Exception as e:
        raise click.ClickException(f"Shutdown failed: {str(e)}") from e


@cli.command("list")
@click.argument("model-name", required=False)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def list_models(model_name: Optional[str] = None, json_mode: bool = False) -> None:
    """List all available models, or get default setup of a specific model.

    Parameters
    ----------
    model_name : str, optional
        Name of specific model to get information for
    json_mode : bool, default=False
        Whether to output in JSON format

    Raises
    ------
    click.ClickException
        If list operation fails
    """
    try:
        # Start the client
        client = VecInfClient()
        list_display = ListCmdDisplay(CONSOLE, json_mode)
        if model_name:
            model_config = client.get_model_config(model_name)
            list_display.display_single_model_output(model_config)
        else:
            model_infos = client.list_models()
            list_display.display_all_models_output(model_infos)
    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"List models failed: {str(e)}") from e


@cli.command("metrics")
@click.argument("slurm_job_id", type=str, nargs=1)
def metrics(slurm_job_id: str) -> None:
    """Stream real-time performance metrics from the model endpoint.

    Parameters
    ----------
    slurm_job_id : str
        ID of the SLURM job to monitor

    Raises
    ------
    click.ClickException
        If metrics collection fails

    Notes
    -----
    This command continuously streams metrics with a 2-second refresh interval
    until interrupted. If metrics are not available, it will display status
    information instead.
    """
    try:
        # Start the client and get inference server metrics
        client = VecInfClient()
        metrics_response = client.get_metrics(slurm_job_id)
        metrics_formatter = MetricsResponseFormatter(metrics_response.metrics)

        # Check if metrics response is ready
        if isinstance(metrics_response.metrics, str):
            metrics_formatter.format_failed_metrics(metrics_response.metrics)
            CONSOLE.print(metrics_formatter.table)
            return

        with Live(refresh_per_second=1, console=CONSOLE) as live:
            while True:
                metrics_response = client.get_metrics(slurm_job_id)
                metrics_formatter = MetricsResponseFormatter(metrics_response.metrics)

                if isinstance(metrics_response.metrics, str):
                    # Show status information if metrics aren't available
                    metrics_formatter.format_failed_metrics(metrics_response.metrics)
                else:
                    metrics_formatter.format_metrics()

                live.update(metrics_formatter.table)
                time.sleep(2)
    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Metrics check failed: {str(e)}") from e


@cli.command("cleanup")
@click.option("--log-dir", type=str, help="Path to SLURM log directory")
@click.option("--model-family", type=str, help="Filter by model family")
@click.option("--model-name", type=str, help="Filter by model name")
@click.option(
    "--job-id", type=int, help="Only remove logs with this exact SLURM job ID"
)
@click.option(
    "--before-job-id",
    type=int,
    help="Remove logs with job ID less than this value",
)
@click.option("--dry-run", is_flag=True, help="List matching logs without deleting")
def cleanup_logs_cli(
    log_dir: Optional[str],
    model_family: Optional[str],
    model_name: Optional[str],
    job_id: Optional[int],
    before_job_id: Optional[int],
    dry_run: bool,
) -> None:
    """Clean up log files based on optional filters.

    Parameters
    ----------
    log_dir : str or Path, optional
        Root directory containing log files. Defaults to ~/.vec-inf-logs.
    model_family : str, optional
        Only delete logs for this model family.
    model_name : str, optional
        Only delete logs for this model name.
    job_id : int, optional
        If provided, only match directories with this exact SLURM job ID.
    before_job_id : int, optional
        If provided, only delete logs with job ID less than this value.
    dry_run : bool
        If True, return matching files without deleting them.
    """
    try:
        client = VecInfClient()
        matched = client.cleanup_logs(
            log_dir=log_dir,
            model_family=model_family,
            model_name=model_name,
            job_id=job_id,
            before_job_id=before_job_id,
            dry_run=dry_run,
        )

        if not matched:
            if dry_run:
                click.echo("Dry run: no matching log directories found.")
            else:
                click.echo("No matching log directories were deleted.")
        elif dry_run:
            click.echo(f"Dry run: {len(matched)} directories would be deleted:")
            for f in matched:
                click.echo(f"  - {f}")
        else:
            click.echo(f"Deleted {len(matched)} log directory(ies).")
    except Exception as e:
        raise click.ClickException(f"Cleanup failed: {str(e)}") from e


if __name__ == "__main__":
    cli()
