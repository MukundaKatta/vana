"""Report generation: formatted terminal output and matplotlib charts."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vana.models import Alert, DeforestationEvent
from vana.analysis.trend import TrendResult


console = Console()


def print_event_table(events: Sequence[DeforestationEvent]) -> None:
    """Print a Rich table summarising deforestation events."""
    table = Table(title="Deforestation Events", show_lines=True)
    table.add_column("Period", style="cyan")
    table.add_column("Region", style="white")
    table.add_column("Hectares Lost", justify="right", style="red")
    table.add_column("Mean NDVI Drop", justify="right", style="yellow")
    table.add_column("Pixels", justify="right")

    for ev in events:
        table.add_row(
            f"{ev.start_date:%Y-%m-%d} -> {ev.end_date:%Y-%m-%d}",
            ev.region_id,
            f"{ev.hectares_lost:.2f}",
            f"{ev.mean_ndvi_drop:.4f}",
            str(ev.affected_pixels),
        )
    console.print(table)


def print_alerts(alerts: Sequence[Alert]) -> None:
    """Print alerts as styled Rich panels."""
    if not alerts:
        console.print("[green]No alerts triggered.[/green]")
        return
    for alert in alerts:
        style = {
            "low": "yellow",
            "medium": "dark_orange",
            "high": "red",
            "critical": "bold red on white",
        }.get(alert.severity.value, "white")
        console.print(
            Panel(alert.message, title=f"Alert [{alert.severity.value.upper()}]", style=style)
        )


def print_trend_summary(result: TrendResult) -> None:
    """Print a trend analysis summary."""
    console.print(
        Panel(
            f"[bold]Trend Analysis[/bold]\n"
            f"  Slope: {result.slope_hectares_per_day:.4f} hectares/day\n"
            f"  R-squared: {result.r_squared:.4f}\n"
            f"  p-value: {result.p_value:.4e}\n"
            f"  Total cumulative loss: {result.cumulative_loss[-1]:.2f} hectares",
            title="Deforestation Trend",
        )
    )


def plot_trend(result: TrendResult, save_path: str | None = None) -> None:
    """Plot cumulative loss and rolling average using matplotlib.

    Args:
        result: Output of TrendAnalyzer.analyze().
        save_path: If given, save the figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cumulative loss.
    ax1.plot(result.dates, result.cumulative_loss, "o-", color="darkred", label="Cumulative loss")
    # Trend line.
    day_nums = np.array([(d.toordinal() - result.dates[0].toordinal()) for d in result.dates])
    trend_line = result.slope_hectares_per_day * day_nums + result.intercept
    ax1.plot(result.dates, trend_line, "--", color="gray", label="Linear trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative hectares lost")
    ax1.set_title("Cumulative Deforestation")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=30)

    # Rolling average.
    ax2.bar(result.dates, result.rolling_avg, width=20, color="orange", alpha=0.7)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Hectares lost (rolling avg)")
    ax2.set_title("Rolling Average Loss per Period")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        console.print(f"[green]Chart saved to {save_path}[/green]")
    else:
        plt.show()
    plt.close(fig)
