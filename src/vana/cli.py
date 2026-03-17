"""CLI entry point for VANA deforestation monitor."""

from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="vana")
def cli() -> None:
    """VANA - Deforestation Monitor using satellite imagery."""


@cli.command()
@click.option("--regions", default=1, help="Number of regions to simulate.")
@click.option("--timesteps", default=6, help="Number of time steps per region.")
@click.option("--rate", default=0.03, help="Deforestation rate per step (0-1).")
@click.option("--seed", default=42, help="Random seed.")
def demo(regions: int, timesteps: int, rate: float, seed: int) -> None:
    """Run a full demo with synthetic satellite data."""
    from vana.simulator import Simulator
    from vana.detection import ChangeDetector, ForestClassifier, AlertSystem
    from vana.analysis import TrendAnalyzer
    from vana import report

    sim = Simulator(seed=seed)
    detector = ChangeDetector()
    alert_system = AlertSystem()
    trend_analyzer = TrendAnalyzer()
    classifier = ForestClassifier()

    for r in range(regions):
        region, images = sim.generate_time_series(
            region_id=f"region_{r}",
            n_steps=timesteps,
            deforestation_rate=rate,
        )
        console.print(f"\n[bold cyan]Region: {region.name}[/bold cyan]")
        console.print(
            f"  Location: ({region.latitude:.2f}, {region.longitude:.2f})"
        )

        # Classification of first image.
        label_map = classifier.classify_rules(images[0])
        fractions = classifier.class_fractions(label_map)
        console.print("  Initial land cover:")
        for lc, frac in fractions.items():
            console.print(f"    {lc}: {frac:.1%}")

        # Detect changes across time series.
        events = detector.detect_series(images)
        report.print_event_table(events)

        # Alerts.
        alerts = alert_system.evaluate_many(events)
        report.print_alerts(alerts)

        # Trend analysis.
        dates = [ev.end_date for ev in events]
        losses = [ev.hectares_lost for ev in events]
        if len(dates) >= 2:
            trend = trend_analyzer.analyze(dates, losses)
            report.print_trend_summary(trend)


@cli.command()
@click.option("--regions", default=1, help="Number of regions.")
@click.option("--timesteps", default=6, help="Time steps per region.")
@click.option("--seed", default=42, help="Random seed.")
@click.option("--plot", is_flag=True, help="Show trend plot.")
def analyze(regions: int, timesteps: int, seed: int, plot: bool) -> None:
    """Analyze synthetic data and show results."""
    from vana.simulator import Simulator
    from vana.detection import ChangeDetector
    from vana.analysis import TrendAnalyzer
    from vana import report

    sim = Simulator(seed=seed)
    detector = ChangeDetector()
    trend_analyzer = TrendAnalyzer()

    for r in range(regions):
        _, images = sim.generate_time_series(
            region_id=f"region_{r}", n_steps=timesteps
        )
        events = detector.detect_series(images)
        report.print_event_table(events)

        dates = [ev.end_date for ev in events]
        losses = [ev.hectares_lost for ev in events]
        if len(dates) >= 2:
            trend = trend_analyzer.analyze(dates, losses)
            report.print_trend_summary(trend)
            if plot:
                report.plot_trend(trend)


@cli.command()
@click.option("--threshold", default=1.0, help="Minimum hectares for alert.")
@click.option("--regions", default=1, help="Number of regions.")
@click.option("--timesteps", default=6, help="Time steps per region.")
@click.option("--seed", default=42, help="Random seed.")
def alerts(threshold: float, regions: int, timesteps: int, seed: int) -> None:
    """Show deforestation alerts from synthetic data."""
    from vana.simulator import Simulator
    from vana.detection import ChangeDetector, AlertSystem
    from vana import report

    sim = Simulator(seed=seed)
    detector = ChangeDetector()
    alert_system = AlertSystem(low_threshold=threshold)

    all_alerts = []
    for r in range(regions):
        _, images = sim.generate_time_series(
            region_id=f"region_{r}", n_steps=timesteps
        )
        events = detector.detect_series(images)
        all_alerts.extend(alert_system.evaluate_many(events))

    report.print_alerts(all_alerts)
    console.print(f"\n[bold]Total alerts: {len(all_alerts)}[/bold]")


if __name__ == "__main__":
    cli()
