# VANA - Deforestation Monitor

VANA is a deforestation monitoring tool that analyzes satellite imagery to detect, classify, and report forest cover changes over time.

## Features

- **Change Detection**: Compare satellite images across time periods using NDVI differencing to identify areas of forest loss.
- **Land Classification**: Classify land cover into forest, cleared, water, and urban categories using scikit-learn.
- **Alert System**: Automatically trigger alerts when deforestation exceeds configurable thresholds.
- **NDVI Analysis**: Compute Normalized Difference Vegetation Index from multispectral (NIR + Red) bands.
- **Area Calculation**: Convert pixel-level detections into hectares of forest loss using spatial resolution metadata.
- **Trend Analysis**: Track deforestation rates over time with linear regression and rolling statistics.
- **Synthetic Data**: Generate realistic satellite imagery for testing and development.
- **Rich Reports**: Produce formatted terminal reports and matplotlib visualizations.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Run a demo with synthetic satellite data
vana demo

# Generate synthetic data and save to disk
vana simulate --regions 3 --timesteps 6 --output data/

# Analyze a region and print a report
vana analyze --data data/ --region region_0

# List detected alerts
vana alerts --data data/ --threshold 5.0
```

## Project Structure

```
src/vana/
  cli.py              - Click CLI entry point
  models.py           - Pydantic data models
  simulator.py        - Synthetic satellite data generator
  report.py           - Report generation and visualization
  detection/
    change_detector.py - NDVI-based change detection
    classifier.py      - Forest/cleared/water/urban classification
    alert.py           - Deforestation alert system
  analysis/
    ndvi.py            - NDVI calculation from multispectral bands
    area.py            - Hectare-level area calculation
    trend.py           - Deforestation rate trend analysis
```

## Dependencies

- numpy, scipy, scikit-learn
- pydantic, click, rich, matplotlib

## Author

Mukunda Katta
