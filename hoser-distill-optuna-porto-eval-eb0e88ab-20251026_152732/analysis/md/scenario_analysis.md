### 4.5.2 Per-Scenario Performance Comparison

**Train Set Scenarios:**

| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |
|----------|------------------|----------------|----------|--------------------|------------------|------------|
| `city_center` | 83.4% | 86.2% | -2.7% | 0.0087 | 0.0089 | -0.0002 |
| `from_center` | 75.3% | 77.0% | -1.7% | 0.0294 | 0.0345 | -0.0051 |
| `off_peak` | 82.4% | 85.2% | -2.8% | 0.0098 | 0.0100 | -0.0002 |
| `peak` | 81.7% | 84.4% | -2.7% | 0.0466 | 0.0394 | +0.0072 |
| `suburban` | 73.7% | 76.5% | -2.8% | 0.0506 | 0.0515 | -0.0010 |
| `to_center` | 80.4% | 84.7% | -4.3% | 0.0422 | 0.0310 | +0.0112 |
| `weekday` | 82.0% | 84.9% | -2.8% | 0.0105 | 0.0106 | -0.0000 |
| `weekend` | 82.4% | 85.3% | -2.9% | 0.0152 | 0.0177 | -0.0025 |
| `within_center` | 86.3% | 88.9% | -2.7% | 0.0123 | 0.0113 | +0.0010 |

**Test Set Scenarios:**

| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |
|----------|------------------|----------------|----------|--------------------|------------------|------------|
| `city_center` | 82.8% | 85.7% | -2.9% | 0.0069 | 0.0068 | +0.0001 |
| `from_center` | 72.8% | 77.0% | -4.2% | 0.0316 | 0.0302 | +0.0014 |
| `off_peak` | 82.3% | 85.1% | -2.8% | 0.0070 | 0.0074 | -0.0004 |
| `peak` | 82.5% | 84.7% | -2.2% | 0.0426 | 0.0435 | -0.0009 |
| `suburban` | 77.5% | 77.6% | -0.1% | 0.0598 | 0.0719 | -0.0121 |
| `to_center` | 80.9% | 84.7% | -3.8% | 0.0434 | 0.0432 | +0.0003 |
| `weekday` | 82.5% | 85.2% | -2.8% | 0.0086 | 0.0090 | -0.0004 |
| `weekend` | 81.3% | 83.8% | -2.5% | 0.0175 | 0.0173 | +0.0002 |
| `within_center` | 86.0% | 88.3% | -2.3% | 0.0105 | 0.0088 | +0.0017 |

### 4.5.3 Notable Scenarios

**Top-5 Scenarios Where Distilled Outperforms (Distance JSD, Test):**

1. `within_center`: Δ = +0.0017 (distilled better)
2. `from_center`: Δ = +0.0014 (distilled better)
3. `to_center`: Δ = +0.0003 (distilled better)
4. `weekend`: Δ = +0.0002 (distilled better)
5. `city_center`: Δ = +0.0001 (distilled better)

**Top-5 Scenarios Where Vanilla Outperforms (Distance JSD, Test):**

1. `suburban`: Δ = -0.0121 (vanilla better)
2. `peak`: Δ = -0.0009 (vanilla better)
3. `off_peak`: Δ = -0.0004 (vanilla better)
4. `weekday`: Δ = -0.0004 (vanilla better)
5. `city_center`: Δ = +0.0001 (vanilla better)