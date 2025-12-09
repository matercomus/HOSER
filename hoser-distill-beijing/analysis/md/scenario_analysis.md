### 4.5.2 Per-Scenario Performance Comparison

**Train Set Scenarios:**

| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |
|----------|------------------|----------------|----------|--------------------|------------------|------------|
| `city_center` | 99.1% | 99.6% | -0.5% | 0.0122 | 0.0119 | +0.0003 |
| `from_center` | 98.1% | 99.4% | -1.3% | 0.0355 | 0.0324 | +0.0031 |
| `off_peak` | 98.8% | 99.3% | -0.6% | 0.0068 | 0.0064 | +0.0004 |
| `peak` | 99.0% | 99.5% | -0.5% | 0.0281 | 0.0299 | -0.0018 |
| `suburban` | 98.7% | 99.2% | -0.6% | 0.0080 | 0.0074 | +0.0006 |
| `to_center` | 99.1% | 99.4% | -0.3% | 0.0325 | 0.0399 | -0.0073 |
| `weekday` | 98.8% | 99.3% | -0.6% | 0.0072 | 0.0063 | +0.0009 |
| `weekend` | 99.0% | 99.5% | -0.5% | 0.0193 | 0.0181 | +0.0011 |
| `within_center` | 99.9% | 99.9% | +0.0% | 0.0331 | 0.0312 | +0.0020 |

**Test Set Scenarios:**

| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |
|----------|------------------|----------------|----------|--------------------|------------------|------------|
| `city_center` | 99.2% | 99.6% | -0.4% | 0.0131 | 0.0119 | +0.0012 |
| `from_center` | 98.6% | 99.2% | -0.6% | 0.0389 | 0.0349 | +0.0040 |
| `off_peak` | 98.8% | 99.2% | -0.4% | 0.0067 | 0.0065 | +0.0002 |
| `peak` | 99.5% | 99.5% | +0.0% | 0.0315 | 0.0331 | -0.0016 |
| `suburban` | 98.7% | 99.0% | -0.4% | 0.0081 | 0.0075 | +0.0006 |
| `to_center` | 98.9% | 99.4% | -0.6% | 0.0390 | 0.0402 | -0.0012 |
| `weekday` | 98.9% | 99.3% | -0.4% | 0.0071 | 0.0063 | +0.0008 |
| `weekend` | 98.7% | 99.1% | -0.4% | 0.0162 | 0.0177 | -0.0016 |
| `within_center` | 99.9% | 100.0% | -0.1% | 0.0315 | 0.0358 | -0.0043 |

### 4.5.3 Notable Scenarios

**Top-5 Scenarios Where Distilled Outperforms (Distance JSD, Test):**

1. `from_center`: Δ = +0.0040 (distilled better)
2. `city_center`: Δ = +0.0012 (distilled better)
3. `weekday`: Δ = +0.0008 (distilled better)
4. `suburban`: Δ = +0.0006 (distilled better)
5. `off_peak`: Δ = +0.0002 (distilled better)

**Top-5 Scenarios Where Vanilla Outperforms (Distance JSD, Test):**

1. `within_center`: Δ = -0.0043 (vanilla better)
2. `peak`: Δ = -0.0016 (vanilla better)
3. `weekend`: Δ = -0.0016 (vanilla better)
4. `to_center`: Δ = -0.0012 (vanilla better)
5. `off_peak`: Δ = +0.0002 (vanilla better)