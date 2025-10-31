### 4.5.2 Per-Scenario Performance Comparison

**Train Set Scenarios:**

| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |
|----------|------------------|----------------|----------|--------------------|------------------|------------|
| `city_center` | 87.5% | 4.3% | +83.2% | 0.0288 | 0.2063 | -0.1775 |
| `from_center` | 83.2% | 2.6% | +80.5% | 0.0573 | 0.3000 | -0.2427 |
| `off_peak` | 86.4% | 6.7% | +79.7% | 0.0222 | 0.1484 | -0.1262 |
| `peak` | 86.5% | 6.9% | +79.5% | 0.0432 | 0.1724 | -0.1292 |
| `suburban` | 85.9% | 8.2% | +77.7% | 0.0226 | 0.1281 | -0.1055 |
| `to_center` | 88.9% | 3.5% | +85.4% | 0.0587 | 0.2504 | -0.1917 |
| `weekday` | 86.3% | 6.5% | +79.9% | 0.0219 | 0.1507 | -0.1288 |
| `weekend` | 86.9% | 7.7% | +79.2% | 0.0375 | 0.1636 | -0.1261 |
| `within_center` | 90.4% | 6.3% | +84.1% | 0.0444 | 0.1927 | -0.1483 |

**Test Set Scenarios:**

| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |
|----------|------------------|----------------|----------|--------------------|------------------|------------|
| `city_center` | 88.3% | 4.8% | +83.4% | 0.0321 | 0.2135 | -0.1813 |
| `from_center` | 83.3% | 2.5% | +80.9% | 0.0791 | 0.3210 | -0.2419 |
| `off_peak` | 86.0% | 7.4% | +78.6% | 0.0191 | 0.1546 | -0.1356 |
| `peak` | 88.8% | 7.6% | +81.2% | 0.0465 | 0.1782 | -0.1318 |
| `suburban` | 85.4% | 8.9% | +76.5% | 0.0186 | 0.1332 | -0.1146 |
| `to_center` | 89.6% | 2.9% | +86.8% | 0.0599 | 0.2561 | -0.1962 |
| `weekday` | 86.8% | 7.5% | +79.4% | 0.0208 | 0.1530 | -0.1322 |
| `weekend` | 85.0% | 7.2% | +77.8% | 0.0278 | 0.1735 | -0.1458 |
| `within_center` | 92.0% | 8.6% | +83.4% | 0.0411 | 0.1795 | -0.1384 |

### 4.5.3 Notable Scenarios

**Top-5 Scenarios Where Distilled Outperforms (Distance JSD, Test):**

1. `suburban`: Δ = -0.1146 (distilled better)
2. `peak`: Δ = -0.1318 (distilled better)
3. `weekday`: Δ = -0.1322 (distilled better)
4. `off_peak`: Δ = -0.1356 (distilled better)
5. `within_center`: Δ = -0.1384 (distilled better)

**Top-5 Scenarios Where Vanilla Outperforms (Distance JSD, Test):**

1. `from_center`: Δ = -0.2419 (vanilla better)
2. `to_center`: Δ = -0.1962 (vanilla better)
3. `city_center`: Δ = -0.1813 (vanilla better)
4. `weekend`: Δ = -0.1458 (vanilla better)
5. `within_center`: Δ = -0.1384 (vanilla better)