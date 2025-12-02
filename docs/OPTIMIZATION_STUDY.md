# AI Trader Optimization Study: SGD vs Grid Search vs Bayesian vs Regime-Adaptive

## Executive Summary

This comprehensive study compares multiple optimization approaches for the AI Trader strategy parameters. The goal was to determine whether advanced optimization techniques (SGD, Bayesian Optimization) could outperform the current grid search approach.

### Key Findings

| Method | Alpha vs SPY | Recommendation |
|--------|--------------|----------------|
| **Aggressive Grid** (5%/12%/60) | **+24.5%** | Best performer, but higher risk |
| Bayesian Optimization | -3.8% | Good out-of-sample, worth monitoring |
| Regime-Adaptive | -10.0% | Shows promise, needs tuning |
| SGD Weight Optimization | -13.5% | +5.1% improvement over default |
| Current (Tight Entry) | -21.7% | Underperforming in bull market |

### Bottom Line

**SGD provides marginal improvement (+5.1% alpha) over default weights, but the complexity may not be worth it.** The bigger opportunity lies in:

1. **Wider stops in bull markets** - The "Aggressive" grid (5% SL, 12% TP) massively outperformed
2. **Regime-adaptive parameters** - Switching params based on market regime shows +11.6% improvement over fixed params

---

## Study Methodology

### Test Period
- **Full Period**: January 2023 - November 2024 (23 months)
- **Training Period**: Jan 2023 - Feb 2024 (60%)
- **Test Period**: Feb 2024 - Nov 2024 (40%)

### Universe
16 liquid S&P 500 stocks across sectors + SPY benchmark

### Methods Tested

1. **Grid Search** - Manual parameter combinations (current approach)
2. **SGD (Stochastic Gradient Descent)** - Optimize indicator weights via gradient descent
3. **Bayesian Optimization (Optuna)** - Probabilistic hyperparameter search
4. **Regime-Adaptive** - Dynamic parameters based on market regime detection

---

## Detailed Results

### 1. Grid Search (Baseline)

| Strategy | Stop Loss | Take Profit | Min Score | Return | Alpha | Max DD |
|----------|-----------|-------------|-----------|--------|-------|--------|
| **Aggressive** | 5% | 12% | 60 | +81.2% | **+24.5%** | -12.2% |
| Conservative | 2% | 6% | 75 | +39.5% | -17.2% | -7.4% |
| Tight Entry (Current) | 3% | 8% | 70 | +35.1% | -21.7% | -14.6% |

**Key Insight**: In a strong bull market (SPY +56.8%), wider stops and take profits significantly outperform. The "Tight Entry" parameters optimized for 5-year data (including 2022 bear market) are too conservative for current conditions.

### 2. SGD Weight Optimization

**Objective**: Learn optimal weights for technical indicators (RSI, MACD, BB, Trend) using gradient descent.

**Approach**:
- Train a differentiable model to predict trade profitability
- Use SGD to optimize indicator weights and entry threshold
- Proxy loss: Binary cross-entropy on trade outcomes

**Results**:

| Parameter | Default | SGD Optimized |
|-----------|---------|---------------|
| RSI Weight | 0.30 | 0.30 |
| MACD Weight | 0.25 | 0.24 |
| BB Weight | 0.25 | 0.22 |
| Trend Weight | 0.20 | 0.24 |
| Threshold | 70.0 | 70.0 |

**Test Period Performance**:
- SGD: +5.4% return, -13.5% alpha
- Default: +0.3% return, -18.6% alpha
- **Improvement: +5.1% alpha**

**Analysis**:
SGD found slightly different weights (reducing BB weight, increasing Trend weight) but kept the same threshold. The improvement is modest and may be due to:
- Trend weight increase capturing momentum better
- Reduced BB weight avoiding mean-reversion signals in trending market

**Verdict**: SGD provides marginal improvement. The weights didn't change dramatically because the original weights were already reasonable. The main issue isn't the weights—it's the stop/take-profit parameters.

### 3. Bayesian Optimization (Optuna)

**Approach**: Use Tree-structured Parzen Estimator (TPE) to efficiently search the parameter space.

**Best Parameters Found**:
- Stop Loss: 5.8%
- Take Profit: 5.0%
- Min Score: 60.2

**Test Period Performance**:
- Return: +15.1%
- Alpha: -3.8%

**Analysis**:
Bayesian optimization found parameters closer to the "Aggressive" grid (wider stops, lower score threshold). Interestingly, it preferred a lower take profit (5%) which takes profits quickly—this is a scalping-style approach.

**Verdict**: Bayesian optimization is useful for exploring the parameter space more efficiently than grid search, but the results are only as good as the training period. In this bull market, it gravitates toward aggressive parameters.

### 4. Regime-Adaptive Strategy

**Approach**: Detect market regime (Bull/Bear/Sideways) using SPY's price relative to EMA-50 and 60-day momentum, then apply regime-specific parameters.

**Regime Parameters**:

| Regime | Stop Loss | Take Profit | Min Score | Days Detected |
|--------|-----------|-------------|-----------|---------------|
| Bull | 3% | 10% | 65 | 234 (49%) |
| Bear | 2% | 6% | 75 | 12 (2%) |
| Sideways | 2.5% | 5% | 70 | 235 (49%) |

**Performance**:
- Return: +46.7%
- Alpha: -10.0%
- Improvement over Tight Entry: +11.6%

**Analysis**:
The regime detector classified roughly half the days as "bull" and half as "sideways", with almost no "bear" days. This is consistent with the 2023-2024 market. The strategy beat the fixed "Tight Entry" approach by adapting to conditions.

**Verdict**: Regime-adaptive shows promise. The +11.6% improvement over fixed parameters suggests real value in dynamic parameter selection. However, the regime detection logic needs refinement—it may be too slow to react.

---

## Why SGD Didn't Help More

### The Fundamental Problem

SGD optimizes **continuous, differentiable** functions. Trading P&L is neither:

```
P&L = Σ (exit_price - entry_price) × position_size
```

The P&L depends on **discrete events** (stop hit? target hit?) that have no gradient. Our workaround—predicting trade profitability—creates a proxy loss that doesn't directly optimize what we care about.

### What SGD Could Optimize vs What Matters

| What SGD Optimizes | What Actually Matters |
|--------------------|----------------------|
| Indicator weights | Stop loss level |
| Entry threshold | Take profit level |
| Prediction accuracy | Risk-adjusted returns |

The study confirms: **the weights aren't the bottleneck**. The original weights (RSI 30%, MACD 25%, BB 25%, Trend 20%) are reasonable. The real alpha comes from:

1. **Position sizing** (not optimized)
2. **Stop/take-profit levels** (huge impact)
3. **Market regime adaptation** (significant impact)

---

## Recommendations

### Immediate Actions

1. **Consider widening stops in current bull market**
   - Current: 3% SL, 8% TP
   - Suggested: 4-5% SL, 10-12% TP
   - Rationale: Strong uptrend with pullbacks getting bought

2. **Implement regime detection**
   - Add VIX-based or trend-based regime classifier
   - Use tighter stops in high-volatility/bear regimes
   - Use wider stops in low-volatility/bull regimes

### Future Research

1. **Reinforcement Learning (RL)**
   - Better suited for sequential decision problems
   - Can directly optimize cumulative P&L
   - Requires extensive simulation/backtesting infrastructure

2. **Online Learning**
   - Continuously update weights as new data arrives
   - Adapt faster to changing market conditions

3. **Ensemble Methods**
   - Combine multiple strategies with different parameters
   - Reduce risk of single-strategy failure

### What NOT to Do

1. **Don't over-optimize on recent data**
   - The "Aggressive" grid won this period but may fail in a bear market
   - Your 5-year walk-forward validation remains valuable

2. **Don't add complexity without clear benefit**
   - SGD's +5.1% improvement may disappear in different market conditions
   - Simple, robust strategies often beat complex ones

---

## Code & Reproducibility

All optimization code saved to:
- `/root/ai_trader/backtesting/optimization_study.py` - Full study
- `/root/ai_trader/backtesting/optimization_study_fast.py` - Fast version
- `/root/ai_trader/backtesting/optimization_study_results.json` - Results

### Running the Study

```bash
cd /root/ai_trader
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
python backtesting/optimization_study_fast.py
```

---

## Conclusion

**Should you switch to SGD?** No, not for weight optimization alone. The +5.1% improvement doesn't justify the added complexity.

**What should you do instead?**

1. **Monitor market regime** and consider wider stops in bull markets
2. **Implement regime-adaptive parameters** - this showed the most promise (+11.6%)
3. **Keep your 5-year validated parameters** as the default, but be willing to adjust in extreme conditions

The current "Tight Entry" parameters (3%/8%/70) were validated across multiple market conditions including the 2022 bear market. They may underperform in strong bull markets, but they provide downside protection when conditions change.

---

*Study completed: December 2, 2025*
*Period analyzed: January 2023 - November 2024*
