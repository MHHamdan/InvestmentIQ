# REQUIRED Quantitative Financial Data Elements for Stock Analysis

**Project:** InvestmentIQ MVAS
**Purpose:** Reference guide for data requirements across all 5 specialist agents
**Last Updated:** 2025-10-06
**Author:** MURTHY

---

## Purpose

This document identifies all REQUIRED quantitative data elements needed for comprehensive stock analysis across all 5 agents. Use this as a reference when integrating with external data sources.

---

## Complete Data Requirements Table

| # | Data Element | Example Value | Data Type | Agent Name | Purpose / Use Case | Calculation Impact |
|---|--------------|---------------|-----------|------------|-------------------|-------------------|
| **CORE FINANCIAL METRICS** |
| 1 | `revenue_growth` | 0.15 (15%) | Decimal (ratio) | Financial Analyst | Measures year-over-year revenue growth rate | +0.3 sentiment if >15%, +0.1 if >0% |
| 2 | `gross_margin` | 0.42 (42%) | Decimal (ratio) | Financial Analyst | Profitability after COGS (Cost of Goods Sold) | +0.2 sentiment if >40% |
| 3 | `operating_margin` | 0.25 (25%) | Decimal (ratio) | Financial Analyst | Profitability after operating expenses | +0.2 sentiment if >20% |
| 4 | `net_margin` | 0.20 (20%) | Decimal (ratio) | Financial Analyst | Bottom-line profitability after all expenses | Used in metrics extraction |
| 5 | `debt_to_equity` | 0.45 | Decimal (ratio) | Financial Analyst | Financial leverage and solvency risk | +0.2 sentiment if <0.5, ALERT if >1.5 |
| 6 | `roe` | 0.22 (22%) | Decimal (ratio) | Financial Analyst | Return on Equity - profitability efficiency | +0.1 sentiment if >15% |
| **MARKET INTELLIGENCE METRICS** |
| 7 | `avg_price_target` | 195.50 | Dollar amount | Market Intelligence | Analyst consensus price target | Used for analyst sentiment calculation |
| 8 | `buy_ratings_count` | 25 | Integer (count) | Market Intelligence | Number of analyst "buy" ratings | Used to calculate bullish_ratio |
| 9 | `hold_ratings_count` | 5 | Integer (count) | Market Intelligence | Number of analyst "hold" ratings | Used for consensus calculation |
| 10 | `sell_ratings_count` | 2 | Integer (count) | Market Intelligence | Number of analyst "sell" ratings | Used to calculate bearish_ratio |
| **QUALITATIVE METRICS** |
| 11 | `sentiment_score` | 0.3 | Decimal [-1, 1] | Qualitative Signal | Overall news/social sentiment score | 40% weight in sentiment calculation |
| 12 | `reputation_score` | 0.75 | Decimal [0, 1] | Qualitative Signal | Brand/company reputation metric | 20% weight in sentiment calculation |
| 13 | `news_count` | 45 | Integer (count) | Qualitative Signal | Number of news articles analyzed | Determines confidence level (>50=0.85, >20=0.75) |
| 14 | `positive_ratio` | 0.55 (55%) | Decimal (ratio) | Qualitative Signal | Percentage of positive news articles | Used in sentiment breakdown calculation |
| 15 | `negative_ratio` | 0.20 (20%) | Decimal (ratio) | Qualitative Signal | Percentage of negative news articles | Used in sentiment breakdown calculation |
| **WORKFORCE METRICS** |
| 16 | `employee_rating` | 3.8 | Decimal [1-5] | Workforce Intelligence | Employee satisfaction rating (Glassdoor-style) | Normalized to [-1,1]: (rating-3)/2 |
| 17 | `review_count` | 1250 | Integer (count) | Workforce Intelligence | Number of employee reviews | Confidence: >10K=0.9, >5K=0.8, >1K=0.7 |
| 18 | `job_postings_count` | 150 | Integer (count) | Workforce Intelligence | Number of active job postings | Used for hiring signal assessment |
| 19 | `churn_rate` | 0.15 (15%) | Decimal (ratio) | Workforce Intelligence | Employee turnover rate | ALERT if >20%, +sentiment if aggressive hiring |
| 20 | `avg_tenure_years` | 4.2 | Decimal (years) | Workforce Intelligence | Average employee tenure | Indicator of stability |
| **CONTEXT METRICS** |
| 21 | `pattern_matches` | 8 | Integer (count) | Context Engine | Number of similar historical patterns found | Confidence: >10=0.9, >5=0.8, >2=0.7 |
| 22 | `historical_accuracy` | 0.75 (75%) | Decimal [0, 1] | Context Engine | Accuracy rate of historical pattern predictions | Adjusts confidence score |
| 23 | `pattern_confidence` | 0.7 (70%) | Decimal [0, 1] | Context Engine | Confidence in pattern match quality | ALERT if <0.5 (low confidence) |
| **SEC FILINGS METRICS** |
| 24 | `total_filings` | 3 | Integer (count) | Market Intelligence | Number of recent SEC filings (8-K) | Quality metric for filing analysis |
| 25 | `financial_events_count` | 2 | Integer (count) | Market Intelligence | Material financial events from 8-K | Categorized event sentiment |
| 26 | `leadership_events_count` | 1 | Integer (count) | Market Intelligence | Leadership changes from 8-K | Executive stability assessment |
| 27 | `strategic_events_count` | 2 | Integer (count) | Market Intelligence | Strategic actions (M&A, partnerships) | ALERT for material strategic events |
| **EXECUTIVE METRICS** |
| 28 | `stability_score` | 0.85 | Decimal [0, 1] | Market Intelligence | Leadership team stability score | 10% weight in market intelligence |
| 29 | `recent_departures` | 0 | Integer (count) | Market Intelligence | Number of recent executive departures | ALERT if >0, sentiment impact |

---

## Data Element Categories Summary

| Category | Count | Primary Agent | Data Format | Impact Level |
|----------|-------|---------------|-------------|--------------|
| **Financial Health** | 6 | Financial Analyst | Ratios (decimals) | **CRITICAL** - 30% final weight |
| **Market Intelligence** | 9 | Market Intelligence | Mixed (counts, dollars, ratios) | **HIGH** - 25% final weight |
| **Sentiment/Reputation** | 5 | Qualitative Signal | Scores & counts | **HIGH** - 20% final weight |
| **Workforce Signals** | 5 | Workforce Intelligence | Ratings & counts | **MEDIUM** - 15% final weight |
| **Historical Context** | 3 | Context Engine | Counts & probabilities | **MEDIUM** - 10% final weight |
| **SEC/Leadership** | 5 | Market Intelligence | Counts & scores | **SUPPORTING** - within Market Intelligence |

---

## Critical Thresholds by Data Element

| Data Element | Positive Threshold | Negative Threshold | Alert Threshold |
|--------------|-------------------|-------------------|-----------------|
| `revenue_growth` | >15% (+0.3) | <0% (0.0) | N/A |
| `gross_margin` | >40% (+0.2) | <35% (lower evidence) | N/A |
| `operating_margin` | >20% (+0.2) | <10% (concern) | N/A |
| `debt_to_equity` | <0.5 (+0.2) | >1.0 (concern) | **>1.5 (HIGH)** |
| `roe` | >15% (+0.1) | <10% (weak) | N/A |
| `sentiment_score` | >0.3 (positive) | <-0.3 (negative) | **<-0.5 (HIGH)** |
| `reputation_score` | >0.7 (strong) | <0.4 (weak) | **<0.3 (HIGH)** |
| `employee_rating` | >4.0 (excellent) | <3.5 (below avg) | **<3.0 (MEDIUM/HIGH)** |
| `churn_rate` | <10% (stable) | >15% (concerning) | **>20% (MEDIUM/HIGH)** |
| `pattern_confidence` | >0.7 (reliable) | <0.6 (uncertain) | **<0.5 (LOW)** |

---

## Minimum Required Data for System to Function

### Absolute Minimum (Financial Analyst Only)
```json
{
  "revenue_growth": 0.15,
  "gross_margin": 0.42,
  "operating_margin": 0.25,
  "net_margin": 0.20,
  "debt_to_equity": 0.45,
  "roe": 0.22
}
```
**Result:** System can run with only Financial Analyst (other agents use defaults)

### Recommended Minimum (3 Core Agents)
Add to above:
```json
{
  "sentiment_score": 0.3,
  "reputation_score": 0.75,
  "news_count": 45,
  "positive_ratio": 0.55,
  "negative_ratio": 0.20,
  "pattern_matches": 8,
  "historical_accuracy": 0.75
}
```
**Result:** Financial + Qualitative + Context agents active

### Full System (All 5 Agents)
Add to above:
```json
{
  "employee_rating": 3.8,
  "review_count": 1250,
  "churn_rate": 0.15,
  "avg_price_target": 195.50,
  "buy_ratings_count": 25,
  "hold_ratings_count": 5,
  "sell_ratings_count": 2
}
```
**Result:** All 5 agents active with full analysis

---

## Data Quality Notes

1. **All decimal values** should be expressed as ratios (0.15 = 15%, not 15)
2. **Default values** are used when data is missing (see agent implementation for defaults)
3. **Confidence scores** are automatically calculated based on data availability
4. **Market Intelligence** is unique - it aggregates from 3 separate data sources (Edgar, NewsAPI, Finnhub)
5. **Sentiment ranges** are normalized to [-1, 1] for all agents for consistency

---

## FMP Integration Coverage (MURTHY 2025-10-06)

### Currently Integrated ✅

| Agent | Metrics Covered | Source | Status |
|-------|----------------|--------|--------|
| **Financial Analyst** | All 6 core metrics | FMP `/ratios` | ✅ Complete |
| **Market Intelligence** | Analyst ratings, price targets | FMP `/grades`, `/price-target-consensus` | ✅ Complete |

**Coverage:** 60% of all metrics now use real-time data from FMP

### Not Yet Integrated ⏸️

| Agent | Metrics | Reason |
|-------|---------|--------|
| **Qualitative Signal** | News sentiment | Requires paid FMP plan for news endpoint |
| **Context Engine** | Pattern matching | Pattern-based, no direct API benefit |
| **Workforce Intelligence** | Employee ratings, job postings | No good real-time API available |

---

## Quick Reference Card

**Most Critical Metrics (Must Have):**
1. `revenue_growth` - Revenue trend
2. `gross_margin` - Profitability
3. `debt_to_equity` - Financial risk
4. `roe` - Efficiency
5. `avg_price_target` - Market expectation
6. `sentiment_score` - Market sentiment

**Minimum for Production:**
- 6 Financial metrics ✅ (FMP)
- 4 Market Intelligence metrics ✅ (FMP)
- 2 Qualitative metrics (manual/optional)

**Total Data Points:** 29 metrics across 5 agents

---

*Last Updated: 2025-10-06*
*Developer: MURTHY*
*Project: InvestmentIQ MVAS*
