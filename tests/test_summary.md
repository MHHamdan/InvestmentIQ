# Test Results Summary

**Generated:** 2025-10-08 21:08:59  
**System:** InvestmentIQ ADK Architecture (Gemini-powered agents)  
**Test Count:** 5 stocks analyzed

---

## Overview

| Ticker | Company | Sector | Score | Confidence | Recommendation |
|--------|---------|--------|-------|------------|----------------|
| **MSFT** | Microsoft Corporation | Technology | +0.485 | 0.827 | BUY (High confidence) |
| **AMZN** | Amazon.com, Inc. | Consumer Cyclical | +0.477 | 0.869 | BUY (High confidence) |
| **AAPL** | Apple Inc. | Technology | +0.230 | 0.776 | BUY (High confidence) |
| **TSLA** | Tesla, Inc. | Consumer Cyclical | +0.125 | 0.795 | HOLD (High confidence) |
| **BA** | The Boeing Company | Industrials | -0.162 | 0.654 | HOLD (Moderate confidence) |

---

## Rankings

**By Sentiment Score:**

🥇 **MSFT** (Microsoft Corporation): **+0.485**
🥈 **AMZN** (Amazon.com, Inc.): **+0.477**
🥉 **AAPL** (Apple Inc.): **+0.230**
4. **TSLA** (Tesla, Inc.): **+0.125**
5. **BA** (The Boeing Company): **-0.162**

---

## Detailed Agent Breakdown


### AAPL - Apple Inc.

**Final Score:** +0.230 | **Recommendation:** BUY (High confidence)

**Agent Contributions:**

| Agent | Sentiment | Confidence | Weight | Contribution |
|-------|-----------|------------|--------|--------------|
| Financial Analyst | +0.500 | 0.800 | 0% | +0.000 |
| Market Intelligence | -0.100 | 0.800 | 0% | -0.000 |
| Qualitative Signal | +0.100 | 0.600 | 0% | +0.000 |
| Context Engine | +0.600 | 0.800 | 0% | +0.000 |

**Agent Reasoning:**

**Financial Analyst:**
- *Reasoning:* I evaluated Apple's financial health based on the provided metrics. A 0.0% revenue growth is a significant concern (assigned a -0.4 weight). Apple's gross margin of 46.2% is excellent (assigned a +0.3...
- *Key Factors:* Healthy 46.2% Gross Margin, Strong 31.5% Operating Margin, Concerning 0.0% Revenue Growth

**Market Intelligence:**
- *Reasoning:* 1. **Rating Analysis:** Calculate the buy/sell ratio: 1522 (Buy) / 44 (Sell) = ~34.6. This indicates a generally positive sentiment. However, the presence of 116 hold ratings tempers this enthusiasm. ...
- *Key Factors:* Negative upside potential (-2.4%), High number of hold ratings relative to sell ratings., Strong, but not overwhelming, buy consensus.

**Qualitative Signal:**
- *Reasoning:* Several articles mention tech stocks gaining, which is generally positive for Apple. The article about Apple and Meta nearing settlements in EU antitrust cases is a cautiously positive development. Th...
- *Key Factors:* Tech stock gains, Antitrust settlements, AI bubble concerns

**Context Engine:**
- *Reasoning:* GDP growth of 3.78% is a strong positive signal, indicating economic expansion (+0.4). The unemployment rate of 4.3% is also healthy (+0.2). The federal funds rate at 4.22% is a potential negative as ...
- *Key Factors:* Strong GDP growth of 3.78%, Low unemployment rate of 4.3%, Rising federal funds rate of 4.22%

---

### AMZN - Amazon.com, Inc.

**Final Score:** +0.477 | **Recommendation:** BUY (High confidence)

**Agent Contributions:**

| Agent | Sentiment | Confidence | Weight | Contribution |
|-------|-----------|------------|--------|--------------|
| Financial Analyst | +0.350 | 0.850 | 0% | +0.000 |
| Market Intelligence | +0.850 | 0.950 | 0% | +0.000 |
| Qualitative Signal | +0.200 | 0.800 | 0% | +0.000 |
| Context Engine | +0.500 | 0.800 | 0% | +0.000 |

**Agent Reasoning:**

**Financial Analyst:**
- *Reasoning:* I evaluated each metric based on general benchmarks and their importance to Amazon's business model. Revenue Growth: 0.0% is a negative signal, indicating stagnation. I assigned it a score of -0.3 and...
- *Key Factors:* Healthy 48.9% gross margin, Healthy 10.8% operating margin, Weak 0.0% revenue growth

**Market Intelligence:**
- *Reasoning:* 1. **Rating Analysis:** Calculate the Buy/Sell ratio. 1544 Buy ratings significantly outweigh the 3 Sell ratings. This indicates a overwhelmingly positive sentiment based on analyst ratings. I assigne...
- *Key Factors:* Strong buy consensus (1544 buy vs 3 sell), 15.7% upside potential, High analyst coverage (2036 analysts)

**Qualitative Signal:**
- *Reasoning:* I assessed the sentiment of each article. Article 2 reports Amazon outperforming the stock market (+0.3). Article 3 discusses Anthropic's expansion, implying positive growth for an Amazon-backed compa...
- *Key Factors:* Stock Market Performance, AI Investment Growth, Tech Sector Gains

**Context Engine:**
- *Reasoning:* GDP growth of 3.78% is a strong positive indicator, suggesting economic expansion and increased consumer spending (0.3). An unemployment rate of 4.3% is also positive, indicating a healthy labor marke...
- *Key Factors:* Strong GDP growth of 3.78%, Low unemployment rate of 4.3%, Rising federal funds rate of 4.22%

---

### BA - The Boeing Company

**Final Score:** -0.162 | **Recommendation:** HOLD (Moderate confidence)

**Agent Contributions:**

| Agent | Sentiment | Confidence | Weight | Contribution |
|-------|-----------|------------|--------|--------------|
| Financial Analyst | -0.850 | 0.950 | 0% | -0.000 |
| Market Intelligence | +0.000 | 0.500 | 0% | +0.000 |
| Qualitative Signal | +0.300 | 0.800 | 0% | +0.000 |
| Context Engine | +0.600 | 0.750 | 0% | +0.000 |

**Agent Reasoning:**

**Financial Analyst:**
- *Reasoning:* I assigned a negative score due to the overall weakness indicated by the financial metrics. Specifically:
1. Revenue Growth (0.0%): A flat revenue growth contributes negatively, so I gave it a score o...
- *Key Factors:* Negative Gross Margin, Negative Operating Margin, Stagnant Revenue Growth

**Market Intelligence:**
- *Reasoning:* Gemini API call failed
- *Key Factors:* API error, No analysis available, Using neutral score

**Qualitative Signal:**
- *Reasoning:* I assigned sentiment scores to each article based on its content. Article 2 (Ryanair takes delivery of Boeing jets) and Article 10 (Boeing set to gain EU approval for Spirit Aerosystems deal) are posi...
- *Key Factors:* Increased 737 production, Spirit Aerosystems deal approval, Positive activity update from Air Lease

**Context Engine:**
- *Reasoning:* A strong GDP growth rate of 3.78% contributes positively to the sentiment, indicating a healthy economy capable of supporting Boeing's business. The low unemployment rate of 4.3% further supports this...
- *Key Factors:* Strong GDP growth of 3.78%, Low unemployment rate of 4.3%, Neutral sector performance

---

### MSFT - Microsoft Corporation

**Final Score:** +0.485 | **Recommendation:** BUY (High confidence)

**Agent Contributions:**

| Agent | Sentiment | Confidence | Weight | Contribution |
|-------|-----------|------------|--------|--------------|
| Financial Analyst | +0.500 | 0.800 | 0% | +0.000 |
| Market Intelligence | +0.750 | 0.850 | 0% | +0.000 |
| Qualitative Signal | +0.100 | 0.750 | 0% | +0.000 |
| Context Engine | +0.600 | 0.800 | 0% | +0.000 |

**Agent Reasoning:**

**Financial Analyst:**
- *Reasoning:* The sentiment score is calculated based on the following factors: Revenue Growth (0.0%): This is neutral, contributing 0 to the score. Gross Margin (68.8%): This is very strong, contributing +0.2 to t...
- *Key Factors:* High Gross Margin, High Operating Margin, Zero Debt-to-Equity

**Market Intelligence:**
- *Reasoning:* 1. **Analyze Ratings:** Calculate a 'buy-sell ratio'. Buy ratings (889) are overwhelmingly higher than sell ratings (41), suggesting positive sentiment. The ratio is 889/41 = ~21.7. This is a strong i...
- *Key Factors:* Strong buy consensus (889 buy vs 41 sell), 18.1% upside potential based on average price target, Overwhelmingly positive analyst ratings distribution

**Qualitative Signal:**
- *Reasoning:* The sentiment score is calculated based on a mixed assessment of the news. Articles 3 (Microsoft Tries to Catch Up in AI) and 7 (Google tells judge it wants to retain right to bundle Gemini) present c...
- *Key Factors:* AI competition, Earnings anticipation, Healthcare push

**Context Engine:**
- *Reasoning:* I assigned a positive sentiment score because the GDP growth of 3.78% indicates economic expansion, which is generally favorable for businesses. The unemployment rate of 4.3% is also relatively low, s...
- *Key Factors:* Strong GDP growth of 3.78%, Low unemployment rate of 4.3%, Neutral sector performance

---

### TSLA - Tesla, Inc.

**Final Score:** +0.125 | **Recommendation:** HOLD (High confidence)

**Agent Contributions:**

| Agent | Sentiment | Confidence | Weight | Contribution |
|-------|-----------|------------|--------|--------------|
| Financial Analyst | +0.200 | 0.800 | 0% | +0.000 |
| Market Intelligence | -0.350 | 0.850 | 0% | -0.000 |
| Qualitative Signal | +0.400 | 0.800 | 0% | +0.000 |
| Context Engine | +0.600 | 0.800 | 0% | +0.000 |

**Agent Reasoning:**

**Financial Analyst:**
- *Reasoning:* I assessed Tesla's financial health by evaluating each metric individually and then aggregating the scores. Revenue growth of 0.0% is concerning, contributing a negative score of -0.3. Gross margin of...
- *Key Factors:* Zero debt-to-equity ratio, Healthy operating and net margins, Zero revenue growth

**Market Intelligence:**
- *Reasoning:* 1. **Price Target Assessment:** The average price target ($378.50) is significantly lower than the current price ($438.69), resulting in a negative upside potential of -13.7%. This indicates analysts,...
- *Key Factors:* Negative upside potential (-13.7%), Significant number of sell ratings (213), Large difference between buy ratings and combined sell & hold ratings

**Qualitative Signal:**
- *Reasoning:* I analyzed the sentiment of each article individually. Articles 1, 3, 6, 7, 8, and 9 were considered positive due to mentions of new buy points, outperforming the market, Nvidia's investment in xAI, a...
- *Key Factors:* New buy point, Model Y affordability, Nvidia's investment in xAI

**Context Engine:**
- *Reasoning:* GDP growth of 3.78% is a strong positive indicator, suggesting economic expansion and increased consumer spending (0.3). An unemployment rate of 4.3% is also positive, indicating a healthy labor marke...
- *Key Factors:* Strong GDP growth of 3.78%, Relatively low unemployment rate of 4.3%, Federal Funds Rate at 4.22%

---

## Execution Metrics

| Ticker | Execution Time (s) | Timestamp |
|--------|-------------------|-----------|
| AAPL | 11.63 | 2025-10-08T21:03:44 |
| AMZN | 13.97 | 2025-10-08T21:07:32 |
| BA | 9.82 | 2025-10-08T21:07:12 |
| MSFT | 13.32 | 2025-10-08T21:06:34 |
| TSLA | 14.89 | 2025-10-08T21:06:56 |

---

## System Architecture

**Agents (4):**
1. **Financial Analyst** (35% weight) - Analyzes financial ratios from FMP
2. **Market Intelligence** (30% weight) - Evaluates analyst ratings and price targets
3. **Qualitative Signal** (25% weight) - Assesses news sentiment from EODHD
4. **Context Engine** (10% weight) - Analyzes macro indicators (FRED) and sector trends (FMP)

**Data Sources:**
- Financial Modeling Prep (FMP) - Financial ratios, analyst ratings, price targets
- EODHD - News articles and sentiment
- Federal Reserve Economic Data (FRED) - GDP, unemployment, fed funds rate

**AI Model:**
- Google Gemini 2.0 Flash (via direct API, structured outputs)
- Each agent provides transparent reasoning and key factors

**Fusion Method:**
- Weighted average of agent sentiments
- Custom fusion engine with explainable contributions
