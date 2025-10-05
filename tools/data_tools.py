"""
Data Access Tools for InvestmentIQ MVAS

These tools act as MCP-like interfaces for accessing mock data sources.
In production, these would connect to real APIs and databases.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json


class FinancialDataTool:
    """Tool for accessing financial data"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    async def read_financial_data(self, company_id: str) -> Dict[str, Any]:
        """
        Read financial data for a company.

        Args:
            company_id: Unique identifier for the company

        Returns:
            Dictionary containing financial metrics
        """
        data_file = self.data_dir / f"{company_id}_financial.json"

        if not data_file.exists():
            return {
                "error": f"Financial data not found for company: {company_id}",
                "company_id": company_id
            }

        with open(data_file, 'r') as f:
            data = json.load(f)

        return data

    async def get_financial_ratios(self, company_id: str) -> Dict[str, Any]:
        """
        Calculate and return key financial ratios.

        Args:
            company_id: Unique identifier for the company

        Returns:
            Dictionary containing calculated ratios
        """
        raw_data = await self.read_financial_data(company_id)

        if "error" in raw_data:
            return raw_data

        # Extract metrics
        revenue = raw_data.get("revenue", 0)
        gross_profit = raw_data.get("gross_profit", 0)
        operating_income = raw_data.get("operating_income", 0)
        net_income = raw_data.get("net_income", 0)
        total_debt = raw_data.get("total_debt", 0)
        total_equity = raw_data.get("total_equity", 0)

        # Calculate ratios
        gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0
        operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
        net_margin = (net_income / revenue * 100) if revenue > 0 else 0
        debt_to_equity = (total_debt / total_equity) if total_equity > 0 else 0

        return {
            "company_id": company_id,
            "ratios": {
                "gross_margin": round(gross_margin, 2),
                "operating_margin": round(operating_margin, 2),
                "net_margin": round(net_margin, 2),
                "debt_to_equity": round(debt_to_equity, 2)
            },
            "raw_data": raw_data
        }


class QualitativeDataTool:
    """Tool for processing qualitative/unstructured data"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    async def process_unstructured_text(self, company_id: str) -> Dict[str, Any]:
        """
        Process unstructured text data (news, reviews, sentiment).

        Args:
            company_id: Unique identifier for the company

        Returns:
            Dictionary containing processed sentiment data
        """
        data_file = self.data_dir / f"{company_id}_qualitative.txt"

        if not data_file.exists():
            return {
                "error": f"Qualitative data not found for company: {company_id}",
                "company_id": company_id
            }

        with open(data_file, 'r') as f:
            raw_text = f.read()

        # Analyze sentiment (mock implementation)
        sentiment_score = self._analyze_sentiment(raw_text)
        key_themes = self._extract_themes(raw_text)

        return {
            "company_id": company_id,
            "raw_text": raw_text,
            "sentiment_score": sentiment_score,
            "sentiment_label": self._get_sentiment_label(sentiment_score),
            "key_themes": key_themes,
            "text_length": len(raw_text)
        }

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text and return a score.

        Returns:
            Score from -1.0 (very negative) to 1.0 (very positive)
        """
        # Mock sentiment analysis based on keyword counts
        positive_keywords = [
            "growth", "profit", "success", "innovation", "strong",
            "opportunity", "gain", "improve", "excellent", "positive"
        ]
        negative_keywords = [
            "loss", "decline", "problem", "issue", "layoff", "exodus",
            "poor", "concern", "risk", "negative", "crisis", "scandal"
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)

        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0

        # Calculate normalized score
        score = (positive_count - negative_count) / total_keywords
        return round(score, 2)

    def _extract_themes(self, text: str) -> list:
        """Extract key themes from text"""
        themes = []
        text_lower = text.lower()

        theme_keywords = {
            "leadership": ["ceo", "executive", "management", "leadership"],
            "financial": ["revenue", "profit", "margin", "earnings"],
            "operations": ["operations", "production", "efficiency"],
            "workforce": ["employee", "layoff", "hiring", "turnover"],
            "market": ["market", "competition", "share", "position"],
            "innovation": ["innovation", "technology", "r&d", "development"]
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)

        return themes

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score <= -0.6:
            return "Very Negative"
        elif score <= -0.2:
            return "Negative"
        elif score < 0.2:
            return "Neutral"
        elif score < 0.6:
            return "Positive"
        else:
            return "Very Positive"


class ContextRuleTool:
    """Tool for accessing context rules and historical patterns"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    async def get_context_rule(
        self,
        scenario_type: str,
        sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve context rule for a given scenario.

        Args:
            scenario_type: Type of scenario (e.g., 'contrarian_opportunity')
            sector: Optional sector filter

        Returns:
            Dictionary containing the applicable rule
        """
        rules_file = self.data_dir / "context_rules.json"

        if not rules_file.exists():
            return {
                "error": "Context rules file not found",
                "scenario_type": scenario_type
            }

        with open(rules_file, 'r') as f:
            all_rules = json.load(f)

        # Find matching rule
        for rule in all_rules.get("rules", []):
            if rule.get("scenario_type") == scenario_type:
                if sector and rule.get("sector") != sector:
                    continue
                return rule

        return {
            "error": f"No rule found for scenario: {scenario_type}",
            "scenario_type": scenario_type
        }

    async def get_historical_correlation(
        self,
        pattern: str,
        lookback_years: int = 5
    ) -> Dict[str, Any]:
        """
        Get historical correlation data for a pattern.

        Args:
            pattern: Pattern identifier
            lookback_years: Number of years to look back

        Returns:
            Dictionary containing correlation statistics
        """
        # Mock implementation
        return {
            "pattern": pattern,
            "lookback_years": lookback_years,
            "correlation_strength": 0.75,
            "sample_size": 42,
            "confidence_level": 0.85
        }
