"""
RAG Context Tool for InvestmentIQ MVAS

Advanced Feature: RAG-based context enhancement using Pinecone vector database.
Augments static context rules with dynamic historical case retrieval.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from utils.llm_factory import get_llm_factory
from utils.observability import trace_tool

load_dotenv()


class RAGContextTool:
    """
    RAG-powered context retrieval tool using Pinecone.

    Features:
    - Index historical investment cases
    - Retrieve similar scenarios using semantic search
    - Augment context rules with real-world precedents
    - Provide evidence-based recommendations
    """

    def __init__(
        self,
        index_name: str = "investment-iq-context",
        dimension: int = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension
    ):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.dimension = dimension

        # Initialize LLM factory for embeddings
        self.llm_factory = get_llm_factory()

        # Initialize Pinecone client
        self.pc = None
        self.index = None
        self.embeddings = None

        if self.api_key and self.api_key != "your_pinecone_key_here":
            self._initialize_pinecone()
        else:
            print("Warning: Pinecone API key not configured. RAG features disabled.")

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)

            # Create index if it doesn't exist
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                print(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"✓ Index '{self.index_name}' created")
            else:
                print(f"✓ Using existing index: {self.index_name}")

            # Connect to index
            self.index = self.pc.Index(self.index_name)

            # Initialize embeddings model
            self.embeddings = self.llm_factory.create_embeddings(
                provider="huggingface"
            )

            print("✓ RAG Context Tool initialized successfully")

        except Exception as e:
            print(f"Warning: Failed to initialize Pinecone: {e}")
            self.pc = None
            self.index = None

    @trace_tool("rag_context_tool")
    def index_historical_cases(self, cases: List[Dict[str, Any]]) -> int:
        """
        Index historical investment cases into Pinecone.

        Args:
            cases: List of historical case dictionaries with:
                - id: Unique case identifier
                - scenario: Investment scenario description
                - outcome: What happened (success/failure)
                - recommendation: What was recommended
                - accuracy: How accurate the recommendation was
                - metadata: Additional context

        Returns:
            Number of cases indexed
        """
        if not self.index or not self.embeddings:
            print("Warning: RAG not initialized. Skipping indexing.")
            return 0

        try:
            vectors = []

            for case in cases:
                # Create text representation for embedding
                text = f"{case['scenario']} | Outcome: {case['outcome']} | Recommendation: {case['recommendation']}"

                # Generate embedding
                embedding = self.embeddings.embed_query(text)

                # Prepare metadata
                metadata = {
                    "scenario": case['scenario'],
                    "outcome": case['outcome'],
                    "recommendation": case['recommendation'],
                    "accuracy": case.get('accuracy', 0.0),
                    **case.get('metadata', {})
                }

                vectors.append({
                    "id": case['id'],
                    "values": embedding,
                    "metadata": metadata
                })

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)

            print(f"✓ Indexed {len(vectors)} historical cases")
            return len(vectors)

        except Exception as e:
            print(f"Error indexing cases: {e}")
            return 0

    @trace_tool("rag_retrieval")
    def retrieve_similar_cases(
        self,
        scenario: str,
        top_k: int = 3,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical cases using semantic search.

        Args:
            scenario: Current investment scenario description
            top_k: Number of similar cases to retrieve
            min_score: Minimum similarity score threshold

        Returns:
            List of similar historical cases with metadata
        """
        if not self.index or not self.embeddings:
            print("Warning: RAG not initialized. Returning empty results.")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(scenario)

            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Filter by minimum score and format results
            similar_cases = []
            for match in results['matches']:
                if match['score'] >= min_score:
                    similar_cases.append({
                        "case_id": match['id'],
                        "similarity_score": match['score'],
                        "scenario": match['metadata'].get('scenario', ''),
                        "outcome": match['metadata'].get('outcome', ''),
                        "recommendation": match['metadata'].get('recommendation', ''),
                        "accuracy": match['metadata'].get('accuracy', 0.0),
                        "metadata": match['metadata']
                    })

            return similar_cases

        except Exception as e:
            print(f"Error retrieving cases: {e}")
            return []

    def augment_context_with_rag(
        self,
        scenario: str,
        financial_health: str,
        sentiment: str
    ) -> Dict[str, Any]:
        """
        Augment context engine analysis with RAG-retrieved historical cases.

        Args:
            scenario: Current investment scenario
            financial_health: Financial analysis result
            sentiment: Sentiment analysis result

        Returns:
            Augmented context with historical precedents
        """
        # Create scenario description for retrieval
        scenario_text = f"Financial Health: {financial_health}, Sentiment: {sentiment}, Scenario: {scenario}"

        # Retrieve similar historical cases
        similar_cases = self.retrieve_similar_cases(scenario_text, top_k=3)

        if not similar_cases:
            return {
                "rag_enabled": False,
                "similar_cases_found": 0,
                "historical_context": "No similar historical cases found"
            }

        # Aggregate insights from historical cases
        total_accuracy = sum(case['accuracy'] for case in similar_cases)
        avg_accuracy = total_accuracy / len(similar_cases) if similar_cases else 0

        # Extract common outcomes
        outcomes = [case['outcome'] for case in similar_cases]
        recommendations = [case['recommendation'] for case in similar_cases]

        return {
            "rag_enabled": True,
            "similar_cases_found": len(similar_cases),
            "historical_accuracy": avg_accuracy,
            "similar_cases": similar_cases,
            "common_outcomes": outcomes,
            "historical_recommendations": recommendations,
            "confidence_boost": 0.1 if len(similar_cases) >= 2 else 0.0,
            "historical_context": self._generate_historical_summary(similar_cases)
        }

    def _generate_historical_summary(self, cases: List[Dict[str, Any]]) -> str:
        """Generate human-readable summary of historical cases."""
        if not cases:
            return "No historical precedents available"

        summary_parts = [f"Found {len(cases)} similar historical cases:"]

        for i, case in enumerate(cases, 1):
            summary_parts.append(
                f"\n{i}. {case['scenario']} → {case['outcome']} "
                f"(similarity: {case['similarity_score']:.2f}, accuracy: {case['accuracy']:.2f})"
            )

        return "\n".join(summary_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if not self.index:
            return {"status": "disabled", "reason": "Pinecone not initialized"}

        try:
            stats = self.index.describe_index_stats()
            return {
                "status": "active",
                "total_vectors": stats.total_vector_count,
                "dimension": self.dimension,
                "index_name": self.index_name
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
_rag_context_tool = None


def get_rag_context_tool() -> RAGContextTool:
    """Get singleton RAG context tool instance."""
    global _rag_context_tool
    if _rag_context_tool is None:
        _rag_context_tool = RAGContextTool()
    return _rag_context_tool
