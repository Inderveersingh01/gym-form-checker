from typing import Any, Dict, List, Optional
from coaching.rag.vector_store import VectorStore
from exercises.base import Fault


class BiomechanicsRAGRetriever:
    """
    Domain-specific RAG Retriever for Strength & Conditioning Corrections.
    Bridges kinematic fault detections with peer-reviewed literature.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve_corrections_for_faults(
        self,
        exercise: str,
        faults: List[Fault],
        top_k_per_fault: int = 1
    ) -> Dict[str, Any]:
        """
        Takes detected kinematic faults from the movement engine and retrieves
        corresponding scientific root causes, verbal cues, and corrective drills.
        """
        unique_fault_codes = list(set(f.code for f in faults))
        retrieved_items: List[Dict[str, Any]] = []
        seen_cards = set()

        for fault in faults:
            if fault.code in seen_cards:
                continue
            seen_cards.add(fault.code)

            # Build semantic query from fault details
            query = f"{fault.name} {fault.description} {exercise}"
            results = self.vector_store.search(query=query, exercise=exercise, top_k=top_k_per_fault)

            if results:
                card, score = results[0]
                retrieved_items.append({
                    "fault_code": fault.code,
                    "fault_name": fault.name,
                    "severity": fault.severity.value,
                    "observed_metric": fault.description,
                    "similarity_score": round(score, 3),
                    "source": card.get("source", "Evidence-Based Strength Science"),
                    "biomechanical_root_cause": card.get("biomechanical_root_cause"),
                    "verbal_cues": card.get("verbal_cues", []),
                    "corrective_drills": card.get("corrective_drills", []),
                })

        return {
            "exercise": exercise,
            "total_fault_types_queried": len(unique_fault_codes),
            "retrieved_protocols": retrieved_items,
        }

    def format_context_for_prompt(self, retrieval_result: Dict[str, Any]) -> str:
        """
        Formats retrieved literature into a clean, markdown-grounded context block
        for the Groq LLM prompt.
        """
        protocols = retrieval_result.get("retrieved_protocols", [])
        if not protocols:
            return "No specific movement faults detected. Set was performed with standard kinematic alignment."

        lines = ["### EVIDENCE-BASED STRENGTH & CONDITIONING CONTEXT (GROUND TRUTH):"]
        for p in protocols:
            lines.append(f"\n#### FAULT PATTERN: {p['fault_name']} [{p['fault_code']}]")
            lines.append(f"- **Primary Literature Source**: {p['source']}")
            lines.append(f"- **Biomechanical Root Cause**: {p['biomechanical_root_cause']}")
            lines.append("- **Verified Verbal Cues**:")
            for cue in p["verbal_cues"]:
                lines.append(f"  * \"{cue}\"")
            lines.append("- **Targeted Corrective Drills**:")
            for drill in p["corrective_drills"]:
                lines.append(f"  * **{drill['drill']}** ({drill['prescription']}): {drill['coaching_points']}")

        return "\n".join(lines)
