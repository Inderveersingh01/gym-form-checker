import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq

from coaching.rag.retriever import BiomechanicsRAGRetriever
from exercises.base import SetAnalysisResult


class GroqCoachEngine:
    """
    RAG-grounded AI Strength & Conditioning Coach powered by Groq LLM.
    Combines numerical kinematic telemetry with evidence-based sports science literature.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-20b",
        retriever: Optional[BiomechanicsRAGRetriever] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Please set it in .env file.")

        self.client = Groq(api_key=self.api_key, timeout=15.0)
        self.model_name = model_name
        self.retriever = retriever or BiomechanicsRAGRetriever()

    def generate_coach_report(
        self,
        set_result: SetAnalysisResult,
        user_name: str = "Lifter",
        skill_level: str = "Intermediate"
    ) -> Dict[str, Any]:
        """
        Generates a grounded, actionable coaching report from SetAnalysisResult.
        """
        # 1. Flatten all unique faults across reps
        all_faults = [f for r in set_result.reps for f in r.faults]

        # 2. Retrieve grounded evidence from RAG Vector DB
        rag_data = self.retriever.retrieve_corrections_for_faults(
            exercise=set_result.exercise,
            faults=all_faults,
            top_k_per_fault=1
        )
        rag_context_str = self.retriever.format_context_for_prompt(rag_data)

        # 3. Format telemetry summary
        telemetry_summary = []
        for rep in set_result.reps:
            telemetry_summary.append(
                f"- Rep {rep.rep_number}: Duration {rep.duration_seconds:.2f}s "
                f"(Eccentric {rep.eccentric_seconds:.2f}s, Concentric {rep.concentric_seconds:.2f}s) | "
                f"Status: {'PASS' if rep.passed else 'FAIL'} | "
                f"Metrics: {rep.metrics}"
            )
        telemetry_str = "\n".join(telemetry_summary)

        # 4. Construct System & User Prompt
        system_prompt = (
            "You are an elite Certified Strength & Conditioning Specialist (CSCS) and Olympic weightlifting coach.\n"
            "Your objective: Deliver an encouraging, technically precise, and actionable post-set coaching debrief.\n\n"
            "STRICT GROUNDING INSTRUCTIONS:\n"
            "1. Ground all root cause explanations, verbal cues, and corrective drills STRICTLY in the provided RAG Context.\n"
            "2. DO NOT invent or hallucinate corrective exercises not listed in the Ground Truth.\n"
            "3. Cite the literature source (e.g. 'According to NSCA / Starting Strength...') when explaining why a fault happened.\n"
            "4. Keep the tone motivational, authoritative, and direct (like an Olympic coach in the gym with the athlete)."
        )

        user_prompt = f"""
ATHLETE PROFILE:
- Name: {user_name}
- Skill Level: {skill_level}
- Exercise: {set_result.exercise.upper()}
- Total Repetitions: {set_result.total_reps} (Successful: {set_result.successful_reps})
- Overall Form Score: {set_result.overall_score:.1f} / 100

KINEMATIC TELEMETRY LOG:
{telemetry_str}

{rag_context_str}

TASK:
Write a structured, professional coaching report with the following 5 sections:
1. **Set Executive Summary & Form Score** (Score / 100 with a quick verdict)
2. **Rep-by-Rep Telemetry Breakdown** (Highlights of depth, angles, and tempo)
3. **Biomechanical Root Cause Analysis** (Explain the physical cause of detected faults using the literature)
4. **The #1 Focus Cue for Your Next Set** (Single most impactful cue the lifter should remember)
5. **Recommended Warmup / Corrective Protocol** (Prescribe the exact drills retrieved from the context)
"""

        # 5. Call Groq with fallback model support
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=1024,
            )
            report_text = response.choices[0].message.content
        except Exception as e:
            # Fallback to high-speed 20b model if 120b is busy or rate limited
            fallback_model = "openai/gpt-oss-20b"
            print(f"Notice: Primary model '{self.model_name}' encountered: {e}. Switching to '{fallback_model}'...")
            response = self.client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=1024,
            )
            report_text = response.choices[0].message.content

        return {
            "exercise": set_result.exercise,
            "overall_score": set_result.overall_score,
            "rag_protocols": rag_data,
            "coaching_report": report_text,
        }
