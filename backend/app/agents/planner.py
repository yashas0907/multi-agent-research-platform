"""Planner Agent — decomposes the research objective into a structured plan."""
from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent, new_id
from app.schemas.research import ResearchPlan, ResearchState, SubQuestion


class PlannerAgent(BaseAgent):
    name = "planner"
    prompt_name = "planner"

    async def run(self, state: ResearchState) -> ResearchPlan:
        user_payload = self._prompt.render_user(
            question=state.question,
            depth=state.depth.value,
            format=state.requested_format.value,
        )
        plan = await self._structured(user_payload, ResearchPlan)

        # Enforce depth profile limits (cost control — never trust the model
        # to respect the budget).
        max_sq = self.ctx.profile["max_subquestions"]
        if len(plan.subquestions) > max_sq:
            plan.subquestions = plan.subquestions[:max_sq]

        state.research_plan = plan
        state.subquestions = [
            SubQuestion(id=new_id("sq"), text=sq) for sq in plan.subquestions
        ]
        await self.emit(
            "info",
            f"Plan created: {len(plan.subquestions)} subquestions, "
            f"{len(plan.required_evidence)} evidence requirements",
            subquestions=len(plan.subquestions),
            question_type=plan.question_type,
        )
        return plan
