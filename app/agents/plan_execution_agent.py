import operator
from typing import Annotated, List, Tuple, Union

from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[BaseMessage], operator.add]
    current_step: Annotated[int, operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""
    response: str


class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

class PlanExecuteAgent:
    """Agent that plans steps before execution."""

    def __init__(self, tools):
        self.tools_description = "\n".join([f"Tool name: {tool.name}, tool description: {tool.description}" for tool in tools])
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.executor = self._create_executor(tools)
        self.planner = self._create_planner()
        self.re_planner = self._create_re_planner()
        self.app = self._build_graph()

    def run(self, message: str):
        config = {"recursion_limit": 20}
        return self.app.invoke({"input": message}, config=config)

    def _build_graph(self):
        """Build graph."""
        workflow = StateGraph(PlanExecute)

        workflow.add_node("planner", self._plan_step)
        workflow.add_node("agent", self._execute_step)
        workflow.add_node("re_plan", self._re_plan_step)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue_execution,
            ["re_plan", "agent"],
        )

        workflow.add_conditional_edges(
            "re_plan",
            self._should_end,
            ["agent", END],
        )
        return workflow.compile()

    def _execute_step(self, state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))

        current_step = state["current_step"] or 0

        task = plan[current_step]

        agent_response = self.executor.invoke({"messages": state["past_steps"] ,"plan": plan_str, "current_step": current_step, "task": task})

        outputs = []
        for tool_call in agent_response.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {
            "past_steps": [agent_response, *outputs],
            "current_step": 1,
        }

    def _plan_step(self, state: PlanExecute):
        plan = self.planner.invoke({"messages": [("user", state["input"])], "tools" : self.tools_description})
        return {"plan": plan.steps}

    def _re_plan_step(self, state: PlanExecute):
        output = self.re_planner.invoke({**state, "tools" : self.tools_description})
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    @staticmethod
    def _should_continue_execution(state: PlanExecute):
        if state["plan"] and state["current_step"] < len(state["plan"]):
            return "agent"
        else:
            return "re_plan"

    @staticmethod
    def _should_end(state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        else:
            return "agent"

    @staticmethod
    def _create_executor(tools):
        llm = ChatOpenAI(model_name="gpt-4o-mini")
        llm_with_tools = llm.bind_tools(tools)

        executor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant.
                    for the following plan:
                    {plan}
            
                    You are tasked with executing step {current_step}, {task}.
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        return executor_prompt | llm_with_tools

    @staticmethod
    def _create_planner():
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the given objective, come up with a simple step by step plan.
                    
                    You have the following tools to use them to achieve the objective:
                    {tools}
                    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \

                    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
                ),
                ("placeholder", "{messages}"),
            ]
        )

        return planner_prompt | ChatOpenAI(
            model_name="gpt-4o-mini", temperature=0
        ).with_structured_output(Plan)

    @staticmethod
    def _create_re_planner():
        re_planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the given objective, come up with a simple step by step plan.
            You have the following tools to use them to achieve the objective:
                    {tools}
                    
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.    
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        Your objective was this:
        {input}

        Your original plan was this:
        {plan}

        Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. 
        Otherwise, fill out the plan. 
        Only add steps to the plan that still NEED to be done. 
        Do not return previously done steps as part of the plan.
        THINK ABOUT SAVING TOKENS""",
                ),
                ("placeholder", "{past_steps}"),
            ]
        )

        return re_planner_prompt | ChatOpenAI(
            model_name="gpt-4o-mini", temperature=0
        ).with_structured_output(Act)