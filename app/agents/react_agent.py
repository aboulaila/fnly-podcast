from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[BaseMessage], add_messages]

class ReActAgent:
    """Agent that reacts to user input."""
    def __init__(self, llm, prompt: ChatPromptTemplate, tools, memory):
        self.model = llm
        self.prompt = prompt
        self.model = self.model.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.memory = memory
        self._build_graph()

    def invoke(self, message: str, config: RunnableConfig = None):
        """Invoke the agent."""
        return self.graph.invoke({"messages": [("user", message)]}, config)

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("model", self._call_model)
        workflow.add_node("tools", self._tool_node)

        workflow.set_entry_point("model")

        workflow.add_conditional_edges("model",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )

        workflow.add_edge("tools", "model")
        self.graph = workflow.compile(checkpointer=self.memory)

    def _tool_node(self, state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            print(f"\033[94m{tool_result}\033[0m")
        return {"messages": outputs}

    def _call_model(self, state: AgentState, config: RunnableConfig):
        system_prompt = self.prompt.invoke(state, config).to_messages()
        response = self.model.invoke(system_prompt + state["messages"], config)
        print(f"\033[92m{response}\033[0m")
        return {"messages": [response]}

    @staticmethod
    def _should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"