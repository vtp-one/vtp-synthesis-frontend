from typing import Optional
import reflex as rx
import time
import os
import rich
import ollama
import traceback
import httpx
import json
import pydantic
import sqlmodel
import sqlalchemy
import uuid
import functools
import pymongo
import asyncio
from copy import copy

from .utils import get_substate

from ..state import BACKGROUND_TASKS, GlobalState

###
#
#
MONGO_HOST = os.environ.get("MONGODB_HOST", "localhost")
MONGO_PORT = os.environ.get("MONGODB_PORT", 27017)
MONGO_USER = os.environ.get("MONGODB_USER", None)
MONGO_PASSWORD = os.environ.get("MONGODB_PASSWORD", None)
MONGO_CLIENT = lambda: pymongo.MongoClient(
    host=MONGO_HOST,
    port=int(MONGO_PORT),
    username=MONGO_USER,
    password=MONGO_PASSWORD)

MONGO_DB = lambda: MONGO_CLIENT().synthesis

#
###

###
#
GLOBAL_MESSAGE_STYLE = dict()
GLOBAL_MESSAGE_STYLE["display"] = "inline-block"
GLOBAL_MESSAGE_STYLE["padding"] = "1em"
GLOBAL_MESSAGE_STYLE["border_radius"] = "8px"
GLOBAL_MESSAGE_STYLE["max_width"] = ["30em", "30em", "50em", "50em", "50em", "50em"]

#
###

###
#
SYSTEM_PROMPT = """
You are SYNTHESIS. You respond to user input however you see fit.
Be imaginative but keep your response focused on the user prompt.
""".lstrip().rstrip()

class SystemPrompt(rx.Model):
    role: str = "system"
    pname: str
    content: str

DEFAULT_PROMPT = SystemPrompt(pname="Default", content=SYSTEM_PROMPT)

#
###


###
#
def _time():
    return str(int(time.time()))

class Message(rx.Base):
    pass

class Message(Message):
    #id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    role: str
    content: str
    timestamp: int
    kind: str = "None"
    response: Optional[Message] = None

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs["timestamp"] = int(time.time())
        return cls(*args, **kwargs)

DEFAULT_MESSAGE = Message.create(role="user", content="TEST MESSAGE", kind="debug")
long_message = Message(
    role="debug",
    content="<br/>".join([str(x) for x in range(100)]),
    timestamp="0")

class Plan(rx.Base):
    content: str
    timestamp: int
    kind: str = "None"

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs["timestamp"] = int(time.time())
        return cls(*args, **kwargs)


DEFAULT_PLAN = Plan(content="NO PLAN", timestamp=-1, kind="debug")
long_plan = Plan.create(content="<br/>".join([str(x) for x in range(100)]))


class ContextObject(rx.Base):
    cname: str
    cid: int
    did: str = "-1"
    content: str

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs["cid"] = kwargs.get("cid") or int(time.time())
        kwargs["did"] = str(kwargs["cid"])
        return cls(*args, **kwargs)

    @classmethod
    def default(cls, *args, **kwargs):
        kwargs["cname"] = "None"
        kwargs["cid"] = -1
        kwargs["content"] = "NO CONTEXT ITEMS"

        return cls.create(*args, **kwargs)

class SessionObject(rx.Base):
    sname: str
    key: str
    state: dict = {}
    created: str
    updated: str

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs["key"] = kwargs.get("key") or str(uuid.uuid4())
        kwargs["created"] = kwargs.get("created") or str(int(time.time()))
        kwargs["updated"] = kwargs.get("updated") or str(int(time.time()))
        return cls(*args, **kwargs)

    @classmethod
    def default(cls, *args, **kwargs):
        kwargs["sname"] = "Default Session"

        return cls.create(*args, **kwargs)

DEFAULT_SESSION = SessionObject.create(sname="Default Session")

#
###

###
#
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_PORT: str = os.environ.get("OLLAMA_PORT", "11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "dolphin-llama3:8b-256k-v2.9-q8_0")

TOOLKIT_HOST: str = os.environ.get("TOOLKIT_HOST", "localhost")
TOOLKIT_PORT: str = os.environ.get("TOOLKIT_PORT", "42024")

TOOLKIT_OLLAMA_ROOT: str = os.environ.get("TOOLKIT_OLLAMA_ROOT", "ollama")
TOOLKIT_OLLAMA_TARGET: str = os.environ.get("TOOLKIT_OLLAMA_TARGET", "chat/completions")

TOOLKIT_LANGCHAIN_ROOT: str = os.environ.get("TOOLKIT_LANGCHAIN_ROOT", "langchain")
TOOLKIT_LANGCHAIN_CHAT: str = os.environ.get("TOOLKIT_LANGCHAIN_CHAT", "chat")

#TOOLKIT_ENDPOINTS: dict = httpx.get(f"http://{TOOLKIT_HOST}:{TOOLKIT_PORT}/dir").json()


API_TIMEOUT = None
class CallState(rx.State):
    processing: bool = False
    ollama_model: str = OLLAMA_MODEL
    ollama_host: str = OLLAMA_HOST
    ollama_port: str = OLLAMA_PORT

    toolkit_host: str = TOOLKIT_HOST
    toolkit_port: str = TOOLKIT_PORT

    toolkit_ollama_root: str = TOOLKIT_OLLAMA_ROOT
    toolkit_ollama_target: str = TOOLKIT_OLLAMA_TARGET

    toolkit_langchain_root: str = TOOLKIT_LANGCHAIN_ROOT
    toolkit_langchain_chat: str = TOOLKIT_LANGCHAIN_CHAT

    @rx.background
    async def ollama_call(self):
        rich.print(f"OLLAMA_CALL")
        #
        # TODO
        # => need to make this work for other system_output panels
        # => need to get the n1 from the current user message object
        # => to get the correct substate
        #
        output = None
        try:
            # set processing
            async with self:
                self.processing = True
                yield

            state = GlobalState.bundle_state(self)

            # render output message
            output = Message.create(
                role="assistant",
                content="",
                kind="init")

            state["system_output_state"].message_list.append(output)
            yield

            output = state["system_output_state"].message_list[-1]
            state["user_input_state"].message_list[-1].response = output
            yield

            # get client
            host = f"http://{self.ollama_host}:{self.ollama_port}"
            client = ollama.Client(host=host)

            if self.ollama_model not in [x["name"] for x in client.list()["models"]]:
                e = f"OLLAMA_CALL - MODEL NOT AVAILABLE - {self.ollama_model}"
                raise RuntimeError(e)

            # bundle message output
            messages = []
            messages.append({
                "role":state["system_prompt_state"].system_prompt.role,
                "content":state["system_prompt_state"].system_prompt.content.lstrip().rstrip()
                })

            if plan := state["plan_state"].plan:
                messages.append({
                    "role":"system",
                    "content":f"CURRENT PLAN: {plan.content}".lstrip().rstrip()
                    })

            if len(state["context_state"].context_list):
                for context in state["context_state"].context_list:
                    messages.append({
                        "role":"system",
                        "content":f"CONTEXT: {context.cname} = {context.content}".lstrip().rstrip()
                        })

            for m in state["user_input_state"].message_list:
                if m.response is None:
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)

                elif m.response.kind == "init":
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)
                    output.kind = "stream"

                elif m.response.kind == "response":
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)

                    out = {"role":m.response.role, "content":m.response.content}
                    messages.append(out)

            # call ollama
            rich.print(f"OLLAMA_CALL: MESSAGES")
            rich.print(messages)

            session = client.chat(
                model=self.ollama_model,
                messages=messages,
                stream=True
                )

            for item in session:
                rich.print(f"CHAT_ITEM: {item}")

                content = item["message"].get("content", "")
                output.content += content
                yield

            output.kind = "response"
            yield

        except Exception as exc:
            if output is None:
                raise

            rich.print(exc)
            output.content = traceback.format_exc()
            output.kind = "error"
            yield

        finally:
            async with self:
                self.processing = False
                yield

    @rx.background
    async def toolkit_ollama_call(self):
        rich.print(f"TOOLKIT_OLLAMA_CALL")
        output = None
        try:
            # set processing
            async with self:
                self.processing = True
                yield

            state = GlobalState.bundle_state(self)

            # render output message
            output = Message.create(
                role="assistant",
                content="",
                kind="init")

            state["system_output_state"].message_list.append(output)
            yield

            output = state["system_output_state"].message_list[-1]
            state["user_input_state"].message_list[-1].response = output
            yield

            # build params
            params = {}
            params["model"] = self.ollama_model
            params["stream"] = True

            messages = []

            messages.append({
                "role":"system",
                "content":state["system_prompt_state"].system_prompt.content.lstrip().rstrip()
                })

            for m in state["user_input_state"].message_list:
                if m.response is None:
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)

                elif m.response.kind == "init":
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)
                    output.kind = "stream"

                elif m.response.kind == "response":
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)

                    out = {"role":m.response.role, "content":m.response.content}
                    messages.append(out)

            params["messages"] = messages

            # get stream
            target = f"http://{self.toolkit_host}:{self.toolkit_port}/{self.toolkit_ollama_root}/{self.toolkit_ollama_target}"
            rich.print(f"PARAMS: {params}")
            with httpx.stream("POST", target, json=params, timeout=API_TIMEOUT) as stream:
                stream.raise_for_status()
                for item in stream.iter_bytes():
                    item = json.loads(item)
                    rich.print(f"ITEM: {item}")
                    content = item["choices"][0]["delta"]["content"]
                    output.content += content
                    yield

            output.kind = "response"
            yield

        except Exception as exc:
            if output is None:
                raise

            rich.print(exc)
            output.content = traceback.format_exc()
            output.kind = "error"
            yield

        finally:
            async with self:
                self.processing = False
                yield

    @rx.background
    async def toolkit_langchain_chat_call(self):
        rich.print(f"TOOLKIT_LANGCHAIN_CHAT_CALL")
        output = None
        try:
            # set processing
            async with self:
                self.processing = True
                yield

            state = GlobalState.bundle_state(self)

            # render output message
            output = Message.create(
                role="assistant",
                content="",
                kind="init")

            state["system_output_state"].message_list.append(output)
            yield

            output = state["system_output_state"].message_list[-1]
            state["user_input_state"].message_list[-1].response = output
            yield

            # build params
            params = {}
            params["model_type"] = self.ollama_model
            params["stream"] = True

            params["system_prompt"] = state["system_prompt_state"].system_prompt.content.lstrip().rstrip()

            messages = []
            for m in state["user_input_state"].message_list[:-1]:
                if m.response is None:
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)

                elif m.response.kind == "init":
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)
                    output.kind = "stream"

                elif m.response.kind == "response":
                    out = {"role":m.role, "content":m.content}
                    messages.append(out)

                    out = {"role":m.response.role, "content":m.response.content}
                    messages.append(out)

            params["history_list"] = messages

            user_prompt = state["user_input_state"].message_list[-1]
            params["user_prompt"] = {
                "role":user_prompt.role,
                "content":user_prompt.content
                }

            context_list = []
            #
            # TODO
            # => seperate planning
            #
            if plan := state["plan_state"].plan:
                context_list.append({
                    "key":"plan",
                    "content":plan.content
                    })

            if len(state["context_state"].context_list):
                for context in state["context_state"].context_list:
                    context_list.append({
                        "key":context.cname,
                        "content":context.content
                        })

            params["context_list"] = context_list

            # get stream
            target = f"http://{self.toolkit_host}:{self.toolkit_port}/{self.toolkit_langchain_root}/{self.toolkit_langchain_chat}"
            #rich.print(f"PARAMS: {params}")
            with httpx.stream("POST", target, json=params, timeout=API_TIMEOUT) as stream:
                stream.raise_for_status()
                for item in stream.iter_bytes():
                    item = json.loads(item)
                    match item["event"]:
                        # ON_PROMPT
                        case "on_prompt_start":
                            pass

                        case "on_prompt_end":
                            pass


                        # ON_CHAT | ON_LLM
                        case "on_chat_model_start" | "on_llm_start":
                            rich.print(item)

                        case "on_chat_model_stream" | "on_llm_stream":
                            #rich.print(item)
                            content = item["data"]["chunk"]["content"]
                            if content == "":
                                output.content += "<br/>"

                            else:
                                output.content += content

                            yield

                        case "on_chat_model_end" | "on_llm_end":
                            pass


                        # ON_PARSER
                        case "on_parser_start":
                            pass

                        case "on_parser_stream":
                            pass

                        case "on_parser_end":
                            pass


                        # ON_CHAIN
                        case "on_chain_start":
                            pass

                        case "on_chain_stream":
                            pass

                        case "on_chain_end":
                            pass


                        # ON_TOOL
                        case "on_tool_start":
                            pass

                        case "on_tool_stream":
                            pass

                        case "on_tool_end":
                            pass


                        # ON_RETRIEVER
                        case "on_retriever_start":
                            pass

                        case "on_retriever_chunk":
                            pass

                        case "on_retriever_end":
                            pass


                        case _:
                            rich.print(f"ITEM: {item}")
                            ev = item["event"]
                            e = f"UNHANDLED EVENT - {ev}"
                            raise RuntimeError(e)


                    #content = item["event"]
                    #output.content += content
                    #output.content += "<br/>"
                    #yield

            output.kind = "response"
            yield

        except Exception as exc:
            if output is None:
                raise

            rich.print(exc)
            #output.content = traceback.format_exc()
            output.kind = "error"
            yield

            # render output message
            error = Message.create(
                role="assistant",
                content=traceback.format_exc(),
                kind="error")

            state["system_output_state"].message_list.append(error)
            yield

        finally:
            async with self:
                self.processing = False
                yield


    #
    ###

CALL_MAP = {}
CALL_MAP["ollama-chat"] = CallState.ollama_call
CALL_MAP["toolkit-ollama"] = CallState.toolkit_ollama_call
CALL_MAP["langchain-chat"] = CallState.toolkit_langchain_chat_call

SUBMIT_DEFAULT_TARGET = "ollama-chat"
SUBMIT_DEFAULT = ""

#
###

###
#
USER_MESSAGE_STYLE = dict()
USER_MESSAGE_STYLE["background_color"] = rx.color("accent", 4)
USER_MESSAGE_STYLE["color"] = rx.color("black", 12)

class UserInputState(rx.State):
    message_list: list[Message] = []

    submit_target: str = SUBMIT_DEFAULT_TARGET
    submit_default: str = SUBMIT_DEFAULT

    ###
    #
    def submit(self, data):
        content = data["user_input"]

        if content != "":
            m = Message.create(
                role="user",
                content=content,
                kind=f"user_{self.submit_target}")

            self.message_list.append(m)
            yield

            return CALL_MAP.get(self.submit_target)

    def message_clear(self):
        self.message_list = []
        yield

    #
    ###

    @rx.background
    async def auto_scroll(self):
        previous = None
        while True:
            if len(self.message_list):
                current = self.message_list[-1]
                scroll = False
                if previous is None:
                    scroll = True

                elif current.timestamp != previous.timestamp:
                    scroll = True

                elif current.content != previous.content:
                    scroll = True

                if scroll:
                    previous = current
                    div_target = f"user_input-{current.timestamp}"
                    yield rx.scroll_to(div_target)

            await asyncio.sleep(1e-1)

class UserInput(rx.ComponentState):
    @staticmethod
    def message_render(message: Message) -> rx.Component:
        output = rx.card(
            rx.hstack(
                rx.vstack(
                    rx.text(f"Owner: {message.role}"),
                    rx.text(f"Time: {message.timestamp}"),
                ),
                rx.markdown(
                    message.content,
                    flex="1",
                    **USER_MESSAGE_STYLE,
                    **GLOBAL_MESSAGE_STYLE,
                    ),
                rx.box(id=f"user_input-{message.timestamp}")
                ),
            width="100%",
            )

        return output

    @staticmethod
    def history_render(cls):
        output = rx.scroll_area(
            rx.flex(
                rx.cond(UserInputState.message_list.length(),
                    rx.foreach(UserInputState.message_list, UserInput.message_render),
                    UserInput.message_render(Message(role="system", content="No Messages", timestamp="0"))
                    ),
                direction="column",
                spacing="1",
                overflow="hidden"
                ),
            type="auto",
            scrollbars="both",
            max_height="75vh",
            id="user_output"
            )

        return output

    @staticmethod
    def input_render(cls):
        output = rx.vstack(
            UserInput.input_form(cls),
            position="sticky",
            bottom="0",
            left="0",
            padding_y="16px",
            backdrop_filter="auto",
            backdrop_blur="lg",
            border_top=f"1px solid {rx.color('orange', 3)}",
            background_color=rx.color("gray", 2),
            align_items="stretch",
            width="100%"
            )

        return output

    @staticmethod
    def input_form(cls):
        output = rx.vstack(
                rx.hstack(
                    rx.select(
                        list(CALL_MAP.keys()),
                        placeholder="target",
                        label="target",
                        name="target",
                        default_value=UserInputState.submit_target,
                        on_change=UserInputState.set_submit_target,
                    ),
                ),
                rx.form(
                    rx.hstack(
                        rx.input(
                            default_value=UserInputState.submit_default,
                            placeholder="User Input",
                            name="user_input",
                            flex="1"
                        ),
                        rx.button("⏩",
                            type="submit",
                            loading=CallState.processing
                        ),
                    ),
                    on_submit=lambda x: UserInputState.submit(x),
                    reset_on_submit=True,
                ),
            )
        return output

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = rx.flex(
            rx.card(
                rx.hstack(
                    rx.heading(f"User Output", size="3"),
                    rx.spacer(),
                    rx.button(
                        "CLEAR",
                        color_scheme="orange",
                        radius="none",
                        on_click=UserInputState.message_clear
                        ),
                    ),
                ),
            rx.card(
                cls.history_render(cls),
                rx.spacer(),
                cls.input_render(cls),
                flex="1",
                ),
            width="100%",
            spacing="1",
            direction="column"
            )

        return output

    #
    ###

BACKGROUND_TASKS.append(UserInputState.auto_scroll)

#
###

###
#
SYSTEM_MESSAGE_STYLE = dict()
#SYSTEM_MESSAGE_STYLE["background_color"] = rx.color("green", 4)
SYSTEM_MESSAGE_STYLE["color"] = rx.color("black", 12)

class SystemOutputState(rx.State):
    message_list: list[Message] = []

    def message_clear(self):
        self.message_list = []
        yield

    @rx.background
    async def auto_scroll(self):
        previous = None
        while True:
            if len(self.message_list):
                current = self.message_list[-1]
                scroll = False
                if previous is None:
                    scroll = True

                elif current.timestamp != previous.timestamp:
                    scroll = True

                elif current.content != previous.content:
                    scroll = True

                if scroll:
                    previous = current
                    div_target = f"system_output-{current.timestamp}"
                    yield rx.scroll_to(div_target)

            await asyncio.sleep(1e-1)

class SystemOutput(rx.ComponentState):
    @staticmethod
    def message_render(message):
        output = rx.card(
            rx.hstack(
                rx.vstack(
                    rx.text(f"Owner: {message.role}"),
                    rx.text(f"Time: {message.timestamp}"),
                ),
                rx.markdown(
                    message.content,
                    flex="1",
                    **SYSTEM_MESSAGE_STYLE,
                    **GLOBAL_MESSAGE_STYLE,
                    background_color=rx.match(
                        message.kind,
                        ("error", rx.color("red", 4)),
                        ("response", rx.color("green", 4)),
                        rx.color("blue", 4)
                        )
                    ),
                rx.box(id=f"system_output-{message.timestamp}")
                ),
            width="100%",
            )

        return output

    @staticmethod
    def history_render(cls):
        output = rx.scroll_area(
            rx.flex(
                rx.cond(SystemOutputState.message_list.length(),
                    rx.foreach(SystemOutputState.message_list, SystemOutput.message_render),
                    SystemOutput.message_render(Message(role="system", content="No Messages", timestamp="0"))
                    ),
                direction="column",
                spacing="1",
                overflow="hidden"
                ),
            type="auto",
            scrollbars="both",
            max_height="80vh",
            id="system_output"
            )

        return output

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = rx.flex(
            rx.card(
                rx.hstack(
                    rx.heading(f"System Output", size="3"),
                    rx.spacer(),
                    rx.button(
                        "CLEAR",
                        color_scheme="orange",
                        radius="none",
                        on_click=SystemOutputState.message_clear
                        ),
                    ),
                ),
            rx.card(
                cls.history_render(cls),
                rx.spacer(),
                flex="1"
                ),
            width="100%",
            spacing="1",
            direction="column"
            )

        return output

    #
    ###

BACKGROUND_TASKS.append(SystemOutputState.auto_scroll)

#
###





###
#
class SystemPromptState(rx.State):
    system_prompt: SystemPrompt = DEFAULT_PROMPT

    def update_prompt(self, data):
        self.system_prompt.content = data

class SystemPromptColumn(rx.ComponentState):
    @staticmethod
    def render(prompt: SystemPrompt):
        output = rx.text_area(
            value=prompt.content,
            height="70vh",
            on_change=SystemPromptState.update_prompt
            )

        return output

    @classmethod
    def get_component(cls, **props):
        output = rx.flex(
            rx.card(
                rx.hstack(
                    rx.heading("System Prompt", size="3"),
                    rx.spacer(),
                    rx.text(SystemPromptState.system_prompt.pname),
                     ),
                ),
            rx.card(
                rx.vstack(
                    rx.scroll_area(
                        rx.flex(
                            cls.render(SystemPromptState.system_prompt),
                            direction="column",
                            spacing="1",
                            overflow="hidden"
                            ),
                        type="auto",
                        scrollbars="both",
                        max_height="80vh",
                        id="plan_output"
                        ),
                    )
                ),
            width="100%",
            spacing="1",
            direction="column"
            )

        return output

    #
    ###

#
###

###
#
PLAN_MESSAGE_STYLE = dict()
PLAN_MESSAGE_STYLE["background_color"] = rx.color("yellow", 4)
PLAN_MESSAGE_STYLE["color"] = rx.color("black", 12)

class PlanState(rx.State):
    plan: Optional[Plan] = None
    plan_content: str = ""

    ###
    #
    def plan_clear(self):
        self.plan = None
        yield

    def plan_edit(self, data):
        self.plan =  Plan.create(**data)
        yield

    #
    ###

    ###
    #
    @staticmethod
    def modal_gen(parent, plan):
        output = rx.dialog.root(
            rx.dialog.trigger(rx.button("EDIT", color_scheme="orange", radius="none")),
            rx.dialog.content(
                    rx.flex(
                        rx.dialog.title("Edit Plan"),
                        rx.form(
                            parent.modal_input(parent, plan),
                            parent.modal_buttons(parent),
                            on_submit=lambda x: parent.plan_edit(x)
                        ),
                    direction="column",
                    flex="1",
                    width="100%",
                    spacing="1",
                    )
                )
            )

        return output

    @staticmethod
    def modal_input(parent, plan):
        PlanState.plan_content = copy(plan.content)
        output = rx.fragment(
            rx.card(
                rx.text("Plan Kind"),
                rx.input(
                    default_value=plan.kind,
                    placeholder="Plan Kind",
                    name="kind"
                    ),
                flex="1"
            ),
            rx.card(
                rx.text("Plan Content"),
                rx.text_area(
                    value=PlanState.plan_content,
                    placeholder="Plan Content",
                    name="content",
                    on_change=PlanState.set_plan_content,
                    height="40vh"
                    ),
                flex="1"
            ),
        )

        return output

    @staticmethod
    def modal_buttons(parent):
        output = rx.card(
            rx.center(
                rx.hstack(
                    rx.dialog.close(
                        rx.button("Cancel",
                            color_scheme="gray",
                            variant="soft"),
                        ),
                    rx.dialog.close(
                        rx.button("Save",
                            color_scheme="green",
                            variant="soft",
                            type="submit"),
                        ),
                    align_items="stretch"
                    )
                )
            )

        return output

    @staticmethod
    def plan_edit_modal(plan):
        output = PlanState.modal_gen(PlanState, plan)

        return output

    #
    ###

class PlanColumn(rx.ComponentState):
    ###
    #
    def plan_clear(self):
        self.plan = None
        yield


    #
    ###

    ###
    #
    @staticmethod
    def plan_render(plan: Plan) -> rx.Component:
        output = rx.card(
            rx.hstack(
                rx.vstack(
                    rx.text(f"Kind: {plan.kind}"),
                    rx.text(f"Time: {plan.timestamp}"),
                    PlanState.plan_edit_modal(plan),
                ),
                rx.markdown(
                    plan.content,
                    flex="1",
                    **PLAN_MESSAGE_STYLE,
                    **GLOBAL_MESSAGE_STYLE,
                    ),
                ),
            width="100%",
            )

        return output

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = rx.flex(
            rx.card(
                rx.hstack(
                    rx.heading("Plan Output", size="3"),
                    rx.spacer(),
                    rx.button(
                        "CLEAR",
                        color_scheme="orange",
                        radius="none",
                        on_click=PlanState.plan_clear
                        ),
                    ),
                ),
            rx.card(
                rx.scroll_area(
                    rx.flex(
                        rx.cond(PlanState.plan,
                            cls.plan_render(PlanState.plan),
                            cls.plan_render(DEFAULT_PLAN)
                            ),
                        direction="column",
                        spacing="1",
                        overflow="hidden"
                        ),
                    type="auto",
                    scrollbars="both",
                    max_height="80vh",
                    id="plan_output"
                    ),
                ),
            width="100%",
            spacing="1",
            direction="column"
            )

        return output

    #
    ###

#
###

###
#
CONTEXT_STYLE = dict()
CONTEXT_STYLE["background_color"] = rx.color("orange", 4)
CONTEXT_STYLE["color"] = rx.color("black", 12)

class ContextState(rx.State):
    context_list: list[ContextObject] = []
    context_content: str = ""

    ###
    #
    def context_new(self, data):
        context = ContextObject.create(**data)
        self.context_list.append(context)
        yield

        div_target = f"context-{context.cid}"
        yield rx.scroll_to(div_target)

    def context_clear(self):
        self.context_list = []
        yield

    def context_edit(self, data):
        if int(data["did"]) > 0:
            target = None
            for obj in self.context_list:
                rich.print(f"OBJ: {obj}")
                if obj.did == data["did"]:
                    target = obj
                    break

            if target:
                target.content = data["content"]
                target.cname = data["cname"]
                yield

            else:
                did = data["did"]
                print(f"NO TARGET - {did}")

        self.context_content = ""

    def context_delete(self, cid):
        if cid > 0:
            target = None
            for obj in self.context_list:
                if cid == obj.cid:
                    target = obj
                    break

            if target:
                self.context_list.remove(target)
                yield

    #
    ###

    ###
    #
    @staticmethod
    def modal_edit_gen(context):
        output = rx.dialog.root(
            rx.dialog.trigger(rx.button("EDIT", color_scheme="orange", radius="none", disabled=rx.cond(context.cid > 0, False, True),)),
            rx.dialog.content(
                    rx.flex(
                        rx.dialog.title("Edit Context Object"),
                        rx.form(
                            ContextState.modal_edit_input(context),
                            ContextState.modal_edit_buttons(),
                            rx.input(
                                default_value=context.did,
                                placeholder="Context ID",
                                name="did",
                                display="none"
                                ),
                            on_submit=lambda x: ContextState.context_edit(x)
                        ),
                    direction="column",
                    flex="1",
                    width="100%",
                    spacing="1",
                    )
                )
            )

        return output

    @staticmethod
    def modal_edit_input(context):
        ContextState.context_content = copy(context.content)
        output = rx.fragment(
            rx.card(
                rx.text("Context Name"),
                rx.input(
                    default_value=context.cname,
                    placeholder="Context Object Name",
                    name="cname"
                    ),
                flex="1"
            ),
            rx.card(
                rx.text("Context Content"),
                rx.text_area(
                    value=ContextState.context_content,
                    placeholder="Context Content",
                    name="content",
                    on_change=ContextState.set_context_content,
                    ),
                flex="1"
            ),
        )

        return output

    @staticmethod
    def modal_edit_buttons():
        output = rx.card(
            rx.center(
                rx.hstack(
                    rx.dialog.close(
                        rx.button("Cancel",
                            color_scheme="gray",
                            variant="soft"),
                        ),
                    rx.dialog.close(
                        rx.button("Save",
                            color_scheme="green",
                            variant="soft",
                            type="submit"),
                        ),
                    align_items="stretch"
                    )
                )
            )

        return output

    @staticmethod
    def context_edit_modal(context):
        output = ContextState.modal_edit_gen(context)

        return output


    @staticmethod
    def modal_new_gen():
        output = rx.dialog.root(
            rx.dialog.trigger(rx.button("NEW", color_scheme="orange", radius="none")),
            rx.dialog.content(
                    rx.flex(
                        rx.dialog.title("New Context Object"),
                        rx.form(
                            ContextState.modal_new_input(),
                            ContextState.modal_new_buttons(),
                            on_submit=lambda x: ContextState.context_new(x)
                        ),
                    direction="column",
                    flex="1",
                    width="100%",
                    spacing="1",
                    )
                )
            )

        return output

    @staticmethod
    def modal_new_input():
        output = rx.fragment(
            rx.card(
                rx.text("Context Name"),
                rx.input(
                    default_value="",
                    placeholder="Context Object Name",
                    name="cname"
                    ),
                flex="1"
            ),
            rx.card(
                rx.text("Context Content"),
                rx.text_area(
                    default_value="",
                    placeholder="Context Content",
                    name="content"
                    ),
                flex="1"
            ),
        )

        return output

    @staticmethod
    def modal_new_buttons():
        output = rx.card(
            rx.center(
                rx.hstack(
                    rx.dialog.close(
                        rx.button("Cancel",
                            color_scheme="gray",
                            variant="soft"),
                        ),
                    rx.dialog.close(
                        rx.button("Save",
                            color_scheme="green",
                            variant="soft",
                            type="submit"),
                        ),
                    align_items="stretch"
                    )
                )
            )

        return output

    #
    ###

class ContextColumn(rx.ComponentState):
    ###
    #
    @staticmethod
    def render_context(context: ContextObject):
        output = rx.card(
            rx.flex(
                rx.vstack(
                    rx.hstack(
                        rx.heading(f"Name: {context.cname}", size="3"),
                        rx.spacer(),
                        rx.heading(f"ID: {context.cid}", size="3"),
                        flex="1",
                        width="100%",
                    ),
                    rx.markdown(
                        context.content,
                        flex="1",
                        width="100%",
                        **CONTEXT_STYLE,
                        **GLOBAL_MESSAGE_STYLE
                        ),
                    rx.hstack(
                        rx.spacer(),
                        ContextState.context_edit_modal(context),
                        rx.button("DELETE",
                            disabled=rx.cond(context.cid > 0, False, True),
                            on_click=ContextState.context_delete(context.cid)),
                        flex="1",
                        width="100%"
                        ),
                    rx.box(id=f"context-{context.cid}"),
                    flex="1",
                    width="100%"
                    ),
                flex="1",
                width="100%"
                ),
            )

        return output

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = rx.flex(
            rx.card(
                rx.hstack(
                    rx.heading(f"Context Objects", size="3"),
                    rx.spacer(),
                    ContextState.modal_new_gen(),
                    rx.button(
                        "CLEAR",
                        color_scheme="orange",
                        radius="none",
                        on_click=ContextState.context_clear
                        ),
                    ),
                ),
            rx.card(
                rx.scroll_area(
                    rx.flex(
                        rx.cond(ContextState.context_list.length(),
                            rx.foreach(ContextState.context_list, ContextColumn.render_context),
                            ContextColumn.render_context(ContextObject.default()),
                        ),
                        direction="column",
                        spacing="1",
                        overflow="hidden"
                    ),
                    type="auto",
                    scrollbars="both",
                    max_height="70vh",
                    id="context_render"
                    ),
                ),
            width="100%",
            spacing="1",
            direction="column"
            )

        return output

    #
    ###

#
###

###
#
class SessionState(rx.State):
    #session: SessionObject = DEFAULT_SESSION
    session_list: list[SessionObject] = []
    session_name: str = ""
    session_key: str = ""

    ###
    #
    def reset_session(self):
        pass


    def save_session(self, data):
        state = {}

        #raise NotImplementedError()

        state = GlobalState.bundle_state(self)

        state["user_input"] = state["user_input_state"].message_list
        #user_input = get_substate(self, "user_input_n1")
        #state["user_input"] = [z.dict() for z in user_input.message_list]

        state["system_output"] = state["system_output_state"].message_list
        #system_output = get_substate(self, "system_output_n1")
        #state["system_output"] = [z.dict() for z in system_output.message_list]

        state["system_prompt"] = state["system_prompt_state"].system_prompt
        #system_prompt = get_substate(self, "system_prompt_column_n1")
        #state["system_prompt"] = system_prompt.system_prompt.dict()

        state["plan"] = state["plan_state"].plan
        #plan = get_substate(self, "plan_output_n1")
        #state["plan"] = plan.plan

        state["context"] = state["context_state"].context_list
        #context = get_substate(self, "context_column_n1")
        #state["context"] = [z.dict() for z in context.context_list]

        if self.session_key != "":
            previous = MONGO_DB().sessions.find_one({"key":self.session_key})
            _id = previous.pop("_id")
            session = SessionObject(**previous)
            session.sname = self.session_name
            session.state = state
            session.updated = str(int(time.time()))
            # update obj
            MONGO_DB().sessions.replace_one(filter={"_id":_id}, replacement=session.dict())

        else:
            session = SessionObject.create(sname=self.session_name, state=state)
            MONGO_DB().sessions.insert_one(session.dict())
            self.session_key = session.key

        yield


    def load_session(self, key):
        obj = MONGO_DB().sessions.find_one({"key":key})
        obj.pop("_id")

        session = SessionObject(**obj)
        self.session_name = session.sname
        self.session_key = session.key

        state = GlobalState.bundle_state(self)
        state["user_input_state"].message_list = [Message(**z) for z in session.state["user_input"]]
        state["system_output_state"].message_list = [Message(**z) for z in session.state["system_output"]]
        state["system_prompt_state"].system_prompt = SystemPrompt(**session.state["system_prompt"])
        state["plan_state"].plan = Plan(**session.state["plan"])
        state["context_state"].context_list = [ContextObject(**z) for z in session.state["context"]]
        yield


    def clone_session(self, key):
        obj = MONGO_DB().sessions.find_one({"key":key})
        obj.pop("_id")

        session = SessionObject(**obj)
        session.key = str(uuid.uuid4())
        session.sname = f"{session.sname}_clone"

        MONGO_DB().sessions.insert_one(session.dict())
        self.session_name = session.sname
        self.session_key = session.key

        state = GlobalState.bundle_state(self)
        state["user_input_state"].message_list = [Message(**z) for z in session.state["user_input"]]
        state["system_output_state"].message_list = [Message(**z) for z in session.state["system_output"]]
        state["system_prompt_state"].system_prompt = SystemPrompt(**session.state["system_prompt"])
        state["plan_state"].plan = Plan(**session.state["plan"])
        state["context_state"].context_list = [ContextObject(**z) for z in session.state["context"]]
        yield


    def delete_session(self, key):
        if key == self.session_key:
            self.session_key = ""

        MONGO_DB().sessions.delete_one({"key":key})
        yield

    #
    ###

    ###
    #
    @rx.background
    async def get_sessions(self):
        #
        # TODO
        # => this is bad
        #

        async def _update():
            async with self:
                keys = [s.key for s in self.session_list]

            _keys = list()
            for session in MONGO_DB().sessions.find():
                session.pop("_id")
                if s := session.get("key"):
                    if s not in keys:
                        session = SessionObject(**session)
                        async with self:
                            self.session_list.append(session)

                    else:
                        async with self:
                            for (n, p) in enumerate(self.session_list):
                                if p.key == s:
                                    if session["updated"] != p.updated:
                                        self.session_list[n] = SessionObject(**session)
                                        break

                    _keys.append(s)

                else:
                    MONGO_DB().sessions.DeleteOne({"_id":session._id})

            for k in keys:
                if k not in _keys:
                    async with self:
                        for s in self.session_list:
                            if s.key == k:
                                self.session_list.remove(s)
                                break


        while True:
            await _update()
            await asyncio.sleep(1)

        """
        # does not handle updates
        l = MONGO_DB.sessions.count_documents({})
        while True:
            q = MONGO_DB.sessions.count_documents({})
            if l != q:
                await _update()
                l = q

            else:
                await asyncio.sleep(1)
        """

        """
        # this locks up for some reason
        with MONGO_DB.sessions.watch() as stream:
            for _ in stream:
                await _update()
        """


    #
    ###

class SessionColumn(rx.ComponentState):
    ###
    #
    @staticmethod
    def render_session(session):

        output = rx.card(
            rx.hstack(
                rx.text(session.sname),
                rx.spacer(),
                rx.button("Load",
                    on_click=lambda: SessionState.load_session(session.key)
                    ),
                rx.button("Clone",
                    on_click=lambda: SessionState.clone_session(session.key)
                    ),
                rx.button("Delete",
                    on_click=lambda: SessionState.delete_session(session.key),
                    color_scheme="red"
                    ),
                ),
            width="100%"
            )

        return output
        rx.text(f"SESSION: {session.sname}")

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = rx.flex(
            rx.card(
                rx.hstack(
                    rx.heading(f"Session Settings", size="3"),
                    rx.spacer(),
                    rx.button(
                        "IMPORT",
                        color_scheme="orange",
                        radius="none"
                    ),
                    rx.button(
                        "EXPORT",
                        color_scheme="orange",
                        radius="none"
                    ),
                    rx.button(
                        "RESET SESSION",
                        color_scheme="orange",
                        radius="none",
                        on_click=GlobalState.reset_all
                    ),
                ),
            ),
            rx.card(
                rx.form(
                    rx.hstack(
                        rx.vstack(
                            rx.text(f"Session Name: "),
                            rx.input(
                                value=SessionState.session_name,
                                placeholder="Session Name",
                                name="sname",
                                width="100%",
                                on_change=SessionState.set_session_name),
                            rx.cond(SessionState.session_key != "",
                                rx.text(f"Session Key: {SessionState.session_key}"),
                                rx.text(f"Session Key: None")
                                ),
                            width="75%"
                        ),
                        rx.spacer(),
                        rx.button("Save",
                            color_scheme="green",
                            variant="soft",
                            type="submit"),
                        align_items="stretch",
                    ),
                on_submit=lambda x: SessionState.save_session(x)
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.heading(f"Load Session", size="3"),
                    rx.cond(SessionState.session_list.length(),
                        rx.foreach(SessionState.session_list, cls.render_session),
                        rx.card(f"NO SESSIONS", width="100%")
                        ),
                    ),
                ),
            width="100%",
            spacing="1",
            direction="column"
            )

        return output

    #
    ###

BACKGROUND_TASKS.append(SessionState.get_sessions)


#
###

###
#
class StateColumn(rx.ComponentState):

    @classmethod
    def get_component(cls, **props):
        output = rx.flex(
            rx.card(rx.heading("Global State", size="3")),
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("System", value="system"),
                    rx.tabs.trigger("Plan", value="plan"),
                    rx.tabs.trigger("Context", value="context"),
                    rx.tabs.trigger("Session", value="session")
                    ),
                rx.tabs.content(
                    SystemPromptColumn.create(),
                    value="system"
                    ),
                rx.tabs.content(
                    PlanColumn.create(),
                    value="plan"
                    ),
                rx.tabs.content(
                    ContextColumn.create(),
                    value="context"
                    ),
                rx.tabs.content(
                    SessionColumn.create(),
                    value="session"
                    ),
                default_value="system"
                ),
            width="100%",
            spacing="1",
            direction="column"
            )

        return output


#
###


###
#
class Content(rx.ComponentState):
    #
    #
    #

    ###
    #

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = rx.grid(
            UserInput.create(),
            SystemOutput.create(),
            StateColumn.create(),
            columns="3",
            spacing="4",
            width="100%",
            flex="1",
            )
        return output

    #
    ###

#
###
