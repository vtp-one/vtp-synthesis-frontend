import rich
import reflex as rx
import os
import time
import asyncio

#from openai import OpenAI
import ollama

from .synthesis import utils

BACKGROUND_TASKS = []

###
#
class GlobalState(rx.State):
    namespace_list: list[str] = [
        "system_output_state",
        "user_input_state",
        "system_prompt_state",
        "plan_state",
        "context_state",
        ]

    ###
    #
    def refresh(self):
        yield GlobalState.reset_all

        return rx.call_script("location.reload(true);")

    def update(self, data):

        return

    #
    ###

    ###
    #
    @staticmethod
    def bundle_state(parent):
        global_state = utils.get_substate(parent, "global_state")

        state = {"global_state":global_state}
        for namespace in global_state.namespace_list:
            state[namespace] = utils.get_substate(parent, namespace)

        return state

    #
    ###

    @rx.background
    async def reset_all(self):
        async with self:
            client_token = self.router.session.client_token
            substates = rx.state.get_state_manager().states[client_token].substates

            for state in substates.values():
                state._reset_client_storage()
                state.reset()

            self._reset_client_storage()
            self.reset()

    @rx.background
    async def scroll_chat(self):
        #previous = {"user_input":None, "system_output":None}
        #processing = "user_input_state"
        #target = ["user_input", "system_output"]

        while True:
            async with self:
                client_token = self.router.session.client_token
                substates = rx.state.get_state_manager().states[client_token].substates

            flag = substates.get(processing, False).processing

            #
            # TODO
            # - this is stupid, but the correct way isn't updating properly
            # - these should be on the state object anyways,
            # - but that doesn't work correctly when using
            # - the statecomponent object or something
            #
            #

            if flag:
                for key in target:
                    _state = f"{key}_n1"
                    if state := substates.get(_state):
                        message_list = state.message_list
                        if len(message_list):
                            current = message_list[-1]
                            div_target = f"{key}-{current.timestamp}"
                            yield rx.scroll_to(div_target)

            """
            async with self:
                client_token = self.router.session.client_token
                substates = rx.state.get_state_manager().states[client_token].substates

                for key, _previous in previous.items():
                    state_target = f"{key}_n1"
                    update = False
                    if state := substates.get(state_target, None):
                        message_list = getattr(state, "message_list", [])
                        if len(message_list):
                            current = message_list[-1]

                            if _previous is None:
                                update = True

                            else:
                                for k,v in current.dict().items():
                                    if v != _previous.dict().get(k):
                                        update = True

                    if update:
                        div_target = f"{key}-{current.timestamp}"
                        rich.print(f"SCROLL TARGET: {div_target}")
                        previous[key] = current
                        yield rx.scroll_to(div_target)
            """

            await asyncio.sleep(1e-1)


#BACKGROUND_TASKS.append(GlobalState.scroll_chat)

#
###