import reflex as rx

###
#
from .settings import Settings
from ..state import GlobalState

#
###

###
#
class NavBar(rx.ComponentState):
    #
    #
    #

    ###
    #
    @staticmethod
    def render_left() -> rx.Component:
        output = rx.hstack(
            rx.button(
                "⚹",
                color_scheme="orange",
                radius="none",
                on_click=GlobalState.refresh
                ),
            align_items="center",
            flex_grow="0",
            )

        return output

    @staticmethod
    def render_right() -> rx.Component:
        output = rx.hstack(
            Settings.create(),
            align_items="center",
            flex_grow="0",
            )

        return output

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props) -> rx.Component:
        output = rx.hstack(
            cls.render_left(),
            rx.spacer(align="stretch"),
            cls.render_right(),
            justify_content="space-between",
            align_items="center",
            )

        return output

    #
    ###

#
###