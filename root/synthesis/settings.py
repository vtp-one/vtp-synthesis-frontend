import reflex as rx

###
#
from ..state import GlobalState

#
###

###
#
class Settings(rx.ComponentState):
    #
    #
    #

    ###
    #
    def modal_create(self):

        return Settings.get_component()

    #
    ###

    ###
    #
    @staticmethod
    def modal_dialog():
        output = rx.dialog.root(
            rx.dialog.trigger(rx.button("⛺", color_scheme="orange", radius="none")),
            rx.dialog.content(
                rx.dialog.title("Settings"),
                rx.dialog.description("Configure Settings"),
                Settings.modal_form()
                )
            )

        return output

    @staticmethod
    def modal_form():
        output = rx.form(
            Settings.modal_input(),
            Settings.modal_buttons(),
            on_submit=lambda x: GlobalState.update(x)
            )

        return output

    @staticmethod
    def modal_input():
        #
        # TODO
        # - generate these inputs based on GlobalState.__fields__
        #
        output = rx.vstack(
            rx.card(
                rx.text("Setting 1"),
                rx.input(
                    default_value="1",
                    placeholder="1",
                    name="1"
                    ), # input
                ),
            rx.card(
                rx.text("Setting 2"),
                rx.input(
                    default_value="2",
                    placeholder="2",
                    name="2"
                    ), # input
                ),
            rx.card(
                rx.text("Setting 3"),
                rx.input(
                    default_value="3",
                    placeholder="3",
                    name="3"
                    )
                )
            )

        return output

    @staticmethod
    def modal_buttons():
        output = rx.hstack(
            rx.spacer(align="stretch"),
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
            rx.spacer(align="stretch"),
            align_items="stretch"
            )
        return output


    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = cls.modal_dialog()

        return output

    #
    ###

#
###
