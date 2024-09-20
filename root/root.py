import reflex as rx

###
#
from .synthesis import NavBar, Content, Footer

from .state import GlobalState, BACKGROUND_TASKS

#
###

###
#
@rx.page(title="VTP-SYNTHESIS", on_load=BACKGROUND_TASKS)
def index() -> rx.Component:
    output = rx.vstack(
        NavBar.create(),
        rx.divider(size="4", color="orange", decorative=True),
        Content.create(),
        #rx.spacer(align="stretch"),
        rx.divider(size="4", color="orange", decorative=True),
        Footer.create(),
        background_color=rx.color("gray", 1),
        color=rx.color("gray", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="1",
        padding="5px",
        )

    return output

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="orange",
    ),
)

#
###
