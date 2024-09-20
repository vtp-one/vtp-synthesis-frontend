import reflex as rx

###
#


#
###

###
#
class Footer(rx.ComponentState):
    content: str = "Footer Content"

    ###
    #

    #
    ###

    ###
    #
    @classmethod
    def get_component(cls, **props):
        output = rx.hstack(
            rx.box(cls.content),
            position="sticky",
            bottom="0",
            left="0",
            align_items="stretch",
            width="100%",
            background_color=rx.color("mauve", 1),
            )

        return output

    #
    ###

#
###