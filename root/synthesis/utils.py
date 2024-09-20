import reflex as rx

###
#
def get_substate(root, target):
    client_token = root.router.session.client_token
    if substate := rx.state.get_state_manager().states[client_token].substates.get(target):
        return substate

    else:
        e = f"SUBSTATE NOT FOUND: {target} - substates: {rx.state.get_state_manager().states[client_token].substates.keys()}"
        raise RuntimeError(e)

#
###