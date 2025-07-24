import random
def route_ab(split=0.5):
    return "A" if random.random() < split else "B"
