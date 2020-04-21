from time import clock
from collections import defaultdict

timers = defaultdict(lambda: 0)

def start_timer():
    return clock()

def end_timer(start, name=None):
    end = clock()
    elapsed = end - start

    if name != None:
        timers[name] += elapsed
    
    return elapsed

def print_timers():
    print('outputing times')
    for key, value in timers.items():
        print(key, 'took', value, 'seconds')
    print("times_predicted =", times_predicted)
    print("predicted_actions =", predicted_actions)

times_predicted = 0
predicted_actions = 0


use_cupy = True

def get_if_cupy(value):
    if use_cupy:
        return value.get()
    else:
        return value