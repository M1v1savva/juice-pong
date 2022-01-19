from pynput import keyboard
from coordinate_processor import throw_to_coordinates

cur_user = False

def on_press(key):
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k in ['enter']:  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        global cur_user
        cur_user ^= True
        print('player switched')
        print('current turn: human' if cur_user else 'current turn: robot')

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread

if cur_user == False:
    coordinates = (600, 400)#coordinates retrieval
    print('thrown')
    throw_to_coordinates(coordinates)