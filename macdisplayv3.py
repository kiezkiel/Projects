import pwnagotchi.plugins as plugins
import subprocess
import logging
import time
import re

class MacDisplay(plugins.Plugin):
    __author__ = "your_name"
    __version__ = "1.3"
    __license__ = "GPL3"
    __description__ = "Changes MAC and displays it on screen"

    def __init__(self):
        self.text_to_set = ""
        self.shown = False

    def on_loaded(self):
        logging.info("[macdisplay] Plugin loaded")
        if 'face' not in self.options:
            self.options['face'] = "^_^"

        try:
            subprocess.run(["ip", "link", "set", "wlan0", "down"], check=True)
            output = subprocess.run(["macchanger", "-r", "wlan0"], capture_output=True, text=True, check=True).stdout
            subprocess.run(["iw", "dev", "wlan0", "set", "type", "monitor"], check=True)
            subprocess.run(["ip", "link", "set", "wlan0", "up"], check=True)

            old_mac = re.search(r"Permanent MAC: ([\w:]+)", output)
            new_mac = re.search(r"New MAC: ([\w:]+)", output)

            if old_mac and new_mac:
                self.text_to_set = f"Old MAC: {old_mac.group(1)}\nNew MAC: {new_mac.group(1)}"
            else:
                self.text_to_set = "MAC changed, but couldn't parse result."

            logging.info("[macdisplay] MAC updated")

        except subprocess.CalledProcessError as e:
            self.text_to_set = f"MAC change failed:\n{e}"
            logging.error(f"[macdisplay] Error: {e}")

    def on_ui_update(self, ui):
        if self.text_to_set and not self.shown:
            ui.set('face', self.options['face'])
            ui.set('status', self.text_to_set)
            self.shown = True
