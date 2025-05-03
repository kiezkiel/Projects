import pwnagotchi.plugins as plugins
import subprocess
import logging
import time
import re

LOG_FILE = "/home/pi/file.log"

def log_to_file(message):
    with open(LOG_FILE, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} {message}\n")

def run_and_log(command, capture_output=False):
    log_to_file(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=capture_output,
            text=True
        )
        if capture_output:
            log_to_file(f"Output:\n{result.stdout.strip()}")
            return result.stdout.strip()
        return None
    except subprocess.CalledProcessError as e:
        log_to_file(f"Error:\n{e.stderr.strip() if e.stderr else str(e)}")
        raise

class MacDisplay(plugins.Plugin):
    __author__ = "your_name"
    __version__ = "1.4"
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
            run_and_log(["ip", "link", "set", "wlan0", "down"])
            output = run_and_log(["macchanger", "-r", "wlan0"], capture_output=True)
            run_and_log(["iw", "dev", "wlan0", "set", "type", "monitor"])
            run_and_log(["ip", "link", "set", "wlan0", "up"])

            old_mac = re.search(r"Permanent MAC: ([\w:]+)", output)
            new_mac = re.search(r"New MAC: ([\w:]+)", output)

            if old_mac and new_mac:
                self.text_to_set = f"Old MAC: {old_mac.group(1)}\nNew MAC: {new_mac.group(1)}"
            else:
                self.text_to_set = "MAC changed, but couldn't parse result."

            logging.info("[macdisplay] MAC updated")
            log_to_file(f"MAC Display: {self.text_to_set}")

        except subprocess.CalledProcessError as e:
            self.text_to_set = f"MAC change failed:\n{e}"
            logging.error(f"[macdisplay] Error: {e}")
            log_to_file(f"[macdisplay] Exception occurred: {e}")

    def on_ui_update(self, ui):
        if self.text_to_set and not self.shown:
            ui.set('face', self.options['face'])
            ui.set('status', self.text_to_set)
            self.shown = True
