from pwnagotchi.ui.components import Text
from pwnagotchi.ui.view import BLACK
import pwnagotchi.plugins as plugins
import pwnagotchi
import subprocess
import threading
import logging
import time
import os

LOG_FILE = "/home/pi/macdisplay.log"

logging.basicConfig(level=logging.INFO)

def log_to_file(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

def run_and_log(command):
    try:
        log_to_file(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        log_to_file(f"Output:\n{result.stdout.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log_to_file(f"Error:\n{e.stderr.strip()}")
        raise

class MacDisplay(plugins.Plugin):
    __author__ = "your_name"
    __version__ = "1.2"
    __license__ = "GPL3"
    __description__ = "Changes MAC, sets monitor mode, logs output, and displays MAC info on boot with face."

    def __init__(self):
        self.mac_text = ""
        self.displayed = False
        self.failed = False

    def on_loaded(self):
        logging.info("[macdisplay] Plugin loaded.")
        try:
            run_and_log(["ip", "link", "set", "wlan0", "down"])
            output = run_and_log(["macchanger", "-r", "wlan0"])
            run_and_log(["iw", "dev", "wlan0", "set", "type", "monitor"])
            run_and_log(["ip", "link", "set", "wlan0", "up"])

            # Parse MAC output
            lines = output.splitlines()
            old_mac = next((l for l in lines if "Permanent MAC" in l), "")
            new_mac = next((l for l in lines if "New MAC" in l), "")
            self.mac_text = f"{old_mac}\n{new_mac}" if old_mac and new_mac else "MAC change failed"
            self.failed = False

        except subprocess.CalledProcessError as e:
            self.mac_text = f"MAC setup failed:\n{e}"
            logging.error(f"[macdisplay] MAC setup error: {e}")
            self.failed = True

        log_to_file(f"Final MAC text: {self.mac_text}")

    def on_ui_setup(self, ui):
        ui.add_element(
            "mac_display",
            Text(
                color=BLACK,
                value="",
                position=(5, 5),
                font=pwnagotchi.ui.fonts.Small
            )
        )

    def on_ready(self, ui):
        if not self.displayed:
            self.displayed = True

            if self.failed:
                pwnagotchi.face.set("(╯°□°）╯︵ ┻━┻")
            else:
                pwnagotchi.face.set("^_^")

            ui.set("mac_display", self.mac_text)

            def hide_after_delay():
                time.sleep(10)
                with ui._lock:
                    ui.set("mac_display", "")

            threading.Thread(target=hide_after_delay).start()
