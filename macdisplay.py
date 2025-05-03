from pwnagotchi.ui.components import Text
from pwnagotchi.ui.view import BLACK
import pwnagotchi.plugins as plugins
import pwnagotchi
import threading
import time
import subprocess
import logging

class MacDisplay(plugins.Plugin):
    __author__ = "your_name"
    __version__ = "1.0"
    __license__ = "GPL3"
    __description__ = "Displays MAC address info for 10 seconds on boot"

    def __init__(self):
        self.mac_text = ""
        self.displayed = False

    def on_loaded(self):
        logging.info("[macdisplay] Plugin loaded.")

        try:
            # Execute macchanger command and capture output
            result = subprocess.run(
                ["macchanger", "-r", "wlan0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            output = result.stdout.strip()
            logging.info(f"[macdisplay] macchanger output:\n{output}")

            # Parse MAC addresses
            lines = output.splitlines()
            old_mac = next((l for l in lines if "Permanent MAC" in l), "")
            new_mac = next((l for l in lines if "New MAC" in l), "")
            self.mac_text = f"{old_mac}\n{new_mac}" if old_mac and new_mac else "MAC change failed"

        except Exception as e:
            self.mac_text = f"Error: {str(e)}"
            logging.error(f"[macdisplay] Failed to get MAC address: {e}")

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
            ui.set("mac_display", self.mac_text)

            # Start thread to hide text after 10 seconds
            def hide_after_delay():
                time.sleep(10)
                with ui._lock:
                    ui.set("mac_display", "")

            threading.Thread(target=hide_after_delay).start()
