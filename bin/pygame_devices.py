#!/usr/bin/env python

import pygame
import pygame._sdl2.audio as sdl2_audio
import time

def get_devices(capture_devices: bool = False) -> tuple[str, ...]:
    pygame.mixer.init()
    devices = tuple(sdl2_audio.get_audio_device_names(capture_devices))
    pygame.mixer.quit()
    return devices

def main():
    # List audio devices
    devices = get_devices()
    print("Available audio devices:", devices)

    # Choose a device and initialize the mixer with it
    chosen_device = devices[0]  # Replace with your chosen device
    pygame.mixer.init(devicename=chosen_device)

    # Cleanup
    pygame.mixer.music.unload()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()

