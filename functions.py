def press_and_release_button(button, press_duration=5):
    pyboy.button_press(button)
    for _ in range(press_duration):
        pyboy.tick()
    pyboy.button_release(button)