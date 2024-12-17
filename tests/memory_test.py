from pyboy import PyBoy

pyboy = PyBoy("nineteenFT.gbc")

def press_and_release_button(button, press_duration=5):
    pyboy.button_press(button)
    for _ in range(press_duration):
        pyboy.tick()
    pyboy.button_release(button)

with open("start_of_game.state", "rb") as f:
    pyboy.load_state(f)

    for _ in range(250):
        pyboy.tick()

    press_and_release_button("left")
    press_and_release_button("left")
    pyboy.button_press("b")

    for _ in range(100):
        pyboy.tick()

    current_score = 50
    memory = pyboy.memory_scanner.scan_memory(current_score, start_addr=0xC000, end_addr=0xDFFF)
    print(memory)
    pyboy.tick(1, True)


