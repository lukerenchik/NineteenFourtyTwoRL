from pyboy import PyBoy

pyboy = PyBoy("nineteenFT.gbc")

def press_and_release_button(button, press_duration=5):
    pyboy.button_press(button)
    for _ in range(press_duration):
        pyboy.tick()
    pyboy.button_release(button)

#Get to Title Screen
for _ in range(1000):
    pyboy.tick()

press_and_release_button("start")

for _ in range(300):
    pyboy.tick()

press_and_release_button("a")

for _ in range(200):
    pyboy.tick()

press_and_release_button("a")

for _ in range(200):
    pyboy.tick()


with open("start_of_game.state", "wb") as f:
    pyboy.save_state(f)

    

press_and_release_button("start")

#Game is Started & Paused



for _ in range(500):
    press_and_release_button("a")


pyboy.stop()


