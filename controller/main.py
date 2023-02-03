import arduino


user_input = input("--Select an option from below--\n1.Detect currency\n2.Identify fake\n->")

if user_input == "1":
    pass
elif user_input == "2":
    arduino.on_uv_light()

