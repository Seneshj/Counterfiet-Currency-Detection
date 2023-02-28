import identify as i


while True:
    option = int(input("Enter one of the below options: \n1. Identify currency\n2. Identify Counterfeit note\n3. Sum "
                       "up currency\n4. Exit\n\n->"))
    if option == 1:
        i.main()
        print(i.get_value())
    elif option == 2:
        i.voice_out("Not done")
    elif option == 3:
        i.voice_out("Note done")
    elif option == 4:
        i.voice_out("Have a nice day!")
        break
