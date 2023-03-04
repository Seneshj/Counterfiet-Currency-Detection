import identify as i
import fake_classifier as f
import detect_note as d

while True:
    option = int(input("Enter one of the below options: \n1. Identify currency\n2. Identify Counterfeit note\n3. Sum "
                       "up currency\n4. Exit\n\n->"))
    if option == 1:
        if d.detect_note():
            print("Image found")
            i.main()
            print(i.get_value())
    elif option == 2:
        print("Image found")
        if d.detect_note():
            i.main()
            i.voice_out(f.fake_classifier(i.get_value()))
    elif option == 3:
        i.voice_out("Note done")
    elif option == 4:
        i.voice_out("Have a nice day!")
        break
