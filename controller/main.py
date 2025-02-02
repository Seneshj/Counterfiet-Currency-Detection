import identify as i
import fake_classifier as f
import detect_note as d
import utility

while True:
    option = 0
    try:
        option = int(
            input("Enter one of the below options: \n1. Identify currency\n2. Identify Counterfeit note\n3. Sum "
                  "up currency\n4. Exit\n\n->"))
    except Exception as e:
        print(e)

    if option == 1:
        if d.detect_note():
            print("Image found")
            i.main()
            print(i.get_value())
    elif option == 2:
        print("Image found")
        if d.detect_note():
            i.main()
            utility.voice_out(f.fake_classifier(i.get_value()))
    elif option == 3:
        utility.voice_out("Note done")
    elif option == 4:
        utility.voice_out("Have a nice day!")
