from chat import chat, clear_memory
print("Financial Support chatbot")
print("Commands: 'exit' to quit, 'clear' to reset memory")    
while True:
    try:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            print("\nBye Bye ;)")
            break
        
        if user_input.lower() == "clear":
            clear_memory()
            print("\nMemory nuked!. Who are you again?\n")
            continue
        
        answer = chat(user_input)
        print(f"\nAI: {answer}\n")
        
    except KeyboardInterrupt:
        print("\n\nBye BYe ;)")
        break
