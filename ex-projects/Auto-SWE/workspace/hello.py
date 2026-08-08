def say_hello(name):
    # Bug: using + with an int
    print("Hello " + name + ", you are " + 25 + " years old.")

if __name__ == "__main__":
    say_hello("Alice")
