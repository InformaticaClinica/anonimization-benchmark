import threading

def test0():
    test1()
    test2()
    test3()

def test1():
    print("Hello, World!")

def test2():
    while True:
        pass

def test3():
    print("Hello, World! 3")

def main():
    for i in range(10):
        t = threading.Thread(target=test0)
        t.start()
        t.join(10)
        # test1()
        # test2()
        # test3()
        


if __name__ == "__main__":
    main()