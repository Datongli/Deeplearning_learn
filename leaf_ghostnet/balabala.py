

def count(num):
    num = num + 1
    global __apple__
    __apple__ = __apple__ + 100
    return num

if __name__ == '__main__':
    __apple__ = 0
    xiejj = 0
    xiejj = count(xiejj)
    print(xiejj)
    print(__apple__)