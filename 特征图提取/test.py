"""
该文件常用于尝试，什么都可以尝试

"""
def tenTotwo(number):
    #定义栈
    s = []
    binstring = ''
    while number > 0:
        #余数进栈
        rem = number % 2
        s.append(rem)
        number = number // 2
    while len(s) > 0:
        #元素全部出栈即为所求二进制数
        binstring = binstring + str(s.pop())
    print(binstring)


if __name__ == '__main__':
    n = int(input())
    num_ten = []
    for i in range(n - 1):
        num_ten.append(eval(input()))
    for i in range(n -1):
        tenTotwo(num_ten[i])


