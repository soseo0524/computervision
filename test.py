def test(a,b=10, *c, d=20, e=30):
    print(a, b, c, d, e)

test(1,2,3,4,e=99)


def another_test(x, y=5, *args, z=15):
    print(x, y, args, z)


    