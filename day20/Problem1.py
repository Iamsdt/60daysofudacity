
num = int(input())

for _ in range(num):
    seat = int(input())
    s = seat % 12
    if s == 0:
        seat -= 11
    elif s < 7:
        seat += (2 * (6 - s) + 1)
    else:
        seat -= (2 * (s - 7) + 1)

    r = seat % 6

    if r == 1 or r == 0:
        print(seat, "WS")
    elif r == 2 or r == 5:
        print(seat, "MS")
    else:
        print(seat, "AS")
